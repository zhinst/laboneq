# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Controller container for the remote controller service.

Manages the lifecycle of the single controller session behind the service:
connecting once at startup, queuing experiments, and cleaning up.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from typing import TYPE_CHECKING, Any, TypedDict

from laboneq.controller.api.async_local_controller import AsyncLocalController
from laboneq.controller.api.commons import SubmissionHandle

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from laboneq.controller.api.controller_api import SubmissionStatus
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.result.results import Results

logger = logging.getLogger(__name__)

# Sentinel for "no fingerprint available" (device setup has no system_description).
_NO_FINGERPRINT = None


def load_callbacks_from_module(module_path: str) -> dict[str, Callable[..., Any]]:
    """Load callback functions from a Python module.

    If the module defines a ``__callbacks__`` list, only those names are
    registered.  Otherwise all public *functions* (not classes or modules)
    defined in the file are registered.

    Args:
        module_path: Path to the Python module file.

    Returns:
        Dictionary mapping callback names to functions.
    """
    spec = importlib.util.spec_from_file_location("user_callbacks", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Explicit list takes priority — register exactly those names.
    explicit_list: list[str] | None = getattr(module, "__callbacks__", None)
    if explicit_list is not None:
        callbacks: dict[str, Callable[..., Any]] = {}
        for name in explicit_list:
            obj = getattr(module, name, None)
            if obj is None or not callable(obj):
                raise ImportError(
                    f"__callbacks__ lists {name!r} but it is not a callable "
                    f"attribute of {module_path}"
                )
            callbacks[name] = obj
            logger.info("Registered callback: %s", name)
        return callbacks

    # Fallback: register public functions defined in the module itself
    # (skip imports, classes, and modules).
    callbacks = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            callbacks[name] = obj
            logger.info("Registered callback: %s", name)

    return callbacks


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------


class CallbackInfoDict(TypedDict):
    """Typed dictionary for callback introspection data."""

    signature: str
    docstring: str | None


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------


def _build_scm_address(host: str, port: str) -> str:
    """Build an ``http://host:port`` URL from bare *host* and *port*.

    Brackets IPv6 addresses for valid URL syntax.
    """
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}" if port else f"http://{host}"


def _apply_dataserver(
    device_setup: DeviceSetup, dataserver: tuple[str, str] | None
) -> None:
    """Override the LabOne dataserver entries of *device_setup* with *dataserver*.

    Classic (QCCS / Gen 2) setups reference a dataserver by UID, with the
    host/port stored on the :class:`DataServer` entry — this overrides them.
    Gen 4 / ZQCS setups carry the SCM address on the instrument itself.

    No-op when no dataserver was configured on the service.
    """
    from laboneq.dsl.device.instruments.zqcs import ZQCS

    if dataserver is None:
        return
    host, port = dataserver
    for server in device_setup.servers.values():
        server.host = host
        server.port = port
    scm_address = _build_scm_address(host, port)
    for instrument in device_setup.instruments:
        if isinstance(instrument, ZQCS):
            instrument.address = scm_address


def _auto_create_device_setup(dataserver: tuple[str, str]) -> DeviceSetup:
    """Create a minimal DeviceSetup from a dataserver (SCM) address.

    When ``--devicesetup`` is not provided, the ``--dataserver`` argument is
    assumed to be the SCM HTTP address.  We only need a single
    ``ZQCS`` instrument pointing to it; the controller
    discovers all hardware properties during connection.

    Args:
        dataserver: ``(host, port)`` tuple from the CLI.  The host may already
            be a full ``http://`` URL; if not, the ``http://`` scheme is
            prepended.

    Returns:
        A :class:`~laboneq.dsl.device.DeviceSetup` ready for connection.
    """
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.device.instruments.zqcs import ZQCS

    address = _build_scm_address(*dataserver)
    setup = DeviceSetup(uid="auto-discovered")
    setup.add_dataserver(host="localhost", port=-1)  # placeholder; not used by SCM
    setup.add_instruments(ZQCS("device", address=address))
    return setup


# ---------------------------------------------------------------------------
# ControllerContainer
# ---------------------------------------------------------------------------


_CREATE_TOKEN: Any = object()


class ControllerContainer:
    """Manages a single controller session for the remote service.

    Construct via :meth:`create` — the classmethod performs the hardware
    connection and returns a ready-to-use container.  Direct instantiation
    raises :class:`RuntimeError`; connection failures propagate so the
    service aborts startup rather than running disconnected.
    """

    def __init__(
        self,
        *,
        _token: Any = None,
        controller: AsyncLocalController,
        device_setup: DeviceSetup,
    ) -> None:
        if _token is not _CREATE_TOKEN:
            raise RuntimeError(
                "ControllerContainer must be constructed via "
                "ControllerContainer.create(...)."
            )
        self._controller = controller
        self._device_setup = device_setup

    @classmethod
    async def create(
        cls,
        *,
        neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
        device_setup: DeviceSetup | None = None,
        dataserver: tuple[str, str] | None = None,
        do_emulation: bool = False,
        reset_devices: bool = False,
    ) -> ControllerContainer:
        """Connect to hardware and return a ready-to-use container.

        Argument combinations:

        - ``device_setup`` alone — use the setup as-is (addresses from its
          ``servers`` dict, or on the instrument for ZQCS).
        - ``dataserver`` alone — Gen 4 auto-discovery: build a new ZQCS
          setup pointing at *dataserver*, download SystemDescription from the SCM
          on connect.
        - ``device_setup`` + ``dataserver`` — Gen 2 override: replace the
          dataserver entries on the setup with *dataserver*.
        - ``do_emulation`` requires an explicit ``device_setup``.

        Args:
            neartime_callbacks: Pre-registered near-time callbacks.
            device_setup: Explicit :class:`DeviceSetup`.
            dataserver: ``(host, port)`` of the hardware server.
            do_emulation: Run in emulation mode (no real hardware).
            reset_devices: Reset hardware on connect.

        Raises:
            ValueError: Invalid combination of *device_setup*, *dataserver*,
                and *do_emulation*.
            Exception: Any failure from the controller factory propagates.
        """
        if do_emulation and device_setup is None:
            raise ValueError(
                "do_emulation requires an explicit device_setup "
                "(the system description cannot be auto-discovered without hardware)."
            )
        if device_setup is None:
            if dataserver is None:
                raise ValueError("Provide either device_setup or dataserver.")
            device_setup = _auto_create_device_setup(dataserver)
        else:
            # Gen 2 setups reference a LabOne dataserver by UID; when the
            # caller supplies one, override the stored addresses with it.
            _apply_dataserver(device_setup, dataserver)

        controller = await cls._create_controller(
            device_setup=device_setup,
            neartime_callbacks=neartime_callbacks,
            do_emulation=do_emulation,
            ignore_version_mismatch=False,
            reset_devices=reset_devices,
            disable_runtime_checks=True,
            timeout=None,
        )
        logger.info(
            "Connected to hardware (emulation=%s, reset_devices=%s)",
            do_emulation,
            reset_devices,
        )
        return cls(
            _token=_CREATE_TOKEN,
            controller=controller,
            device_setup=device_setup,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _create_controller(
        device_setup: DeviceSetup,
        neartime_callbacks: dict[str, Callable[..., Any]] | None,
        do_emulation: bool,
        ignore_version_mismatch: bool,
        reset_devices: bool,
        disable_runtime_checks: bool,
        timeout: float | None,
    ) -> AsyncLocalController:
        """Default factory: convert the device setup and connect to hardware."""
        return await AsyncLocalController.create(
            device_setup=device_setup,
            neartime_callbacks=neartime_callbacks,
            do_emulation=do_emulation,
            ignore_version_mismatch=ignore_version_mismatch,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout,
        )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def device_setup(self) -> DeviceSetup:
        """Return the server's device setup."""
        return self._device_setup

    @property
    def registered_callback_names(self) -> list[str]:
        """Names of registered near-time callbacks."""
        return list(self._controller.neartime_callbacks.keys())

    @property
    def registered_callback_info(self) -> dict[str, CallbackInfoDict]:
        """Registered near-time callbacks with their signatures and docstrings."""
        return {
            name: {
                "signature": str(inspect.signature(func)),
                "docstring": inspect.getdoc(func),
            }
            for name, func in self._controller.neartime_callbacks.items()
        }

    # ------------------------------------------------------------------
    # Main operations
    # ------------------------------------------------------------------

    async def submit_experiment(
        self, experiment_id: int, scheduled_experiment: ScheduledExperiment
    ):
        """Accept an experiment for execution.

        Args:
            experiment_id: Client-supplied UUID for this experiment.
            scheduled_experiment: The compiled experiment.

        Raises:
            LabOneQControllerException: an error occurs during submission.
        """
        await self._controller.submit_experiment(
            scheduled_experiment, handle=SubmissionHandle(id=experiment_id)
        )
        logger.info("Submitted experiment %s", experiment_id)

    async def get_submission_status(self, experiment_id: int) -> SubmissionStatus:
        """Return ``status`` for *experiment_id*."""
        return await self._controller.get_experiment_status(
            SubmissionHandle(id=experiment_id)
        )

    async def get_submission_results(self, experiment_id: int) -> Results:
        """Await completion and return ``results``."""
        return await self._controller.get_experiment(SubmissionHandle(id=experiment_id))

    async def cancel_experiment(self, experiment_id: int) -> None:
        """Cancel a queued/running experiment or release a completed one."""
        await self._controller.close_submission(SubmissionHandle(id=experiment_id))
        logger.info("Cancelled experiment %s", experiment_id)
