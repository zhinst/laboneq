# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Controller container for the remote controller service.

This module manages the lifecycle of a single controller session, including
implicit connection, experiment queuing, and cleanup.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from laboneq.controller.api.async_local_controller import AsyncLocalController
from laboneq.controller.api.controller_api import SubmissionStatus
from laboneq.executor.executor import ExecutorBase, LoopingMode
from laboneq.serializers import from_json

if TYPE_CHECKING:
    from types import ModuleType

    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device import DeviceSetup

logger = logging.getLogger(__name__)

# Sentinel for "no fingerprint available" (device setup has no system_profile).
_NO_FINGERPRINT = None


# ---------------------------------------------------------------------------
# Custom exceptions — mapped to specific HTTP status codes in routes.py
# ---------------------------------------------------------------------------


class ExperimentAlreadyExistsError(Exception):
    """UUID conflict — experiment with this ID is already known.

    Maps to HTTP 400 Bad Request.
    """


class FingerprintMismatchError(Exception):
    """Experiment's device fingerprint doesn't match the server's device setup.

    Maps to HTTP 400 Bad Request.
    """


class MissingCallbacksError(Exception):
    """Experiment requires callbacks not registered on the server.

    Maps to HTTP 400 Bad Request.
    """


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
# Device setup hot-reloading
# ---------------------------------------------------------------------------


class DeviceSetupLoader:
    """Hot-reloadable DeviceSetup from a JSON file.

    Monitors the file's modification time and transparently reloads when the
    file changes on disk — no service restart required.

    Args:
        path: Path to the JSON file produced by ``laboneq.serializers.to_json``.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._last_mtime: float = 0.0
        self._setup: DeviceSetup | None = None
        self._load()

    def _load(self) -> None:
        self._last_mtime = self._path.stat().st_mtime
        with self._path.open(encoding="utf-8") as f:
            self._setup = from_json(f.read())
        logger.debug("Loaded device setup from %s", self._path)

    def get(self) -> DeviceSetup:
        """Return the current device setup, reloading from disk if modified."""
        try:
            mtime = self._path.stat().st_mtime
            if mtime != self._last_mtime:
                self._load()
                logger.info("Reloaded device setup from %s", self._path)
        except OSError:
            logger.warning("Could not stat device setup file: %s", self._path)
        return self._setup  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------


class CallbackInfoDict(TypedDict):
    """Typed dictionary for callback introspection data."""

    signature: str
    docstring: str | None


@dataclass
class SubmissionInfo:
    """Tracking information for a single experiment submission."""

    handle: Any  # Internal controller handle


@dataclass
class Tombstone:
    """Record of a submission that was force-cancelled (e.g. by admin reset).

    Tombstones survive session teardown so that clients polling for status
    after a reset see ``CANCELED`` with a reason, rather than a confusing
    ``EXPERIMENT_NOT_FOUND`` error.
    """

    status: SubmissionStatus
    error: str


#: A submission is either live (has a controller handle) or tombstoned.
Submission = SubmissionInfo | Tombstone


@dataclass
class SessionState:
    """State of an active controller session."""

    device_config_fingerprint: str | None
    controller: AsyncLocalController


class _NeartimeCallbackCollector(ExecutorBase):
    """Walk the execution tree and collect near-time callback names."""

    def __init__(self) -> None:
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY | LoopingMode.ONCE)
        self.callbacks: set[str] = set()

    def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        self.callbacks.add(func_name)


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------


async def _create_controller(
    device_setup: DeviceSetup,
    neartime_callbacks: dict[str, Callable[..., Any]],
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
    """Overwrite server addresses in *device_setup* with *dataserver*.

    For classic (QCCS) instruments this patches the LabOne dataserver
    entries.  For ZQCS instruments the SCM address lives directly on the
    instrument's ``address`` attribute, so we overwrite that too.

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
        connection: SessionState,
        neartime_callbacks: dict[str, Callable[..., Any]],
        device_setup_loader: DeviceSetupLoader | None,
        dataserver: tuple[str, str] | None,
        discovered_device_setup: DeviceSetup | None,
    ) -> None:
        if _token is not _CREATE_TOKEN:
            raise RuntimeError(
                "ControllerContainer must be constructed via "
                "ControllerContainer.create(...)."
            )
        self._connection = connection
        self._neartime_callbacks = neartime_callbacks
        self._device_setup_loader = device_setup_loader
        self._dataserver = dataserver
        self._discovered_device_setup = discovered_device_setup
        self._lock = asyncio.Lock()
        self._submissions: dict[str, Submission] = {}
        self._queue_order: list[str] = []

    @classmethod
    async def create(
        cls,
        *,
        neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
        controller_factory: Callable[..., Awaitable[AsyncLocalController]]
        | None = None,
        device_setup_loader: DeviceSetupLoader | None = None,
        dataserver: tuple[str, str] | None = None,
        do_emulation: bool = False,
        reset_devices: bool = False,
    ) -> ControllerContainer:
        """Connect to hardware and return a ready-to-use container.

        Args:
            neartime_callbacks: Pre-registered near-time callbacks available
                to all experiments.
            controller_factory: Async factory that creates an
                :class:`~laboneq.controller.api.async_local_controller.AsyncLocalController`.
                Override in tests to avoid real hardware connections.
            device_setup_loader: Hot-reloadable loader for the server's
                :class:`~laboneq.dsl.device.DeviceSetup`.  Optional — when
                ``None``, a minimal setup is auto-created from *dataserver*.
            dataserver: ``(host, port)`` of the hardware server (SCM or LabOne
                dataserver).
            do_emulation: Run in emulation mode (no real hardware).
            reset_devices: Reset hardware on connect.

        Raises:
            RuntimeError: No device setup and no dataserver configured.
            Exception: Any failure from the controller factory propagates.
        """
        factory = controller_factory or _create_controller
        callbacks = neartime_callbacks or {}

        if device_setup_loader is not None:
            device_setup = device_setup_loader.get()
            discovered: DeviceSetup | None = None
        elif dataserver is not None:
            device_setup = _auto_create_device_setup(dataserver)
            discovered = device_setup
        else:
            raise RuntimeError(
                "No device setup configured and no dataserver address "
                "available.  Provide --devicesetup or --dataserver."
            )
        _apply_dataserver(device_setup, dataserver)

        fingerprint = cls._compute_fingerprint(device_setup)
        controller = await factory(
            device_setup=device_setup,
            neartime_callbacks=callbacks,
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
            connection=SessionState(
                device_config_fingerprint=fingerprint,
                controller=controller,
            ),
            neartime_callbacks=callbacks,
            device_setup_loader=device_setup_loader,
            dataserver=dataserver,
            discovered_device_setup=discovered,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fingerprint(device_setup: DeviceSetup) -> str | None:
        """Compute hardware fingerprint; returns ``None`` if not possible."""
        profile = device_setup.system_profile
        if profile is None:
            logger.debug(
                "Device setup has no hardware profile; fingerprint check skipped"
            )
            return _NO_FINGERPRINT
        return profile.get_fingerprint()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    def get_queue_depth(self) -> int:
        """Return number of live (non-tombstoned) experiments."""
        return sum(
            1 for s in self._submissions.values() if isinstance(s, SubmissionInfo)
        )

    def is_running(self) -> bool:
        """Return ``True`` if any experiment submission is active.

        .. note::

            This currently returns ``True`` as long as *any* submission
            exists (queued, running, or completed-but-not-yet-released).
            It does not distinguish "actively executing on hardware" from
            "sitting in the queue."  Refine once the controller exposes a
            proper RUNNING status.
        """
        # TODO: track running state explicitly once RUNNING status is implemented.
        return self.get_queue_depth() > 0

    def get_device_setup(self) -> DeviceSetup:
        """Return the server's device setup (hot-reloaded if applicable)."""
        if self._device_setup_loader is not None:
            setup = self._device_setup_loader.get()
            _apply_dataserver(setup, self._dataserver)
            return setup
        assert self._discovered_device_setup is not None
        return self._discovered_device_setup

    @property
    def registered_callback_names(self) -> list[str]:
        """Names of registered near-time callbacks."""
        return list(self._neartime_callbacks.keys())

    @property
    def registered_callback_info(self) -> dict[str, CallbackInfoDict]:
        """Registered near-time callbacks with their signatures and docstrings."""
        return {
            name: {
                "signature": str(inspect.signature(func)),
                "docstring": inspect.getdoc(func),
            }
            for name, func in self._neartime_callbacks.items()
        }

    def submission_exists(self, experiment_id: str) -> bool:
        """Return ``True`` if the given UUID is tracked (live or tombstoned)."""
        return experiment_id in self._submissions

    def queue_position(self, experiment_id: str) -> int | None:
        """Return zero-based queue position, or ``None`` if not in the queue.

        Counts only live submissions (not tombstones) ahead of
        *experiment_id* in insertion order.  Returns ``None`` for
        tombstoned or unknown experiments.
        """
        # TODO: This function has a bug: It assumes that the submission on the controller's queue
        # remains until it is explicitely deleted (via close_submission). This might not be true - to be confirmed.
        if experiment_id not in self._queue_order:
            return None
        if isinstance(self._submissions.get(experiment_id), Tombstone):
            return None
        idx = self._queue_order.index(experiment_id)
        return sum(
            1
            for uid in self._queue_order[:idx]
            if isinstance(self._submissions.get(uid), SubmissionInfo)
        )

    # ------------------------------------------------------------------
    # Main operations
    # ------------------------------------------------------------------

    async def submit_experiment(
        self, experiment_id: str, scheduled_experiment: ScheduledExperiment
    ) -> int:
        """Accept an experiment for execution.

        Args:
            experiment_id: Client-supplied UUID for this experiment.
            scheduled_experiment: The compiled experiment.

        Returns:
            Zero-based queue position at submission time.

        Raises:
            ExperimentAlreadyExistsError: *experiment_id* is already known.
            FingerprintMismatchError: Experiment was compiled for different hardware.
            MissingCallbacksError: Required callbacks are not registered.
        """
        async with self._lock:
            # Reject duplicate UUIDs (live or tombstoned)
            if experiment_id in self._submissions:
                raise ExperimentAlreadyExistsError(
                    f"Experiment {experiment_id!r} is already known; "
                    "use a different UUID."
                )

            # Resolve device setup and fingerprint
            device_setup = self.get_device_setup()
            fingerprint = self._compute_fingerprint(device_setup)

            # Validate device fingerprint
            exp_fp = scheduled_experiment.device_setup_fingerprint
            if fingerprint is not None and exp_fp and exp_fp != fingerprint:
                raise FingerprintMismatchError(
                    f"Experiment was compiled for device fingerprint "
                    f"{exp_fp!r} but the server's device setup has "
                    f"fingerprint {fingerprint!r}. "
                    "Recompile the experiment against the current device setup."
                )

            # Validate callbacks
            collector = _NeartimeCallbackCollector()
            collector.run(scheduled_experiment.execution)
            missing = collector.callbacks - set(self._neartime_callbacks)
            if missing:
                raise MissingCallbacksError(
                    f"Experiment requires callbacks not registered on the "
                    f"server: {sorted(missing)}. "
                    f"Available: {sorted(self._neartime_callbacks)}."
                )

            # Submit to controller
            handle = await self._connection.controller.submit_experiment(
                scheduled_experiment
            )
            self._submissions[experiment_id] = SubmissionInfo(handle=handle)
            self._queue_order.append(experiment_id)
            position = len(self._queue_order) - 1

            logger.info(
                "Submitted experiment %s (queue position %d)",
                experiment_id,
                position,
            )
            return position

    async def get_submission_status(
        self, experiment_id: str
    ) -> tuple[SubmissionStatus, int | None]:
        """Return ``(status, queue_position)`` for *experiment_id*.

        Raises:
            KeyError: Experiment not found.
        """
        submission = self._submissions.get(experiment_id)
        if submission is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        if isinstance(submission, Tombstone):
            return submission.status, None
        status = await self._connection.controller.submission_status(submission.handle)
        pos = self.queue_position(experiment_id)
        return status, pos

    async def get_submission_results(
        self, experiment_id: str
    ) -> tuple[ExperimentResults | None, str | None]:
        """Await completion and return ``(results, error)``.

        Raises:
            KeyError: Experiment not found.
        """
        submission = self._submissions.get(experiment_id)
        if submission is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        if isinstance(submission, Tombstone):
            return None, submission.error
        try:
            results = await self._connection.controller.submission_results(
                submission.handle
            )
        except Exception as exc:
            logger.exception("Error retrieving results for %s", experiment_id)
            return None, str(exc)
        else:
            return results, None

    async def cancel_experiment(self, experiment_id: str) -> None:
        """Cancel a queued/running experiment or release a completed one.

        Raises:
            KeyError: Experiment not found or already cancelled.
        """
        submission = self._submissions.get(experiment_id)
        if submission is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        if isinstance(submission, Tombstone):
            raise KeyError(f"Experiment {experiment_id!r} was already cancelled")
        del self._submissions[experiment_id]
        try:
            self._queue_order.remove(experiment_id)
        except ValueError:
            pass
        await self._connection.controller.cancel_submission(submission.handle)
        logger.info("Cancelled experiment %s", experiment_id)
