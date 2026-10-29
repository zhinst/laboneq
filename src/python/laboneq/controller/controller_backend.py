# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The controller's interface to the instruments of one backend.

Each backend plugin registers one `ControllerBackend` implementation under its
backend name. The controller instantiates the one registered for the backend of
its device setup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from laboneq._backend_plugin import resolve_backend
from laboneq.controller.utilities.exception import LabOneQControllerException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager

    from laboneq.controller.backend_runtime_state import BackendRuntimeState
    from laboneq.controller.devices.device_utils import NodeCollector
    from laboneq.controller.near_time_replacement import NearTimeReplacements
    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.controller.results import ResultsBuilder
    from laboneq.controller.versioning import SetupCaps
    from laboneq.data.nt_step_key import NtStepKey
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.data.setup_descriptions import SetupDescription
    from laboneq.dsl.device.device_setup import DeviceSetup


class ControllerBackend(ABC):
    """Drives the instruments of one device setup."""

    @abstractmethod
    def __init__(
        self,
        device_setup: DeviceSetup,
        setup_caps: SetupCaps,
        ignore_version_mismatch: bool,
    ): ...

    @property
    @abstractmethod
    def device_setup_fingerprint(self) -> str:
        """Fingerprint of the device setup.

        A compiled experiment runs only if its fingerprint matches this one.
        """

    @property
    @abstractmethod
    def setup_description(self) -> SetupDescription | None:
        """Setup description fetched from the instruments, if any."""

    @property
    @abstractmethod
    def not_ready_reason(self) -> str | None:
        """Why the instruments cannot run experiments now, or None if they can.

        Valid once `connect` returns.
        """

    @abstractmethod
    async def connect(
        self,
        do_emulation: bool,
        reset_devices: bool,
        disable_runtime_checks: bool,
        timeout_s: float | None,
    ):
        """Connects to the instruments and brings them to an idle state."""

    @abstractmethod
    async def disconnect(self):
        """Disconnects from the instruments."""

    @abstractmethod
    async def ensure_setup_description(self):
        """Fetches the setup description unless the backend already has the current one.

        Raises if the instruments cannot provide it.
        """

    @abstractmethod
    async def disable_outputs(
        self,
        device_uids: list[str] | None,
        logical_signals: list[str] | None,
        unused_only: bool,
    ):
        """Disables the outputs of the given devices or logical signals."""

    @abstractmethod
    def pre_process(
        self,
        scheduled_experiment: ScheduledExperiment,
        coprocessor_payload_overrides: dict[str, bytes],
    ) -> RecipeData:
        """Prepares a compiled experiment for execution."""

    @abstractmethod
    def create_backend_runtime_state(
        self, recipe_data: RecipeData
    ) -> BackendRuntimeState | None:
        """Creates the state the backend keeps across the near-time steps of a run."""

    @abstractmethod
    def experiment(self, recipe_data: RecipeData) -> AbstractAsyncContextManager[None]:
        """Returns a context spanning one experiment run.

        The run starts before the first near-time step and ends once all
        results are collected.
        """

    @abstractmethod
    async def prepare_nt_step(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        user_set_nodes: NodeCollector,
        replacements: NearTimeReplacements,
    ):
        """Configures the instruments and uploads the artifacts for a near-time step."""

    @abstractmethod
    async def execute_nt_step(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        timeout_s: float,
        results_pending: Callable[[], Awaitable[None]],
    ):
        """Runs the real-time part of a near-time step until it completes.

        Awaits `results_pending` once the results of the step can be collected.
        """

    @abstractmethod
    async def after_nt_step(self):
        """Raises if the instruments report errors after a near-time step."""

    @abstractmethod
    async def read_results(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        results_builder: ResultsBuilder,
    ):
        """Reads the results of a near-time step into `results_builder`."""

    @abstractmethod
    async def process_step_results(
        self,
        recipe_data: RecipeData,
        results_builder: ResultsBuilder,
        backend_runtime_state: BackendRuntimeState | None,
        nt_step: NtStepKey | None,
    ):
        """Processes the results of a near-time step, or of the run if `nt_step` is None."""


_registered_backends: dict[str, type[ControllerBackend]] = {}


def register_controller_backend(backend: str, cls: type[ControllerBackend]) -> None:
    """Registers `cls` as the controller backend for setups of `backend`."""
    registered = _registered_backends.get(backend)
    if registered is not None and registered is not cls:
        raise ValueError(
            f"Backend {backend!r} already has the controller backend "
            f"{registered.__module__}.{registered.__qualname__}."
        )
    _registered_backends[backend] = cls


def create_controller_backend(
    device_setup: DeviceSetup,
    setup_caps: SetupCaps,
    ignore_version_mismatch: bool,
) -> ControllerBackend:
    """Creates the controller backend registered for the backend of `device_setup`."""
    backend = resolve_backend(device_setup)
    cls = _registered_backends.get(backend)
    if cls is None:
        registered = ", ".join(sorted(_registered_backends)) or "none"
        raise LabOneQControllerException(
            f"No controller backend registered for backend {backend!r}. "
            f"Registered: {registered}."
        )
    return cls(
        device_setup=device_setup,
        setup_caps=setup_caps,
        ignore_version_mismatch=ignore_version_mismatch,
    )
