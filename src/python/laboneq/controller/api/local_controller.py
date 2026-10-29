# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import attrs

from laboneq.controller.api.commons import SubmissionRegistry, to_results
from laboneq.controller.api.controller_api import ControllerAPI
from laboneq.controller.constants import DEFAULT_TIMEOUT_S, READY_TIMEOUT_S
from laboneq.controller.controller_core import ControllerCore
from laboneq.core.utilities.async_helpers import EventLoopHolder

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.controller_core import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


_logger = logging.getLogger(__name__)


class LocalController(ControllerAPI):
    """Controller interface."""

    @staticmethod
    def create(
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
        instrument_topology: InstrumentTopology | None = None,
    ) -> LocalController:
        """Create an instance of the LocalController and connect it.

        Returns the instance even if connecting fails. `get_not_ready_reason`
        then reports the failure, and the controller tries again in the
        background when it needs the instruments, at most every
        `ControllerCore._RECONNECT_HOLDOFF_S` seconds. With `reset_devices`, the
        first successful connect resets the instruments, and no later one does.
        """
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = ControllerCore(
            device_setup=device_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        local_controller = LocalController(
            device_setup=device_setup,
            controller=controller,
            instrument_topology=instrument_topology,
            do_emulation=do_emulation,
        )
        try:
            local_controller.connect(
                reset_devices=reset_devices,
                disable_runtime_checks=disable_runtime_checks,
                timeout_s=DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s,
            )
        except Exception as e:
            _logger.warning("Cannot connect to the instruments: %s", e)
        return local_controller

    def __init__(
        self,
        device_setup: DeviceSetup,
        controller: ControllerCore,
        instrument_topology: InstrumentTopology | None = None,
        do_emulation: bool = True,
    ):
        self._device_setup = device_setup
        # Keep reference to avoid garbage collection
        self._controller = controller
        # Runs every coroutine of `controller`, which must all run on one loop.
        self._event_loop = EventLoopHolder()
        self._event_loop.start()
        self._instrument_topology = instrument_topology
        self._do_emulation = do_emulation
        self._submissions = SubmissionRegistry()

    def connect(
        self,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        """Connect to the instruments, or to the emulator under `do_emulation`.

        Raises if connecting fails. The interfaces do not declare this method:
        a remote implementation connects to the service in `create`, and the
        service connects to the instruments.
        """
        self._event_loop.run(
            self._controller.connect,
            do_emulation=self._do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )

    def close(self):
        for handle in self._submissions:
            # A failed submission has already reached the caller through
            # `get_experiment`; teardown has to reach the disconnect regardless.
            with suppress(Exception):
                self.cancel_experiment(handle)
        try:
            self._event_loop.run(self._controller.disconnect)
        finally:
            self._event_loop.stop()

    def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    def get_instrument_topology(self) -> InstrumentTopology | None:
        if self._instrument_topology is None:
            return None
        self._event_loop.run(self._controller.ensure_setup_description)
        # A client rebuilds a `DeviceSetup` from the topology, and compiles for it
        # with the setup description. Emulation has no instruments to fetch one from.
        setup_description = self._controller.setup_description
        if self._do_emulation or setup_description is None:
            return self._instrument_topology
        return attrs.evolve(
            self._instrument_topology, setup_description=setup_description
        )

    def get_not_ready_reason(self) -> str | None:
        return self._event_loop.run(self._controller.get_not_ready_reason)

    def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        self._event_loop.run(self._controller.wait_until_ready, timeout_s)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._controller.neartime_callbacks

    def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller.neartime_callbacks.update(neartime_callbacks)

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        self._event_loop.run(
            self._controller.disable_outputs, device_uids, logical_signals, unused_only
        )

    def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        handle = self._submissions.ensure_unique_handle(handle)
        self._submissions.add(
            handle,
            self._event_loop.run(
                self._controller.submit_compiled,
                scheduled_experiment=scheduled_experiment,
                coprocessor_payload_overrides=coprocessor_payload_overrides,
            ),
        )
        return handle

    def wait_for_experiment(self, handle: SubmissionHandle):
        self._event_loop.run(
            self._controller.wait_submission, self._submissions.get(handle)
        )

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._controller.submission_status(self._submissions.get(handle))

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        self.wait_for_experiment(handle)
        return self.get_experiment_results(handle)

    def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        return to_results(
            self._device_setup,
            self._controller.submission_results(self._submissions.get(handle)),
        )

    def cancel_experiment(self, handle: SubmissionHandle):
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        self.wait_for_experiment(handle)

    def close_submission(self, handle: SubmissionHandle):
        self.cancel_experiment(handle)
        self._submissions.remove(handle)
