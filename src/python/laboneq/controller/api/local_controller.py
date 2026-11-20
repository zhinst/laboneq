# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from laboneq.controller.api.controller_api import ControllerAPI
from laboneq.controller.constants import DEFAULT_TIMEOUT_S, READY_TIMEOUT_S
from laboneq.controller.controller_core import ControllerCore
from laboneq.controller.experiment_dispatcher import ExperimentDispatcher
from laboneq.core.utilities.async_helpers import EventLoopHolder

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

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
            dispatcher=ExperimentDispatcher(controller, device_setup),
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
        dispatcher: ExperimentDispatcher,
        instrument_topology: InstrumentTopology | None = None,
        do_emulation: bool = True,
    ):
        self._device_setup = device_setup
        self._dispatcher = dispatcher
        # Runs every coroutine of `dispatcher`, which must all run on one loop.
        self._event_loop = EventLoopHolder()
        self._event_loop.start()
        self._instrument_topology = instrument_topology
        self._do_emulation = do_emulation

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
            self._dispatcher.connect,
            do_emulation=self._do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )

    def close(self, timeout_s: float | None = None):
        """Shut down the controller and release all resources.

        Drops the queued experiments. Waits up to `timeout_s` seconds, or without
        limit if None, for a running experiment to end, and stops it if it has
        not. The interfaces do not declare `timeout_s`, which matters only to a
        caller that bounds its shutdown, such as the controller service.
        """
        try:
            self._event_loop.run(self._dispatcher.aclose, timeout_s)
        finally:
            self._event_loop.stop()

    def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    def get_instrument_topology(self) -> InstrumentTopology | None:
        if self._instrument_topology is None:
            return None
        return self._event_loop.run(
            self._dispatcher.merge_setup_description, self._instrument_topology
        )

    def get_not_ready_reason(self) -> str | None:
        return self._event_loop.run(self._dispatcher.get_not_ready_reason)

    def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        self._event_loop.run(self._dispatcher.wait_until_ready, timeout_s)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._dispatcher.neartime_callbacks

    def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._dispatcher.neartime_callbacks.update(neartime_callbacks)

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        self._event_loop.run(
            self._dispatcher.disable_outputs, device_uids, logical_signals, unused_only
        )

    def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
        queue_id: Hashable = None,
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The experiment waits in the queue `queue_id`. Each queue runs its
        experiments in submission order, and the queues take turns. The
        interfaces do not declare `queue_id`, which matters only to a caller that
        queues the experiments of several clients, such as the controller service.

        Raises:
            HardwareNotReadyError: The hardware cannot run experiments now.
        """
        return self._event_loop.run(
            self._dispatcher.submit_experiment,
            scheduled_experiment,
            queue_id=queue_id,
            handle=handle,
            coprocessor_payload_overrides=coprocessor_payload_overrides,
        )

    def wait_for_experiment(self, handle: SubmissionHandle):
        self._event_loop.run(self._dispatcher.wait_for_experiment, handle)

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._dispatcher.get_experiment_status(handle)

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        self.wait_for_experiment(handle)
        return self.get_experiment_results(handle)

    def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        return self._dispatcher.get_experiment_results(handle)

    def cancel_experiment(self, handle: SubmissionHandle):
        self._event_loop.run(self._dispatcher.cancel_experiment, handle)

    def close_submission(self, handle: SubmissionHandle):
        self._event_loop.run(self._dispatcher.close_submission, handle)
