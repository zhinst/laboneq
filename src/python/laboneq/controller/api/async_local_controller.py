# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from laboneq.controller.api.async_controller_api import AsyncControllerAPI
from laboneq.controller.constants import DEFAULT_TIMEOUT_S, READY_TIMEOUT_S
from laboneq.controller.controller_core import ControllerCore
from laboneq.controller.experiment_dispatcher import ExperimentDispatcher

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.callback_channel import CallbackChannel
    from laboneq.controller.controller_core import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


_logger = logging.getLogger(__name__)


class AsyncLocalController(AsyncControllerAPI):
    """Asynchronous controller interface."""

    @staticmethod
    async def create(
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
        instrument_topology: InstrumentTopology | None = None,
    ) -> AsyncLocalController:
        """Create an instance of the AsyncLocalController and connect it.

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
        local_controller = AsyncLocalController(
            device_setup=device_setup,
            dispatcher=ExperimentDispatcher(controller, device_setup),
            instrument_topology=instrument_topology,
            do_emulation=do_emulation,
        )
        try:
            await local_controller.connect(
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
        self._instrument_topology = instrument_topology
        self._do_emulation = do_emulation

    async def connect(
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
        await self._dispatcher.connect(
            do_emulation=self._do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )

    async def aclose(self, timeout_s: float | None = None):
        """Shut down the controller and release all resources.

        Drops the queued experiments. Waits up to `timeout_s` seconds, or without
        limit if None, for a running experiment to end, and stops it if it has
        not. The interfaces do not declare `timeout_s`, which matters only to a
        caller that bounds its shutdown, such as the controller service.
        """
        await self._dispatcher.aclose(timeout_s)

    async def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    async def get_instrument_topology(self) -> InstrumentTopology | None:
        if self._instrument_topology is None:
            return None
        return await self._dispatcher.merge_setup_description(self._instrument_topology)

    async def get_not_ready_reason(self) -> str | None:
        return await self._dispatcher.get_not_ready_reason()

    async def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        await self._dispatcher.wait_until_ready(timeout_s)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._dispatcher.neartime_callbacks

    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._dispatcher.neartime_callbacks.update(neartime_callbacks)

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._dispatcher.disable_outputs(
            device_uids, logical_signals, unused_only
        )

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        callback_channel: CallbackChannel | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
        queue_id: Hashable = None,
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The experiment waits in the queue `queue_id`. Each queue runs its
        experiments in submission order, and the queues take turns. The
        interfaces do not declare `queue_id` or `callback_channel`, which matter
        only to a caller that runs the experiments of several remote clients, such
        as the controller service.

        Raises:
            HardwareNotReadyError: The hardware cannot run experiments now.
        """
        return await self._dispatcher.submit_experiment(
            scheduled_experiment,
            queue_id=queue_id,
            handle=handle,
            callback_channel=callback_channel,
            coprocessor_payload_overrides=coprocessor_payload_overrides,
        )

    async def get_callback_channel(
        self, handle: SubmissionHandle
    ) -> CallbackChannel | None:
        """The client callback channel for *handle*, if the experiment has one."""
        return self._dispatcher.get_callback_channel(handle)

    async def wait_for_experiment(self, handle: SubmissionHandle):
        await self._dispatcher.wait_for_experiment(handle)

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._dispatcher.get_experiment_status(handle)

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        await self.wait_for_experiment(handle)
        return await self.get_experiment_results(handle)

    async def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        return self._dispatcher.get_experiment_results(handle)

    async def cancel_experiment(self, handle: SubmissionHandle):
        await self._dispatcher.cancel_experiment(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self._dispatcher.close_submission(handle)
