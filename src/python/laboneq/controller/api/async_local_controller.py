# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import attrs

from laboneq.controller.api.async_controller_api import AsyncControllerAPI
from laboneq.controller.api.commons import SubmissionRegistry
from laboneq.controller.constants import DEFAULT_TIMEOUT_S, READY_TIMEOUT_S
from laboneq.controller.controller import Controller
from laboneq.dsl.result.results import Results

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.callback_channel import CallbackChannel
    from laboneq.controller.controller import SubmissionStatus
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup


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
        `Controller._RECONNECT_HOLDOFF_S` seconds. With `reset_devices`, the
        first successful connect resets the instruments, and no later one does.
        """
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = Controller(
            device_setup=device_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        local_controller = AsyncLocalController(
            device_setup=device_setup,
            controller=controller,
            instrument_topology=instrument_topology,
            do_emulation=do_emulation,
        )
        try:
            await controller._connect_async(
                do_emulation=do_emulation,
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
        controller: Controller,
        instrument_topology: InstrumentTopology | None = None,
        do_emulation: bool = True,
    ):
        self._device_setup = device_setup
        self._controller = controller
        self._instrument_topology = instrument_topology
        self._do_emulation = do_emulation
        self._submissions = SubmissionRegistry()

    async def aclose(self):
        for handle in self._submissions:
            # A failed submission has already reached the caller through
            # `get_experiment`; teardown has to reach the disconnect regardless.
            with suppress(Exception):
                await self.cancel_experiment(handle)
        await self._controller._disconnect_async()

    async def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    async def get_instrument_topology(self) -> InstrumentTopology | None:
        if self._instrument_topology is None:
            return None
        await self._controller._ensure_setup_description_async()
        # A client rebuilds a `DeviceSetup` from the topology, and compiles for it
        # with the setup description. Emulation has no instruments to fetch one from.
        setup_description = self._controller.setup_description
        if self._do_emulation or setup_description is None:
            return self._instrument_topology
        return attrs.evolve(
            self._instrument_topology, setup_description=setup_description
        )

    async def get_not_ready_reason(self) -> str | None:
        return await self._controller._get_not_ready_reason_async()

    async def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        await self._controller._wait_until_ready_async(timeout_s)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._controller.neartime_callbacks

    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller.neartime_callbacks.update(neartime_callbacks)

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        callback_channel: CallbackChannel | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        handle = self._submissions.ensure_unique_handle(handle)
        self._submissions.add(
            handle,
            await self._controller._submit_compiled_async(
                scheduled_experiment=scheduled_experiment,
                callback_channel=callback_channel,
                coprocessor_payload_overrides=coprocessor_payload_overrides,
            ),
        )
        return handle

    async def get_callback_channel(
        self, handle: SubmissionHandle
    ) -> CallbackChannel | None:
        """The client callback channel for *handle*, if the experiment has one."""
        return self._submissions.get(handle).callback_channel

    async def wait_for_experiment(self, handle: SubmissionHandle):
        await self._controller._wait_submission_async(self._submissions.get(handle))

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._controller.submission_status(self._submissions.get(handle))

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        await self.wait_for_experiment(handle)
        experiment_results = self._controller.submission_results(
            self._submissions.get(handle)
        )
        return Results(
            device_setup=self._device_setup,
            acquired_results=experiment_results.acquired_results,
            neartime_callback_results=experiment_results.neartime_callback_results,
            execution_errors=experiment_results.execution_errors,
            pipeline_jobs_timestamps=experiment_results.pipeline_jobs_timestamps,
        )

    async def get_experiment_results(
        self, handle: SubmissionHandle
    ) -> ExperimentResults:
        """Current, possibly partial, results. Does not await completion."""
        return self._controller.submission_results(self._submissions.get(handle))

    async def cancel_experiment(self, handle: SubmissionHandle):
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        await self.wait_for_experiment(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self.cancel_experiment(handle)
        self._submissions.remove(handle)
