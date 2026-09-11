# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

import attrs

from laboneq.controller.api.async_controller_api import AsyncControllerAPI
from laboneq.controller.api.commons import APIError, SubmissionRegistry
from laboneq.controller.constants import DEFAULT_TIMEOUT_S
from laboneq.controller.controller import Controller
from laboneq.controller.target_setup import build_target_setup
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
        """Create an instance of the AsyncLocalController."""
        if timeout_s is None:
            timeout_s = DEFAULT_TIMEOUT_S
        target_setup = build_target_setup(device_setup)
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = Controller(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        await controller._connect_async(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )
        if (
            not do_emulation
            and instrument_topology is not None
            and controller.setup_description is not None
        ):
            # Accompany the topology with the setup description discovered
            # while connecting (e.g. fetched from the ZQCS SCM), so that
            # clients fetching it via get_instrument_topology() can rebuild
            # a DeviceSetup carrying it, without connecting to hardware
            # themselves. Guarded the same way as the equivalent copy in
            # Session.connect(): under emulation there is no real SCM to
            # discover a setup description from.
            instrument_topology = attrs.evolve(
                instrument_topology, setup_description=controller.setup_description
            )
        return AsyncLocalController(
            device_setup=device_setup,
            controller=controller,
            instrument_topology=instrument_topology,
        )

    def __init__(
        self,
        device_setup: DeviceSetup,
        controller: Controller,
        instrument_topology: InstrumentTopology | None = None,
    ):
        self._device_setup = device_setup
        self._controller = controller
        self._instrument_topology = instrument_topology
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

    @property
    def instrument_topology(self) -> InstrumentTopology | None:
        """Return the instrument topology, or None if not configured."""
        return self._instrument_topology

    async def get_instrument_topology(self) -> InstrumentTopology:
        if self._instrument_topology is None:
            raise APIError("No instrument topology configured")
        return self._instrument_topology

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
