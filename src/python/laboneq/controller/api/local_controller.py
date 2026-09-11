# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from laboneq.controller.api.commons import APIError, SubmissionRegistry
from laboneq.controller.api.controller_api import ControllerAPI
from laboneq.controller.constants import DEFAULT_TIMEOUT_S
from laboneq.controller.controller import Controller
from laboneq.controller.target_setup import build_target_setup
from laboneq.dsl.result.results import Results

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.controller import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup


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
        """Create an instance of the Controller."""
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
        controller.start()
        controller.connect(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )
        return LocalController(
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
        # Keep reference to avoid garbage collection
        self._controller = controller
        self._instrument_topology = instrument_topology
        self._submissions = SubmissionRegistry()

    def close(self):
        for handle in self._submissions:
            # A failed submission has already reached the caller through
            # `get_experiment`; teardown has to reach the disconnect regardless.
            with suppress(Exception):
                self.cancel_experiment(handle)
        self._controller.disconnect()

    def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    @property
    def instrument_topology(self) -> InstrumentTopology | None:
        """Return the instrument topology, or None if not configured."""
        return self._instrument_topology

    def get_instrument_topology(self) -> InstrumentTopology:
        if self._instrument_topology is None:
            raise APIError("No instrument topology configured")
        return self._instrument_topology

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._controller.neartime_callbacks

    def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller.neartime_callbacks.update(neartime_callbacks)

    def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        handle = self._submissions.ensure_unique_handle(handle)
        self._submissions.add(
            handle,
            self._controller.submit_compiled(
                scheduled_experiment=scheduled_experiment,
                coprocessor_payload_overrides=coprocessor_payload_overrides,
            ),
        )
        return handle

    def wait_for_experiment(self, handle: SubmissionHandle):
        self._controller.wait_submission(self._submissions.get(handle))

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._controller.submission_status(self._submissions.get(handle))

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        self.wait_for_experiment(handle)
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

    def cancel_experiment(self, handle: SubmissionHandle):
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        self.wait_for_experiment(handle)

    def close_submission(self, handle: SubmissionHandle):
        self.cancel_experiment(handle)
        self._submissions.remove(handle)
