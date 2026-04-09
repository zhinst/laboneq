# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from laboneq.controller.api.controller_api import SubmissionHandle, SubmissionStatus
from laboneq.dsl.device.device_setup import DeviceSetup

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.result.results import Results


class AsyncControllerAPI(ABC):
    """Abstract base class defining the controller API for experiment management."""

    @abstractmethod
    async def aclose(self):
        """Shut down the controller and release all resources."""

    @abstractmethod
    async def get_default_devicesetup(self) -> DeviceSetup:
        """Retrieve the device setup describing the hardware the controller is connected to."""

    @abstractmethod
    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The handle is returned immediately and can be used to track the submission status.
        """

    @abstractmethod
    async def wait_for_experiment(self, handle: SubmissionHandle):
        """Wait for the experiment submission to complete."""

    @abstractmethod
    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        """Retrieve the current status of an experiment submission."""

    @abstractmethod
    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        """Retrieve the results of a completed experiment.

        Blocks until the experiment is complete. On already completed experiments,
        the results are returned immediately.
        """

    # TODO(2K): Streaming of the running experiment results

    @abstractmethod
    async def cancel_experiment(self, handle: SubmissionHandle):
        """Cancel a running experiment submission. Drop a pending submission from the queue."""

    @abstractmethod
    async def close_submission(self, handle: SubmissionHandle):
        """Close the submission handle and release associated resources. The handle becomes invalid."""
