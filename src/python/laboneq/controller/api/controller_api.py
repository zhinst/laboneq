# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from laboneq.dsl.device.device_setup import DeviceSetup

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.result.results import Results


class SubmissionHandle:
    """
    A handle representing a submitted experiment.

    This handle is an anonymous object used to track the status and results
    of an experiment submission. It does not expose any public attributes or methods,
    all interactions are done via the ControllerAPI methods.
    """

    def __init__(self, handle_id: int | None = None):
        self.id = id(self) if handle_id is None else handle_id

    def __hash__(self) -> int:
        return self.id

    @property
    def hex(self) -> str:
        """Hexadecimal string representation of the handle ID."""
        return "%032x" % self.id


class SubmissionStatus(Enum):
    """The status of an experiment submission."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ControllerAPI(ABC):
    """Abstract base class defining the controller API for experiment management."""

    @abstractmethod
    def close(self):
        """Shut down the controller and release all resources."""

    @abstractmethod
    def get_default_devicesetup(self) -> DeviceSetup:
        """Retrieve the device setup describing the hardware the controller is connected to."""

    @abstractmethod
    def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The handle is returned immediately and can be used to track the submission status.
        """

    @abstractmethod
    def wait_for_experiment(self, handle: SubmissionHandle):
        """Wait for the experiment submission to complete."""

    @abstractmethod
    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        """Retrieve the current status of an experiment submission."""

    @abstractmethod
    def get_experiment(self, handle: SubmissionHandle) -> Results:
        """Retrieve the results of a completed experiment.

        Blocks until the experiment is complete. On already completed experiments,
        the results are returned immediately.
        """

    # TODO(2K): Streaming of the running experiment results

    @abstractmethod
    def cancel_experiment(self, handle: SubmissionHandle):
        """Cancel a running experiment submission. Drop a pending submission from the queue."""

    @abstractmethod
    def close_submission(self, handle: SubmissionHandle):
        """Close the submission handle and release associated resources. The handle becomes invalid."""
