# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.scheduled_experiment import ScheduledExperiment


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


class SubmissionStatus(Enum):
    """The status of an experiment submission."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class ControllerAPI(ABC):
    """Abstract base class defining the controller API for experiment management."""

    @abstractmethod
    async def shutdown(self):
        """Shut down the controller and release all resources."""

    @abstractmethod
    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """

    @abstractmethod
    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The handle is returned immediately and can be used to track the submission status.
        """

    @abstractmethod
    async def wait_for_completion(self, handle: SubmissionHandle):
        """Wait for the experiment submission to complete."""

    @abstractmethod
    async def submission_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        """Retrieve the current status of an experiment submission."""

    @abstractmethod
    async def submission_results(self, handle: SubmissionHandle) -> ExperimentResults:
        """Retrieve the results of a completed experiment.

        Blocks until the experiment is complete. On already completed experiments,
        the results are returned immediately.
        """

    # TODO(2K): Streaming of the running experiment results

    @abstractmethod
    async def cancel_submission(self, handle: SubmissionHandle):
        """Cancel a running experiment submission. Drop a pending submission from the queue."""

    @abstractmethod
    async def close_submission(self, handle: SubmissionHandle):
        """Close the submission handle and release associated resources. The handle becomes invalid."""
