# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from laboneq.controller.constants import READY_TIMEOUT_S

if TYPE_CHECKING:
    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.controller import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
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
    async def get_instrument_topology(self) -> InstrumentTopology | None:
        """Retrieve the instrument topology describing the hardware layout.

        Returns None if no instrument topology is configured.

        Raises:
            HardwareNotReadyError: The hardware cannot run experiments now.
        """

    @abstractmethod
    async def get_not_ready_reason(self) -> str | None:
        """Return why the controller cannot run experiments now, or None if it can."""

    @abstractmethod
    async def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        """Return once the controller can run experiments.

        Raises:
            HardwareNotReadyError: The controller still cannot run experiments
                after `timeout_s`. The message gives the last reason.
        """

    @abstractmethod
    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        """Submit an experiment for execution and return a handle.

        The handle is returned immediately and can be used to track the submission status.
        Optional unique handle can be provided to track the submission. If not provided, a new handle is generated.

        Raises:
            HardwareNotReadyError: The hardware cannot run experiments now.
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
