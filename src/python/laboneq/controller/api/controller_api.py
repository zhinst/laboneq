# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from laboneq.controller.constants import READY_TIMEOUT_S

if TYPE_CHECKING:
    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.controller_core import SubmissionStatus
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


class ControllerAPI(ABC):
    """Abstract base class defining the controller API for experiment management."""

    @abstractmethod
    def close(self):
        """Shut down the controller and release all resources."""

    @abstractmethod
    def get_default_devicesetup(self) -> DeviceSetup:
        """Retrieve the device setup describing the hardware the controller is connected to."""

    @abstractmethod
    def get_instrument_topology(self) -> InstrumentTopology | None:
        """Retrieve the instrument topology describing the hardware layout.

        Returns None if no instrument topology is configured.

        Raises:
            HardwareNotReadyError: The hardware cannot run experiments now.
        """

    @abstractmethod
    def get_not_ready_reason(self) -> str | None:
        """Return why the controller cannot run experiments now, or None if it can."""

    @abstractmethod
    def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        """Return once the controller can run experiments.

        Raises:
            HardwareNotReadyError: The controller still cannot run experiments
                after `timeout_s`. The message gives the last reason.
        """

    @abstractmethod
    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        """Disable the outputs of the given logical signals or instruments.

        Without logical signals, disables every output of the given instruments,
        or of all instruments if none are given. With `unused_only`, keeps the
        outputs that a logical signal uses. Logical signals exclude instruments
        and `unused_only`.
        """

    @abstractmethod
    def submit_experiment(
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

    @abstractmethod
    def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        """Retrieve the current results of an experiment.

        Returns immediately. The results are partial while the experiment runs.
        """

    # TODO(2K): Streaming of the running experiment results

    @abstractmethod
    def cancel_experiment(self, handle: SubmissionHandle):
        """Cancel a running experiment submission. Drop a pending submission from the queue."""

    @abstractmethod
    def close_submission(self, handle: SubmissionHandle):
        """Close the submission handle and release associated resources. The handle becomes invalid."""
