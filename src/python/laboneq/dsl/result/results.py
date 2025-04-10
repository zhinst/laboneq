# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
import warnings
from typing import TYPE_CHECKING, Any

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from ..calibration import Calibration
from ..serialization import Serializer
from .acquired_result import AcquiredResult, AcquiredResults

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


@classformatter
@attrs.define(init=False)
class Results:
    """Results of an LabOne Q experiment.

    Attributes:
        experiment (Experiment): The source experiment.
        device_setup (DeviceSetup): The device setup on which the experiment was run.
        compiled_experiment (CompiledExperiment): Compiled experiment.
        acquired_results (AcquiredResults): The acquired results, organized by handle.
        neartime_callback_results (dict[str, list[Any]]): List of the results of near-time callbacks by their name.
        user_func_results (dict[str, list[Any]]): Deprecated. Alias for neartime_callback_results.
        execution_errors (list[tuple[list[int], str, str]]): Any exceptions that occurred during the execution of the experiment.
            Entries are tuples of:

                * the indices of the loops where the error occurred
                * the experiment section uid
                * the error message
        pipeline_jobs_timestamps (dict[str, list[float]]): The timestamps of all pipeline jobs, in seconds.  Organized by signal, then pipeline job id.

    !!! version-changed "Deprecated in version 2.19.0"
        The `user_func_results` attribute was deprecated in version 2.19.0.
        Use `neartime_callback_results` instead.
    """

    experiment: Experiment = attrs.field(default=None)
    device_setup: DeviceSetup = attrs.field(default=None)
    compiled_experiment: CompiledExperiment = attrs.field(default=None)
    acquired_results: AcquiredResults = attrs.field(default=AcquiredResults)
    neartime_callback_results: dict[str, list[Any]] = attrs.field(default=None)
    execution_errors: list[tuple[list[int], str, str]] = attrs.field(default=None)
    pipeline_jobs_timestamps: dict[str, list[float]] = attrs.field(factory=dict)

    def __init__(
        self,
        experiment: Experiment | None = None,
        device_setup: DeviceSetup | None = None,
        compiled_experiment: CompiledExperiment | None = None,
        acquired_results: AcquiredResults | None = None,
        neartime_callback_results: dict[str, list[Any]] | None = None,
        execution_errors: list[tuple[list[int], str, str]] | None = None,
        user_func_results: dict[str, list[Any]] | None = None,
        pipeline_jobs_timestamps: dict[str, list[float]] | None = None,
    ):
        self.experiment = experiment
        self.device_setup = device_setup
        self.compiled_experiment = compiled_experiment
        self.acquired_results = (
            acquired_results if acquired_results is not None else AcquiredResults()
        )
        self.neartime_callback_results = neartime_callback_results
        self.execution_errors = execution_errors

        if user_func_results is not None:
            if neartime_callback_results is not None:
                raise LabOneQException(
                    "Results can only be initialized with either 'neartime_callback_results' or 'user_func_results', not both."
                )
            self.neartime_callback_results = user_func_results
        if pipeline_jobs_timestamps is None:
            pipeline_jobs_timestamps = {}
        self.pipeline_jobs_timestamps = pipeline_jobs_timestamps

    @property
    def user_func_results(self):
        """Deprecated. Alias for neartime_callback_results."""
        warnings.warn(
            "The 'user_func_results' attribute is deprecated. Use 'neartime_callback_results' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.neartime_callback_results

    def _check_handle(self, handle: str) -> None:
        if handle not in self.acquired_results:
            raise LabOneQException(f"No result for handle: {handle}")

    def get_result(self, handle: str) -> AcquiredResult:
        """Returns the acquired result.

        Returns the result acquired for an 'acquire' event with the specific handle
        that was assigned to it in the experiment definition.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.

        Returns:
            result: The acquire event result.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle]

    def get_data(self, handle: str) -> npt.NDArray[Any]:
        """Returns the acquired result data.

        Returns the result acquired for an 'acquire' event with the specific handle
        that was assigned to it in the experiment definition.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.

        Returns:
            A multidimensional numpy array, where each dimension corresponds to a sweep
            loop nesting level, the outermost sweep being the first dimension.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle].data

    def get_axis_name(self, handle: str) -> list[str | list[str]]:
        """Returns the names of axes.

        Returns the list of axis names, that correspond to the dimensions of the result returned by
        'get'. Elements in the list are in the same order as the dimensions of the array returned by
        'get'. Each element is either a string for a simple sweep, or a list of strings for a parallel
        sweep. Values are given by the 'axis_name' argument of the corresponding sweep parameter, or
        the 'uid' of the same parameter, if 'axis_name' is not specified.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.

        Returns:
            A list of axis names. Each element may be either a string or a list of strings.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle].axis_name

    def get_axis(self, handle: str) -> list[npt.NDArray[Any] | list[npt.NDArray[Any]]]:
        """Returns the axes grids.

        Returns the list, where each element represents an axis of the corresponding dimension of
        the result array returned by 'get'. Each element is either a 1D numpy array for a simple
        sweep, or a list of 1D numpy arrays for a parallel sweep. The length of each array matches
        the number of steps of the corresponding sweep, and the values are the sweep parameter
        values at each step.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.

        Returns:
            A list of axis grids. Each element may be either a 1D numpy array or a list of such
            arrays.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle].axis

    def get_last_nt_step(self, handle: str) -> list[int]:
        """Returns the list of axis indices of the last measured near-time point.

        Returns the list of axis indices that represent the last measured near-time point. Use this
        to retrieve the last recorded partial result from the 'data' array. 'None' means that no
        measurements were taken so far. The list only covers axes that correspond to the near-time
        sweeps / dimensions. All the elements of inner real-time sweeps that correspond to a single
        real-time execution step are read at once and filled entirely.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.

        Returns:
            A list of axis indices.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle].last_nt_step

    @property
    def device_calibration(self) -> Calibration | None:
        """Get the device setup's calibration.

        See also
        [DeviceSetup.get_calibration][laboneq.dsl.device.device_setup.DeviceSetup.get_calibration].
        """
        if self.device_setup is None:
            return None
        return self.device_setup.get_calibration()

    @property
    def experiment_calibration(self):
        return self.experiment.get_calibration()

    @property
    def signal_map(self):
        return self.experiment.get_signal_map()

    @staticmethod
    def load(filename: str) -> Results:
        """Load the result from a specified file.

        !!! version-changed "Deprecated in version 2.50.0"
            Use `laboneq.simple.load` instead.

        Args:
            filename: The file to load the result from.
        """
        warnings.warn(
            "The `Results.load` method is deprecated and will be removed in future releases. "
            "Please use the `load` function from the `laboneq.simple` module instead. ",
            FutureWarning,
            stacklevel=2,
        )
        return Serializer.from_json_file(filename, Results)

    def save(self, filename: str) -> None:
        """Save the result to a specified file.

        !!! version-changed "Deprecated in version 2.50.0"
            Use `laboneq.simple.save` instead.

        Args:
            filename (str): The name of the file to save the result to.
        """
        warnings.warn(
            "The `Results.save` method is deprecated and will be removed in future releases. "
            "Please use the `save` function from the `laboneq.simple` module instead. ",
            FutureWarning,
            stacklevel=2,
        )
        Serializer.to_json_file(self, filename)
