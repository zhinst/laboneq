# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from laboneq.core.exceptions import LabOneQException

from ..calibration import Calibration
from ..serialization import Serializer
from .acquired_result import AcquiredResult

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


@dataclass(init=True, repr=True, order=True)
class Results:
    #: The source experiment
    experiment: Experiment = field(default=None)

    #: The device setup on which the experiment was run.
    device_setup: DeviceSetup = field(default=None)

    #: The compiled experiment.
    compiled_experiment: CompiledExperiment = field(default=None)

    #: The acquired results, organized by handle.
    acquired_results: Dict[str, AcquiredResult] = field(default=None)

    #: List of the results of each user user function, by name of the function.
    user_func_results: Dict[str, List[Any]] = field(default=None)

    #: Any exceptions that occurred during the execution of the experiment. Entries are
    #: tuples of
    #:
    #: * the indices of the loops where the error occurred,
    #: * the section uid,
    #: * the error message.
    execution_errors: List[Tuple[List[int], str, str]] = field(default=None)

    def __eq__(self, other):
        if self is other:
            return True
        return (
            self.experiment == other.experiment
            and self.device_setup == other.device_setup
            and self.compiled_experiment == other.compiled_experiment
            and self.acquired_results == other.acquired_results
            and self.user_func_results == other.user_func_results
        )

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
            An object of type :class:`~.dsl.result.acquired_result.AcquiredResult`.

        Raises:
            LabOneQException: No result is available for the provided handle.
        """

        self._check_handle(handle)
        return self.acquired_results[handle]

    def get_data(self, handle: str) -> ArrayLike:
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

    def get_axis_name(self, handle: str) -> List[Union[str, List[str]]]:
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

    def get_axis(self, handle: str) -> List[Union[ArrayLike, List[ArrayLike]]]:
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

    def get_last_nt_step(self, handle: str) -> List[int]:
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
    def device_calibration(self) -> Optional[Calibration]:
        """Get the device setup's calibration.

        See Also:
            :py:meth:`DeviceSetup.get_calibration() <laboneq.dsl.device.device_setup.DeviceSetup.get_calibration>`
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
    def load(filename) -> Results:
        return Serializer.from_json_file(filename, Results)

    def save(self, filename):
        Serializer.to_json_file(self, filename)
