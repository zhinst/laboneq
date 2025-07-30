# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
import copy
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeAlias, deprecated

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.utilities.attribute_wrapper import AttributeWrapper

from ..calibration import Calibration
from .acquired_result import AcquiredResult, AcquiredResults

if TYPE_CHECKING:
    import numpy as np
    from numpy import typing as npt

    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


ErrorList: TypeAlias = list[tuple[list[int], str, str]]


@classformatter
@attrs.define
class Results:
    """Results of an LabOne Q experiment.

    In addition to being accessible via the attributes and methods defined below,
    the `acquired_results` are also accessible directly on the result object
    itself using either attribute or dictionary-like item lookup.

    For example::

        ```python
        results = Results(
            acquired_results={
                "cal_trace/q0/g": AcquiredResult(
                    data=numpy.array([1, 2, 3]),
                    axis_name=["Amplitude"],
                    axis=[numpy.array([0, 1, 2])],
                ),
            },
        )

        # These all return the same data:
        results.cal_trace.q0.g
        results["cal_trace"]["q0"]["g"]
        results["cal_trace/q0/g"]
        ```

    Attributes:
        experiment:
            The source experiment. Deprecated.
        device_setup:
            The device setup on which the experiment was run. Deprecated.
        acquired_results:
            The acquired results, organized by handle.
        neartime_callback_results:
            Dictionary of the results of near-time callbacks by their name.
        execution_errors:
            A list of errors that occurred during the execution of the experiment.
            Entries are tuples of:

                * the indices of the loops where the error occurred
                * the experiment section uid
                * the error message
        pipeline_jobs_timestamps:
            The timestamps of all pipeline jobs, in seconds. Organized by signal, then pipeline job id.

    !!! version-removed "Removed in version 2.57.0"
        Removed the `.user_func_results` attribute and argument that were deprecated
        in version 2.19.0. Use `.neartime_callback_results` instead.

    !!! version-removed "Removed in version 2.55.0"
        The deprecated `.compiled_experiment` attribute was removed.
        Track the compiled experiment separately instead.

    !!! version-removed "Removed in version 2.54.0"
        The following deprecated methods for saving and loading were removed:
        - `load`
        - `save`
        Use the `load` and `save` functions from the `laboneq.simple` module instead.

    !!! version-changed "Deprecated in version 2.52.0."
        The `experiment`, `device_setup` and `compiled_experiment` attributes
        were deprecated in version 2.52.0 and are only populated when
        requested via `Session` or `Session.run`.
    """

    experiment: Experiment = attrs.field(default=None)
    device_setup: DeviceSetup = attrs.field(default=None)
    acquired_results: AcquiredResults = attrs.field(factory=AcquiredResults)
    neartime_callback_results: dict[str, list[Any]] = attrs.field(default=None)
    execution_errors: list[tuple[list[int], str, str]] = attrs.field(factory=list)
    pipeline_jobs_timestamps: dict[str, list[float]] = attrs.field(factory=dict)

    # Support for deprecated init parameters:
    _data: AcquiredResults | None = attrs.field(default=None)

    _acquired_results_wrapper: AttributeWrapper | None = attrs.field(
        init=False, default=None
    )
    _neartime_callbacks_wrapper: AttributeWrapper | None = attrs.field(
        init=False, default=None
    )

    @classmethod
    def _laboneq_exclude_from_legacy_serializer(cls):
        return [
            "_data",
            "_acquired_results_wrapper",
            "_neartime_callbacks_wrapper",
        ]

    def __attrs_post_init__(self):
        if self._data is not None:
            if not self.acquired_results:
                self.acquired_results = self._data
                self._data = None  # remove duplicate reference
            elif not self._data:
                # allow an empty _data to be passed
                self._data = None
            else:
                raise LabOneQException(
                    "Results can only be initialized with either 'acquired_results'"
                    " or 'data', not both."
                )

        if self.neartime_callback_results is None:
            self.neartime_callback_results = {}

        self._acquired_results_wrapper = AttributeWrapper(self.acquired_results)
        self._neartime_callbacks_wrapper = AttributeWrapper(
            self.neartime_callback_results
        )

    def __repr__(self) -> str:
        return (
            f"<{type(self).__qualname__}"
            f" id={id(self)}"
            f" errors={bool(self.execution_errors)}"
            f" len(acquired_results)={len(self.acquired_results)}"
            f" len(neartime_callbacks)={len(self.neartime_callback_results)}"
            ">"
        )

    def __rich_repr__(self):
        yield "errors", self.errors
        yield "data", self.data
        yield "neartime_callbacks", self.neartime_callbacks

    def __contains__(self, key: object) -> bool:
        return key in self._acquired_results_wrapper

    def __deepcopy__(self, memo: dict) -> Results:
        return self.__class__(
            experiment=copy.deepcopy(self.experiment),
            device_setup=copy.deepcopy(self.device_setup),
            acquired_results=copy.deepcopy(self.acquired_results),
            neartime_callback_results=copy.deepcopy(self.neartime_callback_results),
            execution_errors=copy.deepcopy(self.execution_errors),
            pipeline_jobs_timestamps=copy.deepcopy(self.pipeline_jobs_timestamps),
        )

    def __dir__(self):
        return super().__dir__() + list(self._acquired_results_wrapper._key_cache)

    def __getattr__(self, key: object) -> AttributeWrapper | object:
        # This intentionally only looks up the keys on the wrapper and
        # not arbitrary attributes
        try:
            return self._acquired_results_wrapper.__getitem__(key)
        except KeyError as err:
            raise AttributeError(*err.args) from None

    def __getitem__(self, key: object) -> AttributeWrapper | ErrorList | object:
        return self._acquired_results_wrapper.__getitem__(key)

    @property
    def errors(self) -> ErrorList:
        """The errors that occurred during running the experiment.

        !!! version-added "Added in version 2.52.0"
            The `.errors` attribute was added.
            The name was chosen to match that used on the (now removed)
            `RunExperimentResults` class.
        """
        return self.execution_errors

    @property
    def data(self) -> AttributeWrapper:
        """The results acquired during the real-time experiment.

        !!! version-added "Added in version 2.52.0"
            The `.data` attribute was added.
        """
        return self._acquired_results_wrapper

    @property
    def neartime_callbacks(self) -> AttributeWrapper:
        """The results of the near-time user callbacks.

        !!! version-added "Added in version 2.52.0"
            The `.neartime_callbacks` attribute was added.
            The name was chosen to match that used on the (now removed)
            `RunExperimentResults` class.
        """
        return self._neartime_callbacks_wrapper

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

    def get_data(self, handle: str) -> npt.NDArray[Any] | np.complex128:
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
    @deprecated(
        "The `.device_calibration` property is deprecated. Use"
        " `.device_setup.get_calibration()` instead.",
        category=FutureWarning,
    )
    def device_calibration(self) -> Calibration | None:
        """Get the device setup's calibration.

        See also
        [DeviceSetup.get_calibration][laboneq.dsl.device.device_setup.DeviceSetup.get_calibration].

        !!! version-changed "Deprecated in version 2.52.0"
            Use `.device_setup.get_calibration()` instead.
        """
        if self.device_setup is None:
            return None
        return self.device_setup.get_calibration()

    @property
    @deprecated(
        "The `.experiment_calibration` property is deprecated. Use"
        " `.experiment.get_calibration()` instead.",
        category=FutureWarning,
    )
    def experiment_calibration(self):
        """Return the experiment calibration.

        !!! version-changed "Deprecated in version 2.52.0"
            Use `.experiment.get_calibration()` instead.
        """
        if self.experiment is None:
            return None
        return self.experiment.get_calibration()

    @property
    @deprecated(
        "The `.signal_map` property is deprecated. Use"
        " `.experiment.get_signal_map()` instead.",
        category=FutureWarning,
    )
    def signal_map(self):
        """Return the experiment signal map.

        !!! version-changed "Deprecated in version 2.52.0"
            Use `.experiment.get_signal_map()` instead.
        """
        if self.experiment is None:
            return None
        return self.experiment.get_signal_map()
