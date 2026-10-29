# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import attrs
from typing_extensions import TypeAlias

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.attribute_wrapper import AttributeWrapper
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .acquired_result import AcquiredResults

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy import typing as npt

    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment

    from .acquired_result import AcquiredResult
    from .logged_variable import LoggedVariable


ErrorList: TypeAlias = list[tuple[list[int], str, str]]


def _as_acquired_results(
    value: AcquiredResults | Mapping[str, AcquiredResult] | None,
) -> AcquiredResults:
    """Accept any handle-to-result mapping, such as a plain dict."""
    if isinstance(value, AcquiredResults):
        return value
    return AcquiredResults(value or {})


@classformatter
@attrs.define
class Results:
    """Results of an LabOne Q experiment.

    In addition to being accessible via the attributes and methods defined below,
    the `acquired_results` are also accessible directly on the result object
    itself using either attribute or dictionary-like item lookup.

    For example:

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

    !!! note
        Inside a client near-time callback the results are still on the
        controller service. Asking which handles exist, how many there are or
        whether one is present downloads no data. Reading a single handle, by
        key or through `get_result`/`get_data`, downloads only that handle.
        `results.cal_trace` and other group lookups download everything,
        because the group they hand back holds the results.

    Attributes:
        experiment:
            The source experiment. Deprecated.
        device_setup:
            The device setup on which the experiment was run. Deprecated.
        acquired_results:
            The acquired results, organized by handle.
        variable_results:
            The recorded values of logged variables, organized by the handle
            they were logged under.
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

    !!! version-removed "Removed in version 26.10.0"
        Removed the `.device_calibration`, `.experiment_calibration` and
        `.signal_map` properties that were deprecated in version 2.52.0. Use
        `.device_setup.get_calibration()`, `.experiment.get_calibration()` and
        `.experiment.get_signal_map()` instead.

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

    !!! version-changed "Deprecated in version 2.52.0"
        The `experiment`, `device_setup` and `compiled_experiment` attributes
        were deprecated in version 2.52.0 and are only populated when
        requested via `Session` or `Session.run`.
    """

    experiment: Experiment = attrs.field(default=None)
    device_setup: DeviceSetup = attrs.field(default=None)
    acquired_results: AcquiredResults = attrs.field(
        factory=AcquiredResults, converter=_as_acquired_results
    )
    variable_results: dict[str, LoggedVariable] = attrs.field(factory=dict)
    neartime_callback_results: dict[str, list[Any]] = attrs.field(default=None)
    execution_errors: list[tuple[list[int], str, str]] = attrs.field(factory=list)
    pipeline_jobs_timestamps: dict[str, list[float]] = attrs.field(factory=dict)

    # Support for deprecated init parameters:
    _data: AcquiredResults | None = attrs.field(default=None)

    _acquired_results_wrapper: AttributeWrapper | None = attrs.field(
        init=False, default=None, eq=False
    )
    _neartime_callbacks_wrapper: AttributeWrapper | None = attrs.field(
        init=False, default=None, eq=False
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

    def _wrap_acquired_results(self) -> AttributeWrapper:
        # AttributeWrapper iterates all keys of the wrapped mapping when it is
        # built. The results mappings may fetch their content on demand (e.g.
        # from a session while the experiment is still running), so defer
        # building the wrappers until the data is actually accessed.
        if self._acquired_results_wrapper is None:
            self._acquired_results_wrapper = AttributeWrapper(self.acquired_results)
        return self._acquired_results_wrapper

    def _wrap_neartime_callbacks(self) -> AttributeWrapper:
        if self._neartime_callbacks_wrapper is None:
            self._neartime_callbacks_wrapper = AttributeWrapper(
                self.neartime_callback_results
            )
        return self._neartime_callbacks_wrapper

    def __repr__(self) -> str:
        # Printing must not start a download, so a store that has not fetched
        # its handles yet reports "?" instead of a count.
        handles = self.acquired_results._known_handles()
        return (
            f"<{type(self).__qualname__}"
            f" id={id(self)}"
            f" errors={bool(self.execution_errors)}"
            f" len(acquired_results)={len(handles) if handles is not None else '?'}"
            f" len(neartime_callbacks)={len(self.neartime_callback_results)}"
            ">"
        )

    def __rich_repr__(self):
        yield "errors", self.errors
        yield "data", self.data
        yield "neartime_callbacks", self.neartime_callbacks

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str) and key in self.acquired_results:
            return True
        return key in self._wrap_acquired_results()

    def __deepcopy__(self, memo: dict) -> Results:
        return self.__class__(
            experiment=copy.deepcopy(self.experiment),
            device_setup=copy.deepcopy(self.device_setup),
            acquired_results=copy.deepcopy(self.acquired_results),
            variable_results=copy.deepcopy(self.variable_results),
            neartime_callback_results=copy.deepcopy(self.neartime_callback_results),
            execution_errors=copy.deepcopy(self.execution_errors),
            pipeline_jobs_timestamps=copy.deepcopy(self.pipeline_jobs_timestamps),
        )

    def __dir__(self):
        return super().__dir__() + self._wrap_acquired_results()._attr_keys()

    def __getattr__(self, key: object) -> AttributeWrapper | object:
        # This intentionally only looks up the keys on the wrapper and
        # not arbitrary attributes
        #
        # Names starting with an underscore are never result handles: dir()
        # asks for __dict__, notebooks ask for _repr_html_. Looking those up in
        # the results would fetch all of them. results["_name"] still works.
        if isinstance(key, str) and key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(*err.args) from None

    def __getitem__(self, key: object) -> AttributeWrapper | ErrorList | object:
        # A handle needs no wrapper, and building one would fetch every result
        # from a store that keeps them on a service. Checking for the handle
        # first costs that one handle at most.
        if isinstance(key, str) and key in self.acquired_results:
            return self.acquired_results[key]
        return self._wrap_acquired_results().__getitem__(key)

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
        return self._wrap_acquired_results()

    @property
    def neartime_callbacks(self) -> AttributeWrapper:
        """The results of the near-time user callbacks.

        !!! version-added "Added in version 2.52.0"
            The `.neartime_callbacks` attribute was added.
            The name was chosen to match that used on the (now removed)
            `RunExperimentResults` class.
        """
        return self._wrap_neartime_callbacks()

    def _check_handle(self, handle: str) -> None:
        if handle not in self.acquired_results:
            raise LabOneQException(f"No result for handle: {handle}")

    def get_result(
        self, handle: str, nt: Literal["all", "latest"] = "all"
    ) -> AcquiredResult:
        """Returns the acquired result.

        Returns the result acquired for an 'acquire' event with the specific handle
        that was assigned to it in the experiment definition.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.
            nt: Near-time selection: `"all"` returns everything measured so
                far, `"latest"` the slice at the most recently completed near-time step
                with the near-time axes stripped.

        Returns:
            result: The acquire event result.

        Raises:
            LabOneQException: No result is available for the provided handle,
                or `nt="latest"` was requested before any near-time step completed.

        !!! warning
            The `nt` selector is experimental and may change.
        """
        if nt not in ("all", "latest"):
            raise LabOneQException(f"Invalid nt={nt!r}: expected 'all' or 'latest'.")
        try:
            result = self.acquired_results._get_sliced(handle, nt)
        except KeyError:
            raise LabOneQException(f"No result for handle: {handle}") from None
        if result is None:
            raise LabOneQException(
                f"No completed near-time step yet for handle: {handle}"
            )
        return result

    def get_data(
        self, handle: str, nt: Literal["all", "latest"] = "all"
    ) -> npt.NDArray[Any] | np.complex128:
        """Returns the acquired result data.

        Returns the result acquired for an 'acquire' event with the specific handle
        that was assigned to it in the experiment definition.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.
            nt: Near-time selection: `"all"` returns everything measured so
                far, `"latest"` the slice at the most recently completed near-time step
                with the near-time axes stripped.

        Returns:
            A multidimensional numpy array, where each dimension corresponds to a sweep
                loop nesting level, the outermost sweep being the first dimension.

        Raises:
            LabOneQException: No result is available for the provided handle,
                or `nt="latest"` was requested before any near-time step completed.

        !!! warning
            The `nt` selector is experimental and may change.
        """

        return self.get_result(handle, nt).data

    def get_axis_name(
        self, handle: str, nt: Literal["all", "latest"] = "all"
    ) -> list[str | list[str]]:
        """Returns the names of axes.

        Returns the list of axis names, that correspond to the dimensions of the result returned by
        'get'. Elements in the list are in the same order as the dimensions of the array returned by
        'get'. Each element is either a string for a simple sweep, or a list of strings for a parallel
        sweep. Values are given by the 'axis_name' argument of the corresponding sweep parameter, or
        the 'uid' of the same parameter, if 'axis_name' is not specified.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.
            nt: Near-time selection: `"all"` returns everything measured so
                far, `"latest"` the slice at the most recently completed near-time step
                with the near-time axes stripped.

        Returns:
            A list of axis names. Each element may be either a string or a list of strings.

        Raises:
            LabOneQException: No result is available for the provided handle,
                or `nt="latest"` was requested before any near-time step completed.

        !!! warning
            The `nt` selector is experimental and may change.
        """

        return self.get_result(handle, nt).axis_name

    def get_axis(
        self, handle: str, nt: Literal["all", "latest"] = "all"
    ) -> list[npt.NDArray[Any] | list[npt.NDArray[Any]]]:
        """Returns the axes grids.

        Returns the list, where each element represents an axis of the corresponding dimension of
        the result array returned by 'get'. Each element is either a 1D numpy array for a simple
        sweep, or a list of 1D numpy arrays for a parallel sweep. The length of each array matches
        the number of steps of the corresponding sweep, and the values are the sweep parameter
        values at each step.

        Args:
            handle (str): The handle assigned to an 'acquire' event in the experiment definition.
            nt: Near-time selection: `"all"` returns everything measured so
                far, `"latest"` the slice at the most recently completed near-time step
                with the near-time axes stripped.

        Returns:
            A list of axis grids. Each element may be either a 1D numpy array or a list of such
                arrays.

        Raises:
            LabOneQException: No result is available for the provided handle,
                or `nt="latest"` was requested before any near-time step completed.

        !!! warning
            The `nt` selector is experimental and may change.
        """

        return self.get_result(handle, nt).axis

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


def combine_results(results: Sequence[Results]) -> Results:
    """Combine multiple results into one

    Combine multiple `Results` objects into one. The fields are merged according
    to the following rules:

     * `acquired_results` and `neartime_callback_results`: In case of overlapping
       handles, results that appear later in the sequence overwrite earlier ones.
       Note that this currently applies for identical handles and handles that
       share a prefix. This behavior may change in the future, resulting
       in an error when handles overlap.
     * `pipeline_jobs_timestamps`: As in the previous two fields, later results
       overwrite earlier ones. As the access semantics for this field are different,
       keys with common prefixes are not affected.
     * `execution_errors`: The combined result contains a concatenated list of all
       errors from all results.

    Args:
        results: A sequence of `Results` objects.

    Returns:
        The combined `Results` object.
    """
    acquired_results = AcquiredResults()
    execution_errors = []
    neartime_callback_results = {}
    pipeline_jobs_timestamps = {}
    variable_results: dict = {}

    for res in results:
        acquired_results.update(res.acquired_results)
        execution_errors.extend(res.execution_errors or [])
        neartime_callback_results.update(res.neartime_callback_results or {})
        pipeline_jobs_timestamps.update(res.pipeline_jobs_timestamps or {})
        variable_results.update(res.variable_results or {})

    return Results(
        acquired_results=acquired_results,
        execution_errors=execution_errors,
        neartime_callback_results=neartime_callback_results,
        pipeline_jobs_timestamps=pipeline_jobs_timestamps,
        variable_results=variable_results,
    )
