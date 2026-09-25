# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import copy as copy_
from collections import UserDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from laboneq._optional_deps import (
    import_optional,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.validators import dicts_equal

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq._optional_deps import (
        Xarray,
        XarrayDataArray,
        XarrayDataset,
    )


@classformatter
@dataclass
class AcquiredResult:
    """
    This class represents the results acquired for a single result handle.

    The acquired result is a triple consisting of actual data, axis name(s)
    and one or more axes.

    Attributes:
        data (ndarray or complex):
            A multidimensional `numpy` array, where each dimension corresponds to a sweep loop
            nesting level, the outermost sweep being the first dimension. Or, if there is
            only a single measurement, a complex scalar.
        axis_name (list[str | list[str]]):
            A list of axis names.
            Each element may be either a string or a list of strings.
        axis (list[ndarray | list[ndarray]]):
            A list of axis grids.
            Each element may be either a 1D numpy array or a list of such arrays.
        last_nt_step (list[int]):
            A list of axis indices that represent the last measured near-time point.
            Only covers outer near-time dimensions.
        handle (str | None): Acquire handle used for capturing the results.

            !!! version-added "Added in version 2.16.0"
    """

    data: npt.NDArray[Any] | np.complex128
    axis_name: list[str | list[str]]
    axis: list[npt.NDArray[Any] | list[npt.NDArray[Any]]]
    last_nt_step: list[int] | None = None
    handle: str | None = None

    def __eq__(self, other: object):
        if not isinstance(other, AcquiredResult):
            return False
        return (
            dicts_equal(self.data, other.data)
            and self.axis_name == other.axis_name
            and dicts_equal(self.axis, other.axis)
            and self.last_nt_step == other.last_nt_step
            and self.handle == other.handle
        )

    def __repr__(self):
        return (
            f"<{type(self).__qualname__}"
            f" handle={self.handle!r}"
            f" axis_name={self.axis_name}"
            f" data.shape={np.shape(self.data)}"
            f">"
        )

    def __rich_repr__(self):
        yield "handle", self.handle
        yield "axis_name", self.axis_name
        yield "axis", self.axis
        yield "data", self.data
        yield "last_nt_step", self.last_nt_step

    def to_xarray(self, copy: bool = False) -> XarrayDataArray:
        """Convert to [xarray.DataArray][].

        Requires optional dependency [xarray][] to be installed.

        By default, the data in the returned object is a view of the `data`,
        and will get updated if it is created during experiment runtime.

        By setting `copy` to `True`, the returned data
        will be detached from the `AcquiredResult` data.

        Args:
            copy: If `True`, returns a copy of the data instead of a view.

        Returns:
            (DataArray): An [xarray.DataArray][] representation of the object.

                * The name is that of the result handle.
                * The coordinates correspond to the `axis_name` of the individual sweep parameters.
                * The dimensions have fixed names `axis_0`, `axis_1` etc..

        Raises:
            ModuleNotFoundError: If `xarray` is not installed.

        !!! version-changed "Changed in version 2.26.0"
            Names of the coordinates are now the values of the `axis_name` instead of
            generic `sweep_n`.

            Names of the dimensions are in the format of `axis_0`, `axis_1`etc.

        !!! version-added "Added in version 2.16.0"

        """
        xr: Xarray = import_optional(
            "xarray", message="Cannot convert `AcquiredResult` to `xarray` object."
        )
        if copy:
            axis_names = copy_.copy(self.axis_name)
            axes = copy_.copy(self.axis)
            data = copy_.deepcopy(self.data)
            handle = copy_.copy(self.handle)
        else:
            axis_names = self.axis_name
            axes = self.axis
            data = self.data
            handle = self.handle
        coords = {}
        axes_as_dims = [f"axis_{n}" for n in range(len(axis_names))]
        for idx, axis_name in enumerate(axis_names):
            if isinstance(axis_name, list):
                for idx_ax, axis in enumerate(axis_name):
                    coords[axis] = (axes_as_dims[idx], axes[idx][idx_ax])
            else:
                coords[axis_name] = (axes_as_dims[idx], axes[idx])
        return xr.DataArray(name=handle, data=data, dims=axes_as_dims, coords=coords)


def slice_acquired_result(
    acquired: AcquiredResult, nt: Literal["all", "latest"] = "all"
) -> AcquiredResult | None:
    """Restrict a result to the requested near-time selection.

    With ``nt="latest"`` the data is sliced at ``last_nt_step`` and the leading
    near-time axes are stripped; ``None`` is returned when no near-time step
    has completed yet. With ``nt="all"`` the result is returned unchanged.
    """
    if nt == "all":
        return acquired
    if acquired.last_nt_step is None:
        return None
    nt_index = tuple(acquired.last_nt_step)
    n_nt_axes = len(nt_index)
    assert acquired.data is not None
    # Indexing an ndarray returns a view into `acquired.data`, which later
    # near-time steps keep writing into; copy so the caller gets a stable
    # snapshot instead of data that changes under their feet.
    sliced_data = acquired.data[nt_index]
    if isinstance(sliced_data, np.ndarray):
        sliced_data = sliced_data.copy()
    return AcquiredResult(
        data=sliced_data,
        axis_name=list(acquired.axis_name[n_nt_axes:]),
        axis=list(acquired.axis[n_nt_axes:]),
        last_nt_step=list(nt_index),
        handle=acquired.handle,
    )


class AcquiredResults(UserDict[str, AcquiredResult]):
    """A collection of acquired results.

    Keys are handles for a single acquire event.
    Values are acquired results for that handle.

    !!! version-added "Added in version 2.16.0"
    """

    def _get_sliced(
        self, handle: str, nt: Literal["all", "latest"]
    ) -> AcquiredResult | None:
        """Return the result for `handle` restricted to the `nt` selection.

        Stores that keep their results on a service override this
        to fetch only the part asked for.

        Returns:
            The selected result, or `None` if no near-time step has completed
            yet for the handle.

        Raises:
            KeyError: The handle is unknown.
        """
        return slice_acquired_result(self[handle], nt)

    def _known_handles(self) -> set[str] | None:
        """The handles already held here, or `None` if finding out needs a fetch.

        This method is called by Results.__repr__, which must not download
        anything just to print a count.
        """
        return set(self.data)

    def to_xarray(self, copy: bool = False) -> XarrayDataset:
        """Convert to [xarray.Dataset][].

        Requires optional dependency [xarray][] to be installed.

        By default, the data in the returned object is a view of the `data`,
        and will get updated if it is created during experiment runtime.

        By setting `copy` to `True`, the returned data
        will be detached from the `AcquiredResults` data.

        !!! note
            In the case of axis data mismatch in the underlying `AcquiredResult`s, the conversion
            might not work. The mismatch can happen when multiple sweep parameters have the same axis name,
            but the values are different.

            Use `results["result_handle"].to_array()` to convert individual results to [xarray.Dataset][].

            This reads every result, so a store that fetches on demand
            downloads all of them.

        Args:
            copy: If `True`, returns a copy of the data instead of a view.

        Returns:
            (Dataset): An [xarray.Dataset][] of all acquired results.

        Raises:
            ModuleNotFoundError: If `xarray` is not installed.
            LabOneQException: Individual results cannot be merged

        !!! version-changed "Changed in version 2.26.0"

            Name of the coordinates is now the values of the axis names instead of
            `sweep_n`.
        """
        xr: Xarray = import_optional(
            "xarray", message="Cannot convert `AcquiredResults` to `xarray` object."
        )
        # Read through the mapping, not `self.data`: a store that fetches on
        # demand keeps only what was asked for in there.
        datasets = [res.to_xarray(copy=copy) for res in self.values()]
        try:
            ds = xr.merge(
                datasets, combine_attrs="drop_conflicts", compat="broadcast_equals"
            )
        except xr.MergeError as error:
            # NOTE: Sweep parameters are allowed to have duplicate `axis_name`, but not `uid`.
            #   If duplicate `axis_name` is used for different parallel sweep parameters, it is not
            #   compatible with `xarray`.
            raise LabOneQException(
                "Cannot merge results. Use `results['<handle>'].to_array()` instead."
            ) from error
        ds.attrs = {"description": "A collection of acquired results."}
        return ds


@dataclass
class ExperimentResults:
    uid: str | None = None

    #: The acquired results, organized by handle.
    acquired_results: AcquiredResults = field(default_factory=AcquiredResults)

    #: List of the results of each near-time callback, by name of the function.
    neartime_callback_results: dict[str, list[Any]] = field(default_factory=dict)

    #: Any exceptions that occurred during the execution of the experiment. Entries are
    #: tuples of
    #:
    #: * the indices of the loops where the error occurred,
    #: * the section uid,
    #: * the error message.
    execution_errors: list[tuple[list[int], str, str]] = field(default_factory=list)

    # Pipeline job timestamps, by device channel id, then job id
    pipeline_jobs_timestamps: dict[str, list[float]] = field(default_factory=dict)

    execution_payload_hash: str | None = None
