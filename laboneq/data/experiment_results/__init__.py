# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import copy as copy_
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import ArrayLike

from laboneq._optional_deps import (
    Xarray,
    XarrayDataArray,
    XarrayDataset,
    import_optional,
)
from laboneq.core.validators import dicts_equal


@dataclass
class AcquiredResult:
    """
    This class represents the results acquired for a single result handle.

    The acquired result is a triple consisting of actual data, axis name(s)
    and one or more axes.

    Attributes:
        data (ArrayLike): A multidimensional `numpy` array, where each dimension corresponds to a sweep loop
            nesting level, the outermost sweep being the first dimension.
        axis_name (list[str | list[str]]): A list of axis names.
            Each element may be either a string or a list of strings.
        axis (list[ArrayLike | list[ArrayLike]]): A list of axis grids.
            Each element may be either a 1D numpy array or a list of such arrays.
        last_nt_step (list[int]): A list of axis indices that represent the last measured near-time point.
            Only covers outer near-time dimensions.
        handle (str | None): Acquire handle used for capturing the results.

            !!! version-added "Added in version 2.16.0"
    """

    data: ArrayLike | None = None
    axis_name: list[str | list[str]] = field(default_factory=list)
    axis: list[ArrayLike | list[ArrayLike]] = field(default_factory=list)
    last_nt_step: list[int] | None = None
    handle: str | None = None

    def __eq__(self, other: AcquiredResult):
        return (
            dicts_equal(self.data, other.data)
            and self.axis_name == other.axis_name
            and dicts_equal(self.axis, other.axis)
            and self.last_nt_step == other.last_nt_step
            and self.handle == other.handle
        )

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

                * Name is the name of the result handle.
                * The dimensions and coordinates are individual sweeps in the format of:
                    `sweep_0`, `sweep_1` .. `sweep_n`.

        Raises:
            ModuleNotFoundError: If `xarray` is not installed.

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
        coords = []
        dims = []
        sweep_params = []
        for i, axis in enumerate(axis_names):
            dim = f"sweep_{i}"
            dims.append(dim)
            coords.append((dim, axes[i]))
            sweep_params.append({"coord": dim, "param": axis})
        data = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs={
                "description": "Acquired results",
                "sweep_params": sweep_params,
            },
        )
        if handle is not None:
            data.name = handle
        return data


class AcquiredResults(UserDict[str, AcquiredResult]):
    """A collection of acquired results.

    Keys are handles for a single acquire event.
    Values are acquired results for that handle.

    !!! version-added "Added in version 2.16.0"
    """

    def to_xarray(self, copy: bool = False) -> XarrayDataset:
        """Convert to [xarray.Dataset][].

        Requires optional dependency [xarray][] to be installed.

        By default, the data in the returned object is a view of the `data`,
        and will get updated if it is created during experiment runtime.

        By setting `copy` to `True`, the returned data
        will be detached from the `AcquiredResults` data.

        Args:
            copy: If `True`, returns a copy of the data instead of a view.

        Returns:
            (Dataset): An [xarray.Dataset][] of all acquired results.

        Raises:
            ModuleNotFoundError: If `xarray` is not installed.
        """
        xr: Xarray = import_optional(
            "xarray", message="Cannot convert `AcquiredResults` to `xarray` object."
        )
        count = 0
        xarrs = []
        for arr in self.data.values():
            xarr = arr.to_xarray(copy=copy)
            # Backwards compatibility with serializer (laboneq <=2.15)
            if xarr.name is None:
                xarr.name = f"handle_{count}"
                count += 1
            xarrs.append(xarr)
        ds = xr.merge(
            xarrs,
            combine_attrs="drop",
        )
        ds.attrs = {"description": "A collection of acquired results."}
        return ds


@dataclass
class ExperimentResults:
    uid: str = None

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

    experiment_hash: str = None
    compiled_experiment_hash: str = None
    execution_payload_hash: str = None
