# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.data.experiment_results import AcquiredResult, ExperimentResults

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.data.recipe import NtStepKey
    from laboneq.controller.recipe_processor import RecipeData


def init_empty_result_by_shape(recipe_data: RecipeData) -> ExperimentResults:
    results = ExperimentResults()
    for handle, shape_info in recipe_data.result_shapes.items():
        empty_result = make_acquired_result(
            data=np.full(
                shape=tuple(shape_info.base_shape),
                fill_value=np.nan,
                dtype=np.complex128,
            ),
            axis_name=shape_info.base_axis_name,
            axis=shape_info.base_axis,
            handle=handle,
        )
        results.acquired_results[handle] = empty_result
    return results


def make_acquired_result(
    data: NumPyArray,
    axis_name: list[str | list[str]],
    axis: list[NumPyArray | list[NumPyArray]],
    handle: str,
) -> AcquiredResult:
    """Factory function to decouple controller code from the public data model"""
    return AcquiredResult(data, axis_name, axis, handle=handle)


def _get_nt_step_result(result: AcquiredResult, nt_step: NtStepKey):
    inner_res = result.data
    for index in nt_step.indices:
        inner_res = inner_res[index]
    return inner_res


def build_partial_result(
    result: AcquiredResult,
    nt_step: NtStepKey,
    raw_result: Any,
    mapping: list[str | None],
    handle: str,
    pipeline_job_count: int | None,
    chunked_axis_index: int | None,
):
    """Populates result in-place with raw_result.
    raw_result is what is obtained from the instrument and contains a flat sequence of numbers.
    result is a buffer for the complete result to be returned to the user, that contains NaNs in
    locations where this function is about to populate.
    raw_result itself may contain NaNs when acquiring failed, e.g. execution of one chunk errored out.
    """
    assert result.data is not None, "Result data shape is not prepared"
    result.last_nt_step = list(nt_step.indices)
    if len(np.shape(result.data)) == len(nt_step.indices):
        # No loops in RT, just a single value produced
        for raw_result_idx in range(len(raw_result)):
            if mapping[raw_result_idx % len(mapping)] == handle:
                if len(nt_step.indices) == 0:
                    result.data = raw_result[raw_result_idx]
                else:
                    result.data[nt_step.indices] = raw_result[raw_result_idx]
                break
    else:
        raw_result_len = len(raw_result)
        mask = np.fromiter(
            (mapping[i % len(mapping)] == handle for i in range(raw_result_len)),
            dtype=np.bool,
            count=raw_result_len,
        )
        raw_result_for_handle = raw_result[mask]

        if pipeline_job_count:
            # rearrange the data as if it came from a non-chunked experiment

            assert chunked_axis_index is not None
            rt_chunk_shape = list(result.data.shape[len(nt_step.indices) :])
            rt_chunked_axis_index = chunked_axis_index - len(nt_step.indices)
            rt_chunk_shape[rt_chunked_axis_index] //= pipeline_job_count
            # When there are multiple acquires with the same handle, there is an extra dimension
            # at the end that handles this. Here we do not care if the last dimension is such
            # extra dimension, or is ordinary sweep dimension. However, in the case of extra
            # dimension, there is a known problem where the value of the dimension may be incorrect.
            # This happens when the acquires are in case blocks. To mitigate for this issue, we
            # do not require strict size for the last dimension. We can do this since all other
            # dimensions are strict.
            rt_chunk_shape[-1] = -1

            shaped_raw_result_for_handle = np.reshape(
                raw_result_for_handle, (pipeline_job_count, *rt_chunk_shape)
            )
            raw_result_for_handle = np.ravel(
                np.concatenate(shaped_raw_result_for_handle, axis=rt_chunked_axis_index)
            )
        res_flat = np.ravel(_get_nt_step_result(result, nt_step))
        # Ideally we should be able to do res_flat[:] here, since the number of items should be the same.
        # Currently we cannot, because of the known issue of mismatch between expected and actual result
        # shapes in case of acquisition being inside match-case blocks. TODO(hs): Fix this once result
        # shaping is overhauled.
        res_flat[:raw_result_len] = raw_result_for_handle


def build_raw_partial_result(
    result: AcquiredResult,
    nt_step: NtStepKey,
    raw_segments: NumPyArray,
    result_length: int,
    mapping: list[str | None],
    handle: str,
):
    assert result.data is not None, "Result data shape is not prepared"
    result.last_nt_step = list(nt_step.indices)
    inner_res = _get_nt_step_result(result, nt_step)
    # The remaining dimensions correspond to the RT loops and multiple acquires
    # and are flattened, except for the last one, which is for the raw wave samples.
    acquires = np.multiply.reduce(inner_res.shape[:-1], initial=1, dtype=int)
    res_flat = np.reshape(inner_res, (acquires, result_length))
    res_flat_idx = 0
    for raw_result_idx in range(len(raw_segments)):
        if mapping[raw_result_idx % len(mapping)] == handle:
            res_flat[res_flat_idx] = raw_segments[raw_result_idx][0:result_length]
            res_flat_idx += 1
