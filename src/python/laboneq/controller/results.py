# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.data.experiment_results import AcquiredResult

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.data.recipe import NtStepKey


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
):
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
        res_flat = np.ravel(_get_nt_step_result(result, nt_step))
        res_flat_idx = 0
        for raw_result_idx in range(len(raw_result)):
            if mapping[raw_result_idx % len(mapping)] == handle:
                res_flat[res_flat_idx] = raw_result[raw_result_idx]
                res_flat_idx += 1


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
