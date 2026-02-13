# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.iface_compiler_output import CombinedOutput
from laboneq.compiler.common.resource_usage import (
    ResourceUsage,
    ResourceUsageCollector,
    UsageClassification,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.numpy_support import NumPyArray
from laboneq.data.awg_info import AWGInfo, AwgKey
from laboneq.data.compilation_job import DeviceInfo
from laboneq.data.recipe import IntegratorAllocation
from laboneq.data.scheduled_experiment import (
    HandleResultShape,
    ResultShapeInfo,
    ResultSource,
    RtLoopProperties,
)
from laboneq.executor.executor import (
    ExecutorBase,
    LoopFlags,
    LoopingMode,
    Statement,
)


class _Parameter:
    def __init__(self, id: str, values: NumPyArray):
        self.id = id

        sorted_indices = np.argsort(values)
        self._sorted_values = values[sorted_indices]
        self._sorted_indices = sorted_indices

    def index_of(self, value) -> int:
        idx_sorted = np.searchsorted(self._sorted_values, value)
        return self._sorted_indices[idx_sorted]


@dataclass
class _LoopStackEntry:
    count: int
    is_chunked: bool
    axis_names: list[str] = field(default_factory=list)
    axis_points: list[NumPyArray | Iterable[int]] = field(default_factory=list)
    sweep_params: list[_Parameter] = field(default_factory=list)

    @property
    def axis_name(self) -> str | list[str]:
        return self.axis_names[0] if len(self.axis_names) == 1 else self.axis_names

    @property
    def axis(self) -> NumPyArray | list[NumPyArray]:
        if len(self.axis_points) == 1:
            return np.array(self.axis_points[0])
        return [np.array(axis) for axis in self.axis_points]


@dataclass
class _MatchContextStackEntry:
    uid: uuid.UUID
    parameter: str | None
    current_state: int | None


@dataclass
class _HandleResultShape:
    """Mostly mimics HandleResultShape, types of fields may be different,
    and it contains more field(s) that are relevant for intermediate computations
    in this file, but irrelevant when final HandleResultShape instances are exported.
    """

    signal: str
    shape: tuple[int, ...]
    axis_names: list[str | list[str]]
    axis_values: list[NumPyArray | list[NumPyArray]]
    chunked_axis_index: int | None
    match_case_mask: dict[int, list[int]] | None

    match_context_uid: uuid.UUID | None

    def to_HandleResultShape(self) -> HandleResultShape:
        return HandleResultShape(
            signal=self.signal,
            shape=self.shape,
            axis_names=self.axis_names,
            axis_values=self.axis_values,
            chunked_axis_index=self.chunked_axis_index,
            match_case_mask=self.match_case_mask,
        )


def _merge_handle_result_shapes(
    handle: str,
    shapes: list[_HandleResultShape],
) -> HandleResultShape:
    """
    If there are multiple shapes for the same handle, and they are compatible, we combine them:
        1. If respective acquisitions are inside match-case, match-case masks are combined
        2. If not, a bigger shape where the last axis is for the multiple handles.
        3. If shapes are incompatible for any of the operations above, raises error.
    """

    if len(shapes) == 1:
        return shapes[0].to_HandleResultShape()

    assert len(shapes) != 0, "no result shapes to merge"
    first_shape = shapes[0]
    combined_mask = (
        {k: set(v) for k, v in first_shape.match_case_mask.items()}
        if first_shape.match_case_mask is not None
        else None
    )
    for shape_info in shapes[1:]:
        if shape_info.shape != first_shape.shape:
            raise LabOneQException(
                f"Multiple acquire events with the same handle ('{handle}') and different result shapes are not allowed."
            )

        if (shape_info.match_case_mask is None) ^ (combined_mask is None):
            raise LabOneQException(
                f"Cannot use the same handle ('{handle}') in a different context if already used inside a match-case context"
            )

        if shape_info.match_context_uid != first_shape.match_context_uid:
            raise LabOneQException(
                f"Cannot use the same handle ('{handle}') in different match-case contexts"
            )

        if shape_info.match_case_mask is not None and combined_mask is not None:
            for axis_idx, rows in shape_info.match_case_mask.items():
                rows_set = set(rows)
                if rows_set.intersection(combined_mask[axis_idx]):
                    raise LabOneQException(
                        f"Cannot use the same handle ('{handle}') multiple times in a single case context of a match section"
                    )
                combined_mask[axis_idx].update(rows)

    combined_shape = first_shape.to_HandleResultShape()
    if combined_mask is None:
        combined_shape.shape = (
            *combined_shape.shape,
            len(shapes),
        )
        combined_shape.axis_names.append(handle)
        combined_shape.axis_values.append(np.arange(len(shapes), dtype=np.float64))
    else:
        # the shapes are in match-case, so we don't add extra dimension for handle, just update the mask
        combined_shape.match_case_mask = {
            k: sorted(v) for k, v in combined_mask.items()
        }

    return combined_shape


class _ResultShapeExtractor(ExecutorBase):
    def __init__(
        self,
        rt_loop_properties: RtLoopProperties,
        combined_compiler_output: CombinedOutput,
    ):
        super().__init__(looping_mode=LoopingMode.ONCE)
        self._rt_loop_properties = rt_loop_properties
        self._combined_compiler_output = combined_compiler_output

        self._handle_result_shapes: dict[str, list[_HandleResultShape]] = defaultdict(
            list
        )
        self._loop_stack: list[_LoopStackEntry] = []
        self._match_context_stack: list[_MatchContextStackEntry] = []

    def get_result_shapes(self) -> dict[str, HandleResultShape]:
        return {
            handle: _merge_handle_result_shapes(handle, shape_infos)
            for handle, shape_infos in self._handle_result_shapes.items()
        }

    def _single_shot_axis(self) -> range:
        # The number of averages may potentially be large,
        # so we represent it with a lazy iterator.
        return range(self._rt_loop_properties.shots)

    def _match_case_mask(self) -> dict[int, list[int]] | None:
        if len(self._match_context_stack) == 0:
            return None

        mask = {}
        for match_context in self._match_context_stack:
            if match_context.parameter is None:
                continue

            assert match_context.current_state is not None
            loop_idx, case_idx = (None, None)
            for i, loop in enumerate(self._loop_stack):
                for p in loop.sweep_params:
                    if match_context.parameter == p.id:
                        loop_idx = i
                        case_idx = p.index_of(match_context.current_state)
                        break

            # This can happen if matching against a parameter that is not used in any sweep,
            # but this means invalid experiment so should have been caught earlier in some validation.
            assert loop_idx is not None and case_idx is not None
            mask[loop_idx] = [case_idx]

        return mask

    def _is_relevant_loop(self, loop_flags: LoopFlags) -> bool:
        # Averaging loop in non single shot mode is irrelevant for result shapes (does not introduce an axis)
        return (
            not loop_flags.is_average
            or self._rt_loop_properties.averaging_mode == AveragingMode.SINGLE_SHOT
        )

    def acquire_handler(self, handle: str, signal: str):
        shape = [loop.count for loop in self._loop_stack]
        axis_name = [loop.axis_name for loop in self._loop_stack]
        axis = [loop.axis for loop in self._loop_stack]

        # Append extra dimension for samples of the raw acquisition
        if self._rt_loop_properties.acquisition_type is AcquisitionType.RAW:
            raw_acquire_length = self._combined_compiler_output.get_raw_acquire_length(
                signal, handle
            )
            shape.append(raw_acquire_length)
            axis_name.append("samples")
            axis.append(np.arange(raw_acquire_length))

        chunked_axis_index = next(
            (i for i, loop in enumerate(self._loop_stack) if loop.is_chunked),
            None,
        )
        self._handle_result_shapes[handle].append(
            _HandleResultShape(
                signal=signal,
                shape=tuple(shape),
                axis_names=axis_name,
                axis_values=axis,
                chunked_axis_index=chunked_axis_index,
                match_case_mask=self._match_case_mask(),
                match_context_uid=self._match_context_stack[-1].uid
                if self._match_context_stack
                else None,
            )
        )

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: NumPyArray
    ):
        self._loop_stack[-1].axis_names.append(name if axis_name is None else axis_name)
        self._loop_stack[-1].axis_points.append(values)
        self._loop_stack[-1].sweep_params.append(_Parameter(name, values))

    def for_loop_entry_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if not self._is_relevant_loop(loop_flags):
            return
        self._loop_stack.append(
            _LoopStackEntry(
                count=count,
                is_chunked=bool(loop_flags & LoopFlags.CHUNKED),
            )
        )
        if loop_flags.is_average:
            # we are not supposed to be here if this is not true, since it is an irrelevant loop
            assert self._rt_loop_properties.averaging_mode is AveragingMode.SINGLE_SHOT
            self._loop_stack[-1].axis_names.append(self._rt_loop_properties.uid)
            self._loop_stack[-1].axis_points.append(self._single_shot_axis())

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if not self._is_relevant_loop(loop_flags):
            return
        self._loop_stack.pop()

    def match_context_entry_handler(self, sweep_parameter: str | None):
        self._match_context_stack.append(
            _MatchContextStackEntry(
                uid=uuid.uuid4(), parameter=sweep_parameter, current_state=None
            )
        )

    def match_context_exit_handler(self, sweep_parameter: str | None):
        self._match_context_stack.pop()

    def case_context_entry_handler(self, state: int):
        self._match_context_stack[-1].current_state = state

    def case_context_exit_handler(self, state: int):
        assert self._match_context_stack[-1].current_state == state
        self._match_context_stack[-1].current_state = None


def construct_result_shape_info(
    execution: Statement,
    rt_loop_properties: RtLoopProperties,
    awgs: list[AWGInfo],
    combined_compiler_output: CombinedOutput,
    device_infos: list[DeviceInfo],
    compiler_settings: CompilerSettings,
    integration_unit_allocation: list[IntegratorAllocation],
) -> ResultShapeInfo:
    extractor = _ResultShapeExtractor(rt_loop_properties, combined_compiler_output)
    extractor.run(execution)
    shapes = extractor.get_result_shapes()
    result_lengths = _calculate_result_lengths(
        rt_loop_properties,
        awgs,
        shapes,
        combined_compiler_output,
        compiler_settings,
        {di.uid: di for di in device_infos},
        integration_unit_allocation,
    )

    return ResultShapeInfo(
        shapes, combined_compiler_output.result_handle_maps, result_lengths
    )


def _calculate_result_lengths(
    rt_loop_properties: RtLoopProperties,
    awgs: list[AWGInfo],
    handle_result_shapes: dict[str, HandleResultShape],
    combined_compiler_output: CombinedOutput,
    compiler_settings: CompilerSettings,
    device_infos: dict[str, DeviceInfo],
    integration_unit_allocation: list[IntegratorAllocation],
) -> dict[AwgKey, int]:
    """Calculate the result length for each AWG core.

    Result lengths are not necessarily extractable from result shapes, due to specifics of different
    instruments. The result_handle_maps is a more reliable source of information to calculate result lengths.
    For more details take a look at the documentation of result_handle_maps and how they are constructed for
    different instruments.
    """
    result_lengths: dict[AwgKey, int] = {}
    res_usage_collector = ResourceUsageCollector()
    for awg in awgs:
        for sig in awg.signals:
            integration_unit = None
            if rt_loop_properties.acquisition_type != AcquisitionType.RAW:
                integration_unit = next(
                    (u for u in integration_unit_allocation if u.signal_id == sig.id),
                    None,
                )
                if integration_unit is None:
                    continue
                integration_unit = integration_unit.channels[0]

            result_source = ResultSource(awg.device_id, awg.awg_id, integration_unit)
            # NOTE: the mapping comes from a single RT artifact, hence it is already
            # adjusted for chunking. The result_length calculated below is basically
            # the result length for a single chunk of the experiment.
            mapping = combined_compiler_output.result_handle_maps.get(result_source)
            if mapping is None:
                continue

            result_length = len(mapping) * (
                rt_loop_properties.shots
                if rt_loop_properties.averaging_mode is AveragingMode.SINGLE_SHOT
                else 1
            )
            if rt_loop_properties.acquisition_type is AcquisitionType.RAW:
                max_segments = awg.device_type.scope_max_segments
                if max_segments is not None and result_length > max_segments:
                    raise LabOneQException(
                        f"A maximum of {max_segments} raw result(s) is supported per real-time execution."
                    )

                signal_handles = set(
                    h
                    for m in mapping
                    for h in m
                    if handle_result_shapes[h].signal == sig.id
                )
                for handle in signal_handles:
                    raw_acquire_samples = (
                        combined_compiler_output.get_raw_acquire_length(sig.id, handle)
                    )
                    scope_memory_consumption = result_length * raw_acquire_samples
                    scope_memory_size = awg.device_type.scope_memory_size_samples(
                        device_infos[awg.device_id]
                    )
                    if scope_memory_consumption > scope_memory_size:
                        raise LabOneQException(
                            "The total size of the requested raw traces exceeds the instrument's memory capacity."
                        )
            else:
                max_result_vector_length = awg.device_type.max_result_vector_length
                res_usage_collector.add(
                    ResourceUsage(
                        f"Result length for awg {awg.awg_id} on device {awg.device_id}",
                        result_length / max_result_vector_length
                        if max_result_vector_length is not None
                        else UsageClassification.WITHIN_LIMIT,
                    )
                )
            result_lengths[AwgKey(awg.device_id, awg.awg_id)] = result_length
    res_usage_collector.raise_or_pass(compiler_settings=compiler_settings)

    return result_lengths
