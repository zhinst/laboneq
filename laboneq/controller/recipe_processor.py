# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from numpy import typing as npt

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    AttributeValueTracker,
    DeviceAttribute,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IO, Initialization, Recipe, SignalType
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.executor.executor import (
    ExecutorBase,
    LoopFlags,
    LoopingMode,
    Sequence,
    Statement,
)

if TYPE_CHECKING:
    from laboneq.controller.devices.device_collection import DeviceCollection


@dataclass
class HandleResultShape:
    base_shape: list[int]
    base_axis_name: list[str | list[str]]
    base_axis: list[npt.ArrayLike | list[npt.ArrayLike]]
    additional_axis: int = 1


AcquireHandle = str
HandleResultShapes = dict[AcquireHandle, HandleResultShape]


@dataclass(frozen=True)
class AwgKey:
    device_uid: str
    awg_index: int


@dataclass
class AwgConfig:
    # QA
    raw_acquire_length: int | None = None
    result_length: int | None = None
    acquire_signals: set[str] = field(default_factory=set)
    target_feedback_register: int | None = None
    # SG
    command_table_match_offset: int | None = None
    source_feedback_register: int | None = None
    fb_reg_source_index: int | None = None
    fb_reg_target_index: int | None = None
    register_selector_bitmask: int = 0b11
    register_selector_shift: int | None = None


AwgConfigs = dict[AwgKey, AwgConfig]


@dataclass
class DeviceRecipeData:
    iq_settings: dict[int, npt.ArrayLike] = field(default_factory=dict)


DeviceId = str
DeviceSettings = dict[DeviceId, DeviceRecipeData]


@dataclass
class RtExecutionInfo:
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType
    pipeliner_chunk_count: int
    pipeliner_repetitions: int

    # signal id -> set of section ids
    acquire_sections: dict[str, set[str]] = field(default_factory=dict)

    # signal -> flat list of result handles
    # TODO(2K): to be replaced by event-based calculation in the compiler
    signal_result_map: dict[str, list[str]] = field(default_factory=dict)

    def add_acquire_section(self, signal_id: str, section_id: str):
        self.acquire_sections.setdefault(signal_id, set()).add(section_id)

    @staticmethod
    def get_acquisition_type(rt_execution_infos: RtExecutionInfos) -> AcquisitionType:
        # Currently only single RT execution per experiment supported
        rt_execution_info = next(iter(rt_execution_infos.values()), None)
        return RtExecutionInfo.get_acquisition_type_def(rt_execution_info)

    @staticmethod
    def get_acquisition_type_def(
        rt_execution_info: RtExecutionInfo | None,
    ) -> AcquisitionType:
        return (
            AcquisitionType.INTEGRATION
            if rt_execution_info is None
            else rt_execution_info.acquisition_type
        )

    def signal_by_handle(self, handle: str) -> str | None:
        return next(
            (
                signal
                for signal, handles in self.signal_result_map.items()
                if handle in handles
            ),
            None,
        )


RtSectionId = str
RtExecutionInfos = dict[RtSectionId, RtExecutionInfo]


@dataclass
class RecipeData:
    scheduled_experiment: ScheduledExperiment
    recipe: Recipe
    execution: Sequence
    result_shapes: HandleResultShapes
    rt_execution_infos: RtExecutionInfos
    device_settings: DeviceSettings
    awg_configs: AwgConfigs
    attribute_value_tracker: AttributeValueTracker
    oscillator_ids: list[str]

    @property
    def initializations(self) -> Iterator[Initialization]:
        for initialization in self.recipe.initializations:
            yield initialization

    def get_initialization_by_device_uid(self, device_uid: str) -> Initialization:
        for initialization in self.initializations:
            if initialization.device_uid == device_uid:
                return initialization

    def awgs_producing_results(self) -> Iterator[tuple[AwgKey, AwgConfig]]:
        for awg_key, awg_config in self.awg_configs.items():
            if awg_config.result_length is not None:
                yield awg_key, awg_config

    def awg_config_by_acquire_signal(self, signal_id: str) -> AwgConfig | None:
        return next(
            (
                awg_config
                for awg_config in self.awg_configs.values()
                if signal_id in awg_config.acquire_signals
            ),
            None,
        )


def _pre_process_iq_settings_hdawg(initialization: Initialization):
    # TODO(2K): Every pair of outputs with adjacent even+odd channel numbers (starting from 0)
    # is treated as an I/Q pair. I/Q pairs should be specified explicitly instead.

    # Base gains matrix (assuming ideal mixer, i.e. without calibration).
    # It ensures correct phases of I/Q components (correct sideband of
    # the resulting signal), along with the correct settings for:
    #   * playWave channel assignment: playWave(1, 2, I_wave, 1, 2, Q_wave)
    #   * oscillator phase: ch0 -> 0 deg / ch1 -> 90 deg
    #   * modulation mode: ch0 -> 3 (Sine12) / ch1 -> 4 (Sine21)

    outputs = initialization.outputs
    awgs = initialization.awgs
    iq_settings = {}

    for output in outputs or []:
        awg_idx = output.channel // 2

        # The channel already considered? Skip to the next.
        if awg_idx in iq_settings:
            continue

        # Do the outputs form an I/Q pair?
        awg = next((a for a in awgs if a.awg == awg_idx), None)
        if awg is None or awg.signal_type != SignalType.IQ:
            continue

        # Determine I and Q output elements for the IQ pair with index awg_idxs.
        if output.channel % 2 == 0:
            i_out = output
            q_out = next((o for o in outputs if o.channel == output.channel + 1), IO(0))
        else:
            i_out = next((o for o in outputs if o.channel == output.channel - 1), IO(0))
            q_out = output

        if i_out.gains is None or q_out.gains is None:
            continue  # No pair with valid gains found? This is not an IQ signal.

        iq_mixer_calib_mx = np.array(
            [
                [i_out.gains.diagonal, q_out.gains.off_diagonal],
                [i_out.gains.off_diagonal, q_out.gains.diagonal],
            ]
        )

        # Normalize resulting matrix to its inf-norm, to avoid clamping
        iq_mixer_calib_normalized = iq_mixer_calib_mx / np.linalg.norm(
            iq_mixer_calib_mx, np.inf
        )

        iq_settings[awg_idx] = iq_mixer_calib_normalized

    return iq_settings


@dataclass
class _LoopStackEntry:
    count: int
    is_averaging: bool
    axis_names: list[str] = field(default_factory=list)
    axis_points: list[npt.ArrayLike] = field(default_factory=list)

    @property
    def axis_name(self) -> str | list[str]:
        return self.axis_names[0] if len(self.axis_names) == 1 else self.axis_names

    @property
    def axis(self) -> npt.ArrayLike | list[npt.ArrayLike]:
        return self.axis_points[0] if len(self.axis_points) == 1 else self.axis_points


class _LoopsPreprocessor(ExecutorBase):
    def __init__(self):
        super().__init__(looping_mode=LoopingMode.ONCE)

        self.result_shapes: HandleResultShapes = {}
        self.rt_execution_infos: RtExecutionInfos = {}
        self.pipeliner_chunk_count: int = None
        self.pipeliner_repetitions: int = None

        self._loop_stack: list[_LoopStackEntry] = []
        self._current_rt_uid: str = None
        self._current_rt_info: RtExecutionInfo = None

    def _single_shot_axis(self) -> npt.ArrayLike:
        return np.linspace(
            0, self._current_rt_info.averages - 1, self._current_rt_info.averages
        )

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        self._current_rt_info.add_acquire_section(signal, parent_uid)

        # Determine result shape for each acquire handle
        single_shot_cyclic = (
            self._current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
        )
        shape = [
            loop.count
            for loop in self._loop_stack
            if not loop.is_averaging or single_shot_cyclic
        ]
        known_shape = self.result_shapes.get(handle)
        if known_shape is None:
            axis_name = [
                loop.axis_name
                for loop in self._loop_stack
                if not loop.is_averaging or single_shot_cyclic
            ]
            axis = [
                loop.axis
                for loop in self._loop_stack
                if not loop.is_averaging or single_shot_cyclic
            ]
            self.result_shapes[handle] = HandleResultShape(
                base_shape=shape, base_axis_name=axis_name, base_axis=axis
            )
        elif known_shape.base_shape == shape:
            known_shape.additional_axis += 1
        else:
            raise LabOneQControllerException(
                f"Multiple acquire events with the same handle ('{handle}') and different result shapes are not allowed."
            )

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: npt.ArrayLike
    ):
        self._loop_stack[-1].axis_names.append(name if axis_name is None else axis_name)
        self._loop_stack[-1].axis_points.append(values)

    def for_loop_entry_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if loop_flags.is_pipeline:
            self.pipeliner_chunk_count = count
            self.pipeliner_repetitions = math.prod(
                len(l.axis_points) for l in self._loop_stack
            )
            return

        self._loop_stack.append(
            _LoopStackEntry(count=count, is_averaging=loop_flags.is_average)
        )
        if loop_flags.is_average:
            single_shot_cyclic = (
                self._current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
            )
            if single_shot_cyclic:
                self._loop_stack[-1].axis_names.append(self._current_rt_uid)
                self._loop_stack[-1].axis_points.append(self._single_shot_axis())

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if loop_flags.is_pipeline:
            self.pipeliner_chunk_count = None
            self.pipeliner_repetitions = None
            return

        self._loop_stack.pop()

    def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        if averaging_mode != AveragingMode.SINGLE_SHOT:
            max_hw_averages = (
                pow(2, 15) if acquisition_type == AcquisitionType.RAW else pow(2, 17)
            )
            if count > max_hw_averages:
                raise LabOneQControllerException(
                    f"Maximum number of hardware averages is {max_hw_averages}, but {count} was given"
                )

        self._current_rt_uid = uid
        self._current_rt_info = self.rt_execution_infos.setdefault(
            uid,
            RtExecutionInfo(
                averages=count,
                averaging_mode=averaging_mode,
                acquisition_type=acquisition_type,
                pipeliner_chunk_count=self.pipeliner_chunk_count,
                pipeliner_repetitions=self.pipeliner_repetitions,
            ),
        )

    def rt_exit_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        if self._current_rt_info is None:
            raise LabOneQControllerException(
                "Nested 'acquire_loop_rt' are not allowed."
            )
        self._current_rt_uid = None
        self._current_rt_info = None


def _calculate_awg_configs(
    rt_execution_infos: RtExecutionInfos, recipe: Recipe
) -> AwgConfigs:
    awg_configs: AwgConfigs = defaultdict(AwgConfig)

    def awg_key_by_acquire_signal(signal_id: str) -> AwgKey:
        return next(
            awg_key
            for awg_key, awg_config in awg_configs.items()
            if signal_id in awg_config.acquire_signals
        )

    for a in recipe.integrator_allocations:
        if isinstance(a.signal_id, str):
            awg_configs[AwgKey(a.device_id, a.awg)].acquire_signals.add(a.signal_id)
        else:
            assert isinstance(a.signal_id, list) or isinstance(a.signal_id, tuple)
            awg_configs[AwgKey(a.device_id, a.awg)].acquire_signals.update(a.signal_id)

    for initialization in recipe.initializations:
        device_id = initialization.device_uid

        for awg in initialization.awgs or []:
            awg_config = awg_configs[AwgKey(device_id, awg.awg)]

            awg_config.target_feedback_register = awg.target_feedback_register
            awg_config.source_feedback_register = awg.source_feedback_register
            if awg_config.source_feedback_register not in (None, "local"):
                awg_config.fb_reg_source_index = awg.feedback_register_index_select
                awg_config.fb_reg_target_index = awg.awg
            awg_config.register_selector_shift = awg.codeword_bitshift
            awg_config.register_selector_bitmask = awg.codeword_bitmask
            awg_config.command_table_match_offset = awg.command_table_match_offset

    # As currently just a single RT execution per experiment is supported,
    # AWG configs are not cloned per RT execution. May need to be changed in the future.
    for rt_execution_info in rt_execution_infos.values():
        # Determine the raw acquisition lengths across various acquire events.
        # Will use the maximum length, as scope / monitor can only be configured for one.
        # device_id -> set of raw acquisition lengths
        raw_acquire_lengths: dict[str, set[int]] = defaultdict(set)
        if rt_execution_info.acquisition_type == AcquisitionType.RAW:
            for signal, sections in rt_execution_info.acquire_sections.items():
                awg_key = awg_key_by_acquire_signal(signal)
                for section in sections:
                    for al in recipe.acquire_lengths:
                        if al.signal_id == signal and al.section_id == section:
                            raw_acquire_lengths[awg_key.device_uid].add(
                                al.acquire_length
                            )
        for awg_key, awg_config in awg_configs.items():
            # Use dummy raw_acquire_length 4096 if there's no acquire statements in experiment
            awg_config.raw_acquire_length = max(
                raw_acquire_lengths.get(awg_key.device_uid, {4096})
            )

            # signal_id -> sequence of handle/None for each result vector entry.
            # Important! Length must be equal for all acquire signals / integrators of one AWG.
            # All integrators occupy an entry in the respective result vectors per startQA event,
            # regardless of the given integrators mask. Masked-out integrators just leave the
            # value at NaN (corresponds to None in the map).
            awg_result_map: dict[str, list[str]] = defaultdict(list)
            for acquires in recipe.simultaneous_acquires:
                if any(signal in acquires for signal in awg_config.acquire_signals):
                    for signal in awg_config.acquire_signals:
                        awg_result_map[signal].append(acquires.get(signal))
            if len(awg_result_map) > 0:
                rt_execution_info.signal_result_map.update(awg_result_map)
                # All lengths are the same, see comment above.
                any_awg_signal_result_map = next(iter(awg_result_map.values()))
                mapping_repeats = (
                    rt_execution_info.averages
                    if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT
                    else 1
                )
                awg_config.result_length = (
                    len(any_awg_signal_result_map) * mapping_repeats
                )

    return awg_configs


def _pre_process_attributes(
    recipe: Recipe, devices: DeviceCollection
) -> tuple[AttributeValueTracker, list[str]]:
    attribute_value_tracker = AttributeValueTracker()
    oscillator_ids: list[str] = []
    oscillators_check: dict[str, str | float] = {}

    for oscillator_param in recipe.oscillator_params:
        value_or_param = oscillator_param.param or oscillator_param.frequency
        if oscillator_param.id in oscillator_ids:
            osc_index = oscillator_ids.index(oscillator_param.id)
            if oscillators_check[oscillator_param.id] != value_or_param:
                raise LabOneQControllerException(
                    f"Conflicting specifications for the same oscillator id '{oscillator_param.id}' "
                    f"in the recipe: '{oscillators_check[oscillator_param.id]}' != '{value_or_param}'"
                )
        else:
            osc_index = len(oscillator_ids)
            oscillator_ids.append(oscillator_param.id)
            oscillators_check[oscillator_param.id] = value_or_param
        attribute_value_tracker.add_attribute(
            device_uid=oscillator_param.device_id,
            attribute=DeviceAttribute(
                name=AttributeName.OSCILLATOR_FREQ,
                index=osc_index,
                value_or_param=value_or_param,
            ),
        )

    for initialization in recipe.initializations:
        device = devices.find_by_uid(initialization.device_uid)
        for attribute in device.pre_process_attributes(initialization):
            attribute_value_tracker.add_attribute(
                device_uid=initialization.device_uid,
                attribute=attribute,
            )

    return attribute_value_tracker, oscillator_ids


def pre_process_compiled(
    scheduled_experiment: ScheduledExperiment,
    devices: DeviceCollection,
    execution: Statement,
) -> RecipeData:
    recipe = scheduled_experiment.recipe

    device_settings: DeviceSettings = defaultdict(DeviceRecipeData)
    for initialization in recipe.initializations:
        device_settings[initialization.device_uid] = DeviceRecipeData(
            iq_settings=_pre_process_iq_settings_hdawg(initialization)
        )

    lp = _LoopsPreprocessor()
    lp.run(execution)
    rt_execution_infos = lp.rt_execution_infos

    awg_configs = _calculate_awg_configs(rt_execution_infos, recipe)
    attribute_value_tracker, oscillator_ids = _pre_process_attributes(recipe, devices)

    recipe_data = RecipeData(
        scheduled_experiment=scheduled_experiment,
        recipe=recipe,
        execution=execution,
        result_shapes=lp.result_shapes,
        rt_execution_infos=rt_execution_infos,
        device_settings=device_settings,
        awg_configs=awg_configs,
        attribute_value_tracker=attribute_value_tracker,
        oscillator_ids=oscillator_ids,
    )

    return recipe_data


def get_wave(wave_name, waves: list[dict[str, Any]]):
    wave = next(
        (wave for wave in waves if wave.get("filename", None) == wave_name), None
    )
    if wave is None:
        raise LabOneQControllerException(
            f"Wave '{wave_name}' is not found in the compiled waves collection."
        )
    return np.ascontiguousarray(wave["samples"])
