# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from numpy import typing as npt

from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.executor.execution_from_experiment import ExecutionFactoryFromExperiment
from laboneq.executor.executor import (
    ExecutorBase,
    LoopingMode,
    LoopType,
    Sequence,
    Statement,
)

from .recipe_1_4_0 import IO
from .recipe_1_4_0 import Experiment as RecipeExperiment
from .recipe_1_4_0 import Initialization, Recipe
from .recipe_enums import SignalType

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment


@dataclass
class HandleResultShape:
    base_shape: List[int]
    base_axis_name: List[Union[str, List[str]]]
    base_axis: List[Union[npt.ArrayLike, List[npt.ArrayLike]]]
    additional_axis: int = 1


AcquireHandle = str
HandleResultShapes = Dict[AcquireHandle, HandleResultShape]


@dataclass(frozen=True)
class AwgKey:
    device_uid: str
    awg_index: int


@dataclass
class AwgConfig:
    # QA
    raw_acquire_length: int = None
    result_length: int = None
    acquire_signals: Set[str] = field(default_factory=set)
    target_feedback_register: int = None
    # SG
    qa_signal_id: str = None
    command_table_match_offset: int = None
    source_feedback_register: int = None
    zsync_bit: int = None
    feedback_register_bit: int = None


AwgConfigs = Dict[AwgKey, AwgConfig]


@dataclass
class DeviceRecipeData:
    iq_settings: Dict[int, npt.ArrayLike] = field(default_factory=dict)


DeviceId = str
DeviceSettings = Dict[DeviceId, DeviceRecipeData]


@dataclass
class RtExecutionInfo:
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType

    # signal id -> set of section ids
    acquire_sections: Dict[str, Set[str]] = field(default_factory=dict)

    # signal -> flat list of result handles
    # TODO(2K): to be replaced by event-based calculation in the compiler
    signal_result_map: Dict[str, List[str]] = field(default_factory=dict)

    def add_acquire_section(self, signal_id: str, section_id: str):
        self.acquire_sections.setdefault(signal_id, set()).add(section_id)

    @staticmethod
    def get_acquisition_type(rt_execution_infos: RtExecutionInfos) -> AcquisitionType:
        # Currently only single RT execution per experiment supported
        rt_execution_info = next(iter(rt_execution_infos.values()), None)
        acquisition_type = (
            AcquisitionType.INTEGRATION
            if rt_execution_info is None
            else rt_execution_info.acquisition_type
        )
        return acquisition_type

    def signal_by_handle(self, handle: str) -> Optional[str]:
        return next(
            (
                signal
                for signal, handles in self.signal_result_map.items()
                if handle in handles
            ),
            None,
        )


RtSectionId = str
RtExecutionInfos = Dict[RtSectionId, RtExecutionInfo]


@dataclass
class RecipeData:
    compiled: CompiledExperiment
    recipe: Recipe.Data
    execution: Sequence
    result_shapes: HandleResultShapes
    rt_execution_infos: RtExecutionInfos
    device_settings: DeviceSettings
    awg_configs: AwgConfigs
    param_to_device_map: Dict[str, List[str]]

    @property
    def initializations(self) -> Iterator[Initialization.Data]:
        for initialization in self.recipe.experiment.initializations:
            yield initialization

    def get_initialization_by_device_uid(self, device_uid: str) -> Initialization.Data:
        initialization: Initialization.Data
        for initialization in self.initializations:
            if initialization.device_uid == device_uid:
                return initialization

    def awgs_producing_results(self) -> Iterator[Tuple[AwgKey, AwgConfig]]:
        for awg_key, awg_config in self.awg_configs.items():
            if awg_config.result_length is not None:
                yield awg_key, awg_config

    def awg_config_by_acquire_signal(self, signal_id: str) -> Optional[AwgConfig]:
        return next(
            (
                awg_config
                for awg_config in self.awg_configs.values()
                if signal_id in awg_config.acquire_signals
            ),
            None,
        )


def _pre_process_iq_settings_hdawg(initialization: Initialization.Data):
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
            q_out = next(
                (o for o in outputs if o.channel == output.channel + 1), IO.Data(0)
            )
        else:
            i_out = next(
                (o for o in outputs if o.channel == output.channel - 1), IO.Data(0)
            )
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
    axis_names: List[str] = field(default_factory=list)
    axis_points: List[npt.ArrayLike] = field(default_factory=list)

    @property
    def axis_name(self) -> Union[str, List[str]]:
        return self.axis_names[0] if len(self.axis_names) == 1 else self.axis_names

    @property
    def axis(self) -> Union[npt.ArrayLike, List[npt.ArrayLike]]:
        return self.axis_points[0] if len(self.axis_points) == 1 else self.axis_points


class _ResultShapeCalculator(ExecutorBase):
    def __init__(self):
        super().__init__(looping_mode=LoopingMode.ONCE)

        self.result_shapes: HandleResultShapes = {}
        self.rt_execution_infos: RtExecutionInfos = {}

        self._loop_stack: List[_LoopStackEntry] = []
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
        # if single_shot_sequential:
        #     shape.append(self.current_rt_info.averages)
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
            # if single_shot_sequential:
            #     axis_name.append(self.current_rt_uid)
            #     axis.append(self._single_shot_axis())
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

    def for_loop_handler(
        self, count: int, index: int, loop_type: LoopType, enter: bool
    ):
        if enter:
            is_averaging = loop_type in [LoopType.RT_AVERAGE, LoopType.AVERAGE]
            self._loop_stack.append(
                _LoopStackEntry(count=count, is_averaging=is_averaging)
            )
            if is_averaging:
                single_shot_cyclic = (
                    self._current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
                )
                # TODO(2K): single_shot_sequential
                if single_shot_cyclic:
                    self._loop_stack[-1].axis_names.append(self._current_rt_uid)
                    self._loop_stack[-1].axis_points.append(self._single_shot_axis())
        else:
            self._loop_stack.pop()

    def rt_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        enter: bool,
    ):
        if enter:
            if averaging_mode != AveragingMode.SINGLE_SHOT:
                max_hw_averages = (
                    pow(2, 15)
                    if acquisition_type == AcquisitionType.RAW
                    else pow(2, 17)
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
                ),
            )
        else:
            if self._current_rt_info is None:
                raise LabOneQControllerException(
                    "Nested 'acquire_loop_rt' are not allowed."
                )
            self._current_rt_uid = None
            self._current_rt_info = None


def _calculate_result_shapes(
    execution: Statement,
) -> Tuple[HandleResultShapes, RtExecutionInfos]:
    # Skip for recipe-only execution (used in older tests)
    if execution is None:
        return {}, {}
    rs_calc = _ResultShapeCalculator()
    rs_calc.run(execution)
    return rs_calc.result_shapes, rs_calc.rt_execution_infos


def _calculate_awg_configs(
    rt_execution_infos: RtExecutionInfos,
    experiment: RecipeExperiment.Data,
) -> AwgConfigs:
    awg_configs: AwgConfigs = defaultdict(AwgConfig)

    def awg_key_by_acquire_signal(signal_id: str) -> AwgKey:
        return next(
            awg_key
            for awg_key, awg_config in awg_configs.items()
            if signal_id in awg_config.acquire_signals
        )

    def integrator_index_by_acquire_signal(signal_id: str, is_local: bool) -> int:
        integrator = next(
            ia for ia in experiment.integrator_allocations if ia.signal_id == signal_id
        )
        # Only relevant for discrimination mode, where only one channel should
        # be assigned (no multi-state as of now)
        # TODO(2K): Check if HBAR-1359 affects also SHFQA / global feedback
        return integrator.channels[0] * (2 if is_local else 1)

    for a in experiment.integrator_allocations:
        awg_configs[AwgKey(a.device_id, a.awg)].acquire_signals.add(a.signal_id)

    for initialization in experiment.initializations:
        device_id = initialization.device_uid
        for awg in initialization.awgs or []:
            awg_config = awg_configs[AwgKey(device_id, awg.awg)]
            awg_config.qa_signal_id = awg.qa_signal_id
            awg_config.command_table_match_offset = awg.command_table_match_offset
            awg_config.target_feedback_register = awg.feedback_register

    zsync_bits_allocation: Dict[str, int] = defaultdict(int)
    for awg_key, awg_config in awg_configs.items():
        if awg_config.qa_signal_id is not None:
            qa_awg_key = awg_key_by_acquire_signal(awg_config.qa_signal_id)
            feedback_register = awg_configs[qa_awg_key].target_feedback_register
            is_local = feedback_register is None
            awg_config.feedback_register_bit = integrator_index_by_acquire_signal(
                awg_config.qa_signal_id, is_local
            )
            if not is_local:
                awg_config.source_feedback_register = feedback_register
                awg_config.zsync_bit = zsync_bits_allocation[awg_key.device_uid]
                zsync_bits_allocation[awg_key.device_uid] += 1

    # As currently just a single RT execution per experiment is supported,
    # AWG configs are not cloned per RT execution. May need to be changed in the future.
    for rt_execution_uid, rt_execution_info in rt_execution_infos.items():
        # Determine / check the raw acquisition lengths across various acquire events.
        # Must match for a single device.
        # device_id -> set of raw acquisition lengths
        raw_acquire_lengths: Dict[str, Set[int]] = {}
        for signal, sections in rt_execution_info.acquire_sections.items():
            awg_key = awg_key_by_acquire_signal(signal)
            for section in sections:
                for acquire_length_info in experiment.acquire_lengths:
                    if (
                        acquire_length_info.signal_id == signal
                        and acquire_length_info.section_id == section
                    ):
                        raw_acquire_lengths.setdefault(awg_key.device_uid, set()).add(
                            acquire_length_info.acquire_length
                        )
        for device_id, awg_raw_acquire_lengths in raw_acquire_lengths.items():
            if len(awg_raw_acquire_lengths) > 1:
                raise LabOneQControllerException(
                    f"Can't determine unique acquire length for the device '{device_id}' in "
                    f"acquire_loop_rt(uid='{rt_execution_uid}') section. Ensure all 'acquire' "
                    f"statements within this section mapping to this device use the same kernel "
                    f"length."
                )
        for awg_key, awg_config in awg_configs.items():
            # Use dummy raw_acquire_length 4096 if there's no acquire statements in experiment
            awg_config.raw_acquire_length = next(
                iter(raw_acquire_lengths.get(awg_key.device_uid, {4096}))
            )

            # signal_id -> sequence of handle/None for each result vector entry.
            # Important! Length must be equal for all acquire signals / integrators of one AWG.
            # All integrators occupy an entry in the respective result vectors per startQA event,
            # regardless of the given integrators mask. Masked-out integrators just leave the
            # value at NaN (corresponds to None in the map).
            awg_result_map: Dict[str, List[str]] = defaultdict(list)
            for acquires in experiment.simultaneous_acquires:
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


def _pre_process_params(
    experiment: RecipeExperiment.Data,
) -> Dict[str, Set[str]]:
    param_to_device_map: Dict[str, Set[str]] = defaultdict(set)

    oscillator_params = experiment.oscillator_params
    for oscillator_param in oscillator_params:
        param_to_device_map[oscillator_param.param].add(oscillator_param.device_id)

    for initialization in experiment.initializations:
        ppchannels = initialization.ppchannels or {}
        for settings in ppchannels.values():
            for key in ["pump_freq", "pump_power", "probe_frequency", "probe_power"]:
                if isinstance(settings[key], str):
                    param_to_device_map[settings[key]].add(initialization.device_uid)
    return param_to_device_map


def pre_process_compiled(compiled_experiment: CompiledExperiment) -> RecipeData:
    recipe: Recipe.Data = Recipe().load(compiled_experiment.recipe)

    device_settings: DeviceSettings = defaultdict(DeviceRecipeData)
    for initialization in recipe.experiment.initializations:
        device_settings[initialization.device_uid] = DeviceRecipeData(
            iq_settings=_pre_process_iq_settings_hdawg(initialization)
        )

    execution = ExecutionFactoryFromExperiment().make(compiled_experiment.experiment)
    result_shapes, rt_execution_infos = _calculate_result_shapes(execution)
    awg_configs = _calculate_awg_configs(rt_execution_infos, recipe.experiment)
    param_to_device_map = _pre_process_params(recipe.experiment)

    recipe_data = RecipeData(
        compiled=compiled_experiment,
        recipe=recipe,
        execution=execution,
        result_shapes=result_shapes,
        rt_execution_infos=rt_execution_infos,
        device_settings=device_settings,
        awg_configs=awg_configs,
        param_to_device_map=param_to_device_map,
    )

    return recipe_data


def get_wave(wave_name, waves: List[Dict[str, Any]]):
    wave = next(
        (wave for wave in waves if wave.get("filename", None) == wave_name), None
    )
    if wave is None:
        raise LabOneQControllerException(
            f"Wave '{wave_name}' is not found in the compiled waves collection."
        )
    return np.ascontiguousarray(wave["samples"])
