# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Set, Tuple, Union

import numpy as np
from numpy import typing as npt

from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.executor.execution_from_experiment import ExecutionFactoryFromExperiment
from laboneq.executor.executor import ExecutorBase, LoopingMode, Sequence, Statement

from .recipe_1_4_0 import IO, Experiment, Initialization, OscillatorParam, Recipe
from .recipe_enums import SignalType

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment


@dataclass
class DeviceRecipeData:
    iq_settings: Dict[int, npt.ArrayLike] = field(default_factory=dict)


@dataclass
class HandleResultShape:
    base_shape: List[int]
    base_axis_name: List[Union[str, List[str]]]
    base_axis: List[Union[npt.ArrayLike, List[npt.ArrayLike]]]
    additional_axis: int = 1


@dataclass(frozen=True)
class AwgKey:
    device_uid: str
    awg_index: int


@dataclass
class AwgConfig:
    acquire_length: int
    result_length: int
    signals: Set[str]


@dataclass
class RtExecutionInfo:
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType
    per_awg_configs: Dict[AwgKey, AwgConfig] = field(default_factory=dict)
    # signal -> flat list of result handles
    # TODO(2K): to be replaced by event-based calculation in the compiler
    signal_result_map: Dict[str, List[str]] = field(default_factory=dict)
    # Temporary mapping signal id -> set of section ids
    # used to determine / check the acquisition lengths across various acquire events
    _acquire_sections: Dict[str, Set[str]] = field(default_factory=dict)

    @staticmethod
    def get_acquisition_type(
        rt_execution_infos: Dict[str, RtExecutionInfo]
    ) -> AcquisitionType:
        # Currently only single RT execution per experiment supported
        rt_execution_info: RtExecutionInfo = next(
            iter(rt_execution_infos.values()), None
        )
        acquisition_type = (
            AcquisitionType.INTEGRATION
            if rt_execution_info is None
            else rt_execution_info.acquisition_type
        )
        return acquisition_type


@dataclass
class RecipeData:
    compiled: CompiledExperiment
    recipe: Recipe.Data
    execution: Sequence
    # key - acquire handle
    result_shapes: Dict[str, HandleResultShape]
    # key - RT section id
    rt_execution_infos: Dict[str, RtExecutionInfo]
    device_settings: Dict[str, DeviceRecipeData]
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

        # result handle -> list of dimensions
        self.result_shapes: Dict[str, HandleResultShape] = {}

        self.rt_execution_infos: Dict[str, RtExecutionInfo] = {}
        self.current_rt_uid: str = None
        self.current_rt_info: RtExecutionInfo = None

        self.is_averaging_loop: bool = False
        self.loop_stack: List[_LoopStackEntry] = []

    def _single_shot_axis(self) -> npt.ArrayLike:
        return np.linspace(
            0, self.current_rt_info.averages - 1, self.current_rt_info.averages
        )

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        single_shot_cyclic = (
            self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
        )
        signal_acquire_sections = self.current_rt_info._acquire_sections.setdefault(
            signal, set()
        )
        signal_acquire_sections.add(parent_uid)

        # Determine result shape for each acquire handle
        shape = [
            loop.count
            for loop in self.loop_stack
            if not loop.is_averaging or single_shot_cyclic
        ]
        # if single_shot_sequential:
        #     shape.append(self.current_rt_info.averages)
        known_shape = self.result_shapes.get(handle)
        if known_shape is None:
            axis_name = [
                loop.axis_name
                for loop in self.loop_stack
                if not loop.is_averaging or single_shot_cyclic
            ]
            axis = [
                loop.axis
                for loop in self.loop_stack
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
        self.loop_stack[-1].axis_names.append(name if axis_name is None else axis_name)
        self.loop_stack[-1].axis_points.append(values)

    def for_loop_handler(self, count: int, index: int, enter: bool):
        if enter:
            self.loop_stack.append(
                _LoopStackEntry(count=count, is_averaging=self.is_averaging_loop)
            )
            if self.is_averaging_loop:
                single_shot_cyclic = (
                    self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
                )
                # TODO(2K): single_shot_sequential
                if single_shot_cyclic:
                    self.loop_stack[-1].axis_names.append(self.current_rt_uid)
                    self.loop_stack[-1].axis_points.append(self._single_shot_axis())
        else:
            self.loop_stack.pop()
            if self.current_rt_info is not None:
                single_shot_cyclic = (
                    self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
                )
        if self.is_averaging_loop:
            self.is_averaging_loop = False

    def rt_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        enter: bool,
    ):
        if enter:
            self.is_averaging_loop = True
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

            self.current_rt_uid = uid
            self.current_rt_info = self.rt_execution_infos.setdefault(
                uid,
                RtExecutionInfo(
                    averages=count,
                    averaging_mode=averaging_mode,
                    acquisition_type=acquisition_type,
                ),
            )
        else:
            if self.current_rt_info is None:
                raise LabOneQControllerException(
                    "Nested 'acquire_loop_rt' are not allowed."
                )
            self.current_rt_uid = None
            self.current_rt_info = None


def _calculate_result_shapes(
    execution: Statement, experiment: Experiment.Data
) -> Tuple[Dict[str, HandleResultShape], Dict[str, RtExecutionInfo]]:
    # Skip for recipe-only execution (used in older tests)
    if execution is None:
        return {}, {}

    rs_calc = _ResultShapeCalculator()
    rs_calc.run(execution)

    awg_key_to_signals: Dict[AwgKey, Set[str]] = {}
    for a in experiment.integrator_allocations:
        awg_signals = awg_key_to_signals.setdefault(AwgKey(a.device_id, a.awg), set())
        awg_signals.add(a.signal_id)

    for rt_execution_uid, rt_execution_info in rs_calc.rt_execution_infos.items():
        acquire_lengths: Set[int] = set()
        for signal, sections in rt_execution_info._acquire_sections.items():
            for section in sections:
                for acquire_length_info in experiment.acquire_lengths:
                    if (
                        acquire_length_info.signal_id == signal
                        and acquire_length_info.section_id == section
                    ):
                        acquire_lengths.add(acquire_length_info.acquire_length)
        if len(acquire_lengths) > 1:
            raise LabOneQControllerException(
                f"Can't determine unique acquire length for the acquire_loop_rt(uid='{rt_execution_uid}') section. Ensure all 'acquire' statements within this section use the same kernel length."
            )
        # Use dummy acquire_length 4096 if there's no acquire statements in experiment
        acquire_length = acquire_lengths.pop() if len(acquire_lengths) > 0 else 4096

        for awg_key, signals in awg_key_to_signals.items():
            # signal -> list of handles (for one AWG with 'awg_key')
            awg_result_map: Dict[str, List[str]] = {}
            for i, acquires in enumerate(experiment.simultaneous_acquires):
                if any(signal in acquires for signal in signals):
                    for signal in signals:
                        signal_result_map = awg_result_map.setdefault(signal, [])
                        signal_result_map.append(acquires.get(signal))
            if len(awg_result_map) > 0:
                rt_execution_info.signal_result_map.update(awg_result_map)
                any_awg_signal_result_map = next(iter(awg_result_map.values()))
                mapping_repeats = (
                    rt_execution_info.averages
                    if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT
                    else 1
                )
                rt_execution_info.per_awg_configs[awg_key] = AwgConfig(
                    result_length=len(any_awg_signal_result_map) * mapping_repeats,
                    acquire_length=acquire_length,
                    signals=signals,
                )

    return rs_calc.result_shapes, rs_calc.rt_execution_infos


def _pre_process_oscillator_params(
    oscillator_params: List[OscillatorParam.Data],
) -> Dict[str, List[str]]:
    param_to_device_map: Dict[str, List[str]] = {}
    for oscillator_param in oscillator_params:
        param_bindings = param_to_device_map.setdefault(oscillator_param.param, [])
        param_bindings.append(oscillator_param.device_id)
    return param_to_device_map


def pre_process_compiled(compiled_experiment: CompiledExperiment) -> RecipeData:
    recipe: Recipe.Data = Recipe().load(compiled_experiment.recipe)

    device_settings: Dict[str, DeviceRecipeData] = {}
    for initialization in recipe.experiment.initializations:
        device_settings[initialization.device_uid] = DeviceRecipeData(
            iq_settings=_pre_process_iq_settings_hdawg(initialization)
        )

    execution = ExecutionFactoryFromExperiment().make(compiled_experiment.experiment)
    result_shapes, rt_execution_infos = _calculate_result_shapes(
        execution, recipe.experiment
    )
    param_to_device_map = _pre_process_oscillator_params(
        recipe.experiment.oscillator_params
    )

    recipe_data = RecipeData(
        compiled=compiled_experiment,
        recipe=recipe,
        execution=execution,
        result_shapes=result_shapes,
        rt_execution_infos=rt_execution_infos,
        device_settings=device_settings,
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
