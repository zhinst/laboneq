# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal, TypeVar, overload

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    AttributeValueTracker,
    DeviceAttribute,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IO, Initialization, OscillatorParam, Recipe, SignalType
from laboneq.data.scheduled_experiment import CodegenWaveform, ScheduledExperiment
from laboneq.executor.executor import (
    ExecutorBase,
    LoopFlags,
    LoopingMode,
    Sequence,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.devices.device_collection import DeviceCollection
    from laboneq.controller.devices.device_setup_dao import DeviceUID


@dataclass
class HandleResultShape:
    base_shape: list[int]
    base_axis_name: list[str | list[str]]
    base_axis: list[NumPyArray | list[NumPyArray]]
    # If a handle is used in multiple acquires, it adds an extra result dimension.
    # handle_acquire_count tracks the number of such acquires. This works only if
    # outer loops are the same for all acquires with the same handle.
    # base_shape is the result shape without this dimension. The extra dimension
    # is added with the handle as the axis name if the count is >1.
    handle_acquire_count: int = 1


AcquireHandle = str
HandleResultShapes = dict[AcquireHandle, HandleResultShape]


@dataclass(frozen=True)
class AwgKey:
    device_uid: DeviceUID
    awg_index: int | str


@dataclass
class AwgConfig:
    # TODO(2K): use enum
    device_type: str | None = None
    # QA
    raw_acquire_length: int | None = None
    signal_raw_acquire_lengths: dict[str, int] = field(default_factory=dict)
    result_length: int | None = None
    acquire_signals: set[str] = field(default_factory=set)
    target_feedback_register: int | None = None
    # SG
    command_table_match_offset: int | None = None
    source_feedback_register: int | Literal["local"] | None = None
    fb_reg_source_index: int | None = None
    fb_reg_target_index: int | None = None
    register_selector_bitmask: int | None = 0b11
    register_selector_shift: int | None = None


AwgConfigs = dict[AwgKey, AwgConfig]


@dataclass
class DeviceRecipeData:
    iq_settings: dict[int, tuple[int, int]] = field(default_factory=dict)
    osc_params: list[OscillatorParam] = field(default_factory=list)


@dataclass
class RtExecutionInfo:
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType
    pipeliner_job_count: int | None
    pipeliner_repetitions: int | None

    # acquire signal ids
    acquire_signals: set[str] = field(default_factory=set)

    # signal -> flat list of result handles
    # TODO(2K): to be replaced by event-based calculation in the compiler
    signal_result_map: dict[str, list[str | None]] = field(default_factory=dict)

    @property
    def with_pipeliner(self) -> bool:
        return self.pipeliner_job_count is not None

    @property
    def pipeliner_jobs(self) -> int:
        return self.pipeliner_job_count or 1

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

T = TypeVar("T")


def get_artifacts(artifacts: Any, device_class: int, artifacts_class: type[T]) -> T:
    if isinstance(artifacts, artifacts_class):
        return artifacts
    if isinstance(artifacts, dict):
        dev_class_artifacts = artifacts[device_class]
        if isinstance(dev_class_artifacts, artifacts_class):
            return dev_class_artifacts
    raise LabOneQControllerException(
        "Internal error: Unexpected artifacts structure in compiled experiment."
    )


def get_initialization_by_device_uid(
    recipe: Recipe, device_uid: str
) -> Initialization | None:
    for initialization in recipe.initializations:
        if initialization.device_uid == device_uid:
            return initialization
    return None


@dataclass
class RecipeData:
    scheduled_experiment: ScheduledExperiment
    recipe: Recipe
    execution: Sequence
    result_shapes: HandleResultShapes
    rt_execution_infos: RtExecutionInfos
    device_settings: dict[DeviceUID, DeviceRecipeData]
    awg_configs: AwgConfigs
    attribute_value_tracker: AttributeValueTracker
    oscillator_ids: list[str]

    def get_initialization(self, device_uid: DeviceUID) -> Initialization:
        for initialization in self.recipe.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        return Initialization(device_uid=device_uid)

    def get_artifacts(self, device_class: int, artifacts_class: type[T]) -> T:
        return get_artifacts(
            self.scheduled_experiment.artifacts, device_class, artifacts_class
        )

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


def _pre_process_iq_settings_hdawg(
    initialization: Initialization | None,
) -> dict[int, tuple[int, int]]:
    if initialization is None or initialization.device_type != "HDAWG":
        return {}

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
        iq_settings[awg_idx] = (i_out.channel, q_out.channel)

    return iq_settings


@dataclass
class _LoopStackEntry:
    count: int
    is_averaging: bool
    axis_names: list[str] = field(default_factory=list)
    axis_points: list[NumPyArray] = field(default_factory=list)

    @property
    def axis_name(self) -> str | list[str]:
        return self.axis_names[0] if len(self.axis_names) == 1 else self.axis_names

    @property
    def axis(self) -> NumPyArray | list[NumPyArray]:
        return self.axis_points[0] if len(self.axis_points) == 1 else self.axis_points


class _LoopsPreprocessor(ExecutorBase):
    def __init__(self):
        super().__init__(looping_mode=LoopingMode.ONCE)

        self.result_shapes: HandleResultShapes = {}
        self.rt_execution_infos: RtExecutionInfos = {}
        self.pipeliner_job_count: int | None = None
        self.pipeliner_repetitions: int | None = None
        self._loop_stack: list[_LoopStackEntry] = []
        self._current_rt_uid: str | None = None
        self._current_rt_info: RtExecutionInfo | None = None

    @property
    def current_rt_uid(self) -> str:
        assert self._current_rt_uid is not None
        return self._current_rt_uid

    @property
    def current_rt_info(self) -> RtExecutionInfo:
        assert self._current_rt_info is not None
        return self._current_rt_info

    def _single_shot_axis(self) -> NumPyArray:
        return np.linspace(
            0, self.current_rt_info.averages - 1, self.current_rt_info.averages
        )

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        self.current_rt_info.acquire_signals.add(signal)

        # Determine result shape for each acquire handle
        single_shot_cyclic = (
            self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
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
            known_shape.handle_acquire_count += 1
        else:
            raise LabOneQControllerException(
                f"Multiple acquire events with the same handle ('{handle}') and different result shapes are not allowed."
            )

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: NumPyArray
    ):
        self._loop_stack[-1].axis_names.append(name if axis_name is None else axis_name)
        self._loop_stack[-1].axis_points.append(values)

    def for_loop_entry_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if loop_flags.is_pipeline:
            self.pipeliner_job_count = count
            self.pipeliner_repetitions = math.prod(
                len(l.axis_points) for l in self._loop_stack
            )
            return

        self._loop_stack.append(
            _LoopStackEntry(count=count, is_averaging=loop_flags.is_average)
        )
        if loop_flags.is_average:
            single_shot_cyclic = (
                self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
            )
            if single_shot_cyclic:
                self._loop_stack[-1].axis_names.append(self.current_rt_uid)
                self._loop_stack[-1].axis_points.append(self._single_shot_axis())

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        if loop_flags.is_pipeline:
            self.pipeliner_job_count = None
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
        self._current_rt_uid = uid
        self._current_rt_info = self.rt_execution_infos.setdefault(
            uid,
            RtExecutionInfo(
                averages=count,
                averaging_mode=averaging_mode,
                acquisition_type=acquisition_type,
                pipeliner_job_count=self.pipeliner_job_count,
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
    rt_execution_infos: RtExecutionInfos,
    recipe: Recipe,
    devices: DeviceCollection,
) -> AwgConfigs:
    awg_configs: AwgConfigs = defaultdict(AwgConfig)

    def awg_key_by_acquire_signal(signal_id: str) -> AwgKey | None:
        return next(
            (
                awg_key
                for awg_key, awg_config in awg_configs.items()
                if signal_id in awg_config.acquire_signals
            ),
            None,
        )

    for integrator_allocation in recipe.integrator_allocations:
        awg_key = AwgKey(integrator_allocation.device_id, integrator_allocation.awg)
        awg_configs[awg_key].acquire_signals.add(integrator_allocation.signal_id)

    for initialization in recipe.initializations:
        for awg in initialization.awgs or []:
            awg_config = awg_configs[AwgKey(initialization.device_uid, awg.awg)]
            awg_config.device_type = initialization.device_type
            awg_config.target_feedback_register = awg.target_feedback_register
            awg_config.source_feedback_register = awg.source_feedback_register
            if awg_config.source_feedback_register not in (None, "local"):
                if devices.has_qhub:
                    raise LabOneQControllerException(
                        "Global feedback over QHub is not implemented."
                    )
                awg_config.fb_reg_source_index = awg.feedback_register_index_select
                assert isinstance(awg.awg, int)
                awg_config.fb_reg_target_index = awg.awg
            awg_config.register_selector_shift = awg.codeword_bitshift
            awg_config.register_selector_bitmask = awg.codeword_bitmask
            awg_config.command_table_match_offset = awg.command_table_match_offset
            if initialization.device_type == "PRETTYPRINTERDEVICE":
                awg_config.result_length = -1  # result shape determined by device

    acquire_lengths = {al.signal_id: al.acquire_length for al in recipe.acquire_lengths}
    # As currently just a single RT execution per experiment is supported,
    # AWG configs are not cloned per RT execution. May need to be changed in the future.
    for rt_execution_info in rt_execution_infos.values():
        # Determine the raw acquisition lengths across various acquire events.
        # Will use the maximum length, as scope / monitor can only be configured for one.
        # device_uid + awg_index -> raw acquisition lengths
        raw_acquire_lengths: dict[str, dict[str, int]] = defaultdict(dict)
        raw_acquire_channels: dict[str, set[int]] = defaultdict(set)
        if rt_execution_info.acquisition_type == AcquisitionType.RAW:
            for signal in rt_execution_info.acquire_signals:
                ac_awg_key = awg_key_by_acquire_signal(signal)
                if ac_awg_key is None:
                    continue
                raw_acquire_lengths[ac_awg_key.device_uid][signal] = (
                    acquire_lengths.get(signal, 0)
                )
                assert isinstance(ac_awg_key.awg_index, int)
                raw_acquire_channels[ac_awg_key.device_uid].add(ac_awg_key.awg_index)
        for awg_key, awg_config in awg_configs.items():
            dev_raw_acquire_lengths = raw_acquire_lengths.get(awg_key.device_uid, {})
            # Use dummy raw_acquire_length 4096 if there's no acquire statements in experiment
            raw_acquire_length = max(dev_raw_acquire_lengths.values(), default=4096)
            awg_config.raw_acquire_length = raw_acquire_length
            awg_config.signal_raw_acquire_lengths = dev_raw_acquire_lengths

            # signal_id -> sequence of handle/None for each result vector entry.
            # Important! Length must be equal for all acquire signals / integrators of one AWG.
            # All integrators occupy an entry in the respective result vectors per startQA event,
            # regardless of the given integrators mask. Masked-out integrators just leave the
            # value at NaN (corresponds to None in the map).
            awg_result_map: dict[str, list[str | None]] = defaultdict(list)
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
                result_length = len(any_awg_signal_result_map) * mapping_repeats
                is_raw_acquisition = awg_key.device_uid in raw_acquire_lengths
                if is_raw_acquisition:
                    if (
                        rt_execution_info.averaging_mode == AveragingMode.SEQUENTIAL
                        and rt_execution_info.averages > 1
                    ):
                        raise LabOneQControllerException(
                            "Sequential averaging is not supported for raw acquisitions."
                        )
                    device_type = awg_config.device_type
                    if device_type is None:
                        device_type = ""
                    if device_type.startswith("SHFQA"):  # SHFQA or SHFQC/QA
                        if devices.find_by_uid(awg_key.device_uid).options.is_qc:
                            SCOPE_MEMORY_SIZE = 64 * 1024
                        else:
                            SCOPE_MEMORY_SIZE = 256 * 1024
                        enabled_channels = raw_acquire_channels.get(
                            awg_key.device_uid, {0}
                        )
                        if len(enabled_channels) < 2:
                            ch_split = 1
                        elif len(enabled_channels) == 2:
                            ch_split = 2
                        else:
                            ch_split = 4
                        max_length = (
                            SCOPE_MEMORY_SIZE // ch_split // result_length
                        ) & ~0xF
                        max_segments = 1024
                    else:
                        max_length = 4096
                        max_segments = 1
                    if result_length > max_segments:
                        raise LabOneQControllerException(
                            f"A maximum of {max_segments} raw result(s) is supported per real-time execution."
                        )
                    if raw_acquire_length > max_length:
                        raise LabOneQControllerException(
                            "The total size of the requested raw traces exceeds the instrument's memory capacity."
                        )

                awg_config.result_length = result_length

    return awg_configs


def _pre_process_attributes(
    recipe: Recipe, devices: DeviceCollection
) -> tuple[AttributeValueTracker, list[str], dict[str, list[OscillatorParam]]]:
    attribute_value_tracker = AttributeValueTracker()
    oscillator_ids: list[str] = []
    oscillators_check: dict[str, str | float] = {}

    osc_params_per_device = defaultdict(list)
    # Sort oscillator params by id to match the order in the compiler
    for oscillator_param in sorted(recipe.oscillator_params, key=lambda p: p.id):
        osc_params_per_device[oscillator_param.device_id].append(oscillator_param)
        value_or_param = oscillator_param.param or oscillator_param.frequency
        assert value_or_param is not None, "undefined oscillator frequency"
        if oscillator_param.id in oscillator_ids:
            osc_param_index = oscillator_ids.index(oscillator_param.id)
            if oscillators_check[oscillator_param.id] != value_or_param:
                raise LabOneQControllerException(
                    f"Conflicting specifications for the same oscillator id '{oscillator_param.id}' "
                    f"in the recipe: '{oscillators_check[oscillator_param.id]}' != '{value_or_param}'"
                )
        else:
            osc_param_index = len(oscillator_ids)
            oscillator_ids.append(oscillator_param.id)
            oscillators_check[oscillator_param.id] = value_or_param
        attribute_value_tracker.add_attribute(
            device_uid=oscillator_param.device_id,
            attribute=DeviceAttribute(
                name=AttributeName.OSCILLATOR_FREQ,
                index=osc_param_index,
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

    return attribute_value_tracker, oscillator_ids, osc_params_per_device


def pre_process_compiled(
    scheduled_experiment: ScheduledExperiment,
    devices: DeviceCollection,
) -> RecipeData:
    for device_uid, device in devices.all:
        device.validate_scheduled_experiment(device_uid, scheduled_experiment)

    recipe = scheduled_experiment.recipe
    assert recipe is not None
    execution = scheduled_experiment.execution

    attribute_value_tracker, oscillator_ids, osc_params_per_device = (
        _pre_process_attributes(recipe, devices)
    )

    device_settings: dict[DeviceUID, DeviceRecipeData] = {
        device_uid: DeviceRecipeData(
            iq_settings=_pre_process_iq_settings_hdawg(
                get_initialization_by_device_uid(recipe, device_uid)
            ),
            osc_params=osc_params_per_device.get(device_uid, []),
        )
        for device_uid, _ in devices.all
    }

    lp = _LoopsPreprocessor()
    lp.run(execution)
    rt_execution_infos = lp.rt_execution_infos

    awg_configs = _calculate_awg_configs(rt_execution_infos, recipe, devices)

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


@overload
def get_wave(wave_name, waves: dict[str, CodegenWaveform]) -> CodegenWaveform: ...


@overload
def get_wave(
    wave_name, waves: dict[str, CodegenWaveform], optional: bool = True
) -> CodegenWaveform | None: ...


def get_wave(
    wave_name, waves: dict[str, CodegenWaveform], optional: bool = False
) -> CodegenWaveform | None:
    wave = waves.get(wave_name)
    if wave is None:
        if optional:
            return None
        raise LabOneQControllerException(
            f"Wave '{wave_name}' is not found in the compiled waves collection."
        )
    return wave
