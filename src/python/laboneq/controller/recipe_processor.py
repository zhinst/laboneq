# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import ItemsView, Iterator
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from packaging.version import Version, InvalidVersion
from typing import TYPE_CHECKING, Any, KeysView, Literal, TypeVar, cast, overload

import numpy as np

import zhinst.utils  # type: ignore[import-untyped]

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    AttributeValueTracker,
    DeviceAttribute,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IO, Initialization, OscillatorParam, Recipe, SignalType
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    CodegenWaveform,
    ScheduledExperiment,
)
from laboneq.executor.executor import (
    ExecutorBase,
    LoopFlags,
    LoopingMode,
    Statement,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.devices.device_collection import DeviceCollection
    from laboneq.controller.devices.device_setup_dao import DeviceUID


_logger = logging.getLogger(__name__)


MIN_LABONEQ_VERSION_FOR_COMPILED_EXPERIMENT = Version("2.52.0")


@dataclass
class HandleResultShape:
    signal: str
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
    device_type: str | None
    signals: set[str]
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


class AwgConfigs:
    def __init__(self):
        self._configs: dict[AwgKey, AwgConfig] = {}

    def __getitem__(self, key: AwgKey) -> AwgConfig:
        return self._configs[key]

    def items(self) -> ItemsView[AwgKey, AwgConfig]:
        return self._configs.items()

    def keys(self) -> KeysView[AwgKey]:
        return self._configs.keys()

    def add(self, key: AwgKey, config: AwgConfig):
        assert key not in self._configs
        self._configs[key] = config

    def by_signal(self, signal_id: str) -> tuple[AwgKey, AwgConfig]:
        return next(
            (key, config)
            for key, config in self._configs.items()
            if signal_id in config.signals
        )


@dataclass
class DeviceRecipeData:
    iq_settings: dict[int, tuple[int, int]] = field(default_factory=dict)
    osc_params: list[OscillatorParam] = field(default_factory=list)


@dataclass
class RtExecutionInfo:
    uid: str
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType
    pipeliner_job_count: int | None

    # signal -> flat list of result handles
    # TODO(2K): to be replaced by event-based calculation in the compiler
    signal_result_map: dict[str, list[str | None]] = field(default_factory=dict)

    @property
    def with_pipeliner(self) -> bool:
        return self.pipeliner_job_count is not None

    @property
    def pipeliner_jobs(self) -> int:
        return self.pipeliner_job_count or 1

    @property
    def is_raw_acquisition(self) -> bool:
        return self.acquisition_type == AcquisitionType.RAW

    @property
    def effective_averages(self) -> int:
        return 1 if self.averaging_mode == AveragingMode.SINGLE_SHOT else self.averages

    @property
    def mapping_repeats(self) -> int:
        return self.averages if self.averaging_mode == AveragingMode.SINGLE_SHOT else 1

    @property
    def effective_averaging_mode(self) -> AveragingMode:
        # TODO(2K): handle sequential
        return (
            AveragingMode.CYCLIC
            if self.averaging_mode == AveragingMode.SINGLE_SHOT
            else self.averaging_mode
        )


T = TypeVar("T")


def get_artifacts(artifacts: Any, artifacts_class: type[T]) -> T:
    if isinstance(artifacts, artifacts_class):
        return artifacts
    if isinstance(artifacts, dict):
        for artifacts_data in artifacts.values():
            if isinstance(artifacts_data, artifacts_class):
                return artifacts_data
    raise LabOneQControllerException(
        "Internal error: Unexpected artifacts structure in compiled experiment."
    )


def get_initialization_by_device_uid(
    recipe: Recipe | None, device_uid: str
) -> Initialization | None:
    if recipe is None:
        return None
    for initialization in recipe.initializations:
        if initialization.device_uid == device_uid:
            return initialization
    return None


def get_elf(artifacts: ArtifactsCodegen, seqc_ref: str | None) -> bytes | None:
    if seqc_ref is None:
        return None

    if artifacts.src is None:
        seqc = None
    else:
        seqc = next((s for s in artifacts.src if s["filename"] == seqc_ref), None)
    if seqc is None or "elf" not in seqc or not isinstance(seqc["elf"], bytes):
        raise LabOneQControllerException(
            f"SeqC program '{seqc_ref}' not found or invalid"
        )

    return seqc["elf"]


@dataclass
class RecipeData:
    scheduled_experiment: ScheduledExperiment
    recipe: Recipe
    execution: Statement
    result_shapes: HandleResultShapes
    rt_execution_info: RtExecutionInfo
    device_settings: dict[DeviceUID, DeviceRecipeData]
    awg_configs: AwgConfigs
    attribute_value_tracker: AttributeValueTracker
    oscillator_ids: list[str]

    def get_initialization(self, device_uid: DeviceUID) -> Initialization:
        for initialization in self.recipe.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        return Initialization(device_uid=device_uid)

    def get_artifacts(self, artifacts_class: type[T]) -> T:
        return get_artifacts(self.scheduled_experiment.artifacts, artifacts_class)

    def awgs_producing_results(self) -> Iterator[tuple[AwgKey, AwgConfig]]:
        for awg_key, awg_config in self.awg_configs.items():
            if awg_config.result_length is not None:
                yield awg_key, awg_config


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
        self._rt_execution_info: RtExecutionInfo | None = None
        self.pipeliner_job_count: int | None = None
        self._loop_stack: list[_LoopStackEntry] = []
        self._current_rt_info: RtExecutionInfo | None = None

    @property
    def rt_execution_info(self) -> RtExecutionInfo:
        if self._rt_execution_info is None:
            raise LabOneQControllerException(
                "No 'acquire_loop_rt' section found in the experiment."
            )
        return self._rt_execution_info

    @property
    def current_rt_info(self) -> RtExecutionInfo:
        assert self._current_rt_info is not None
        return self._current_rt_info

    def _single_shot_axis(self) -> NumPyArray:
        return np.linspace(
            0, self.current_rt_info.averages - 1, self.current_rt_info.averages
        )

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
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
                signal=signal,
                base_shape=shape,
                base_axis_name=axis_name,
                base_axis=axis,
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
        self._loop_stack.append(
            _LoopStackEntry(count=count, is_averaging=loop_flags.is_average)
        )
        if loop_flags.is_average:
            single_shot_cyclic = (
                self.current_rt_info.averaging_mode == AveragingMode.SINGLE_SHOT
            )
            if single_shot_cyclic:
                self._loop_stack[-1].axis_names.append(self.current_rt_info.uid)
                self._loop_stack[-1].axis_points.append(self._single_shot_axis())

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        self._loop_stack.pop()

    def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        if self._rt_execution_info is not None and self._rt_execution_info.uid != uid:
            raise LabOneQControllerException(
                "Multiple 'acquire_loop_rt' sections per experiment is not supported."
            )
        self._current_rt_info = RtExecutionInfo(
            uid=uid,
            averages=count,
            averaging_mode=averaging_mode,
            acquisition_type=acquisition_type,
            pipeliner_job_count=self.pipeliner_job_count,
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
        self._rt_execution_info = self._current_rt_info
        self._current_rt_info = None


def _calculate_awg_configs(
    rt_execution_info: RtExecutionInfo,
    recipe: Recipe,
    devices: DeviceCollection,
) -> AwgConfigs:
    awg_configs = AwgConfigs()

    for initialization in recipe.initializations:
        for awg in initialization.awgs or []:
            awg_config = AwgConfig(
                device_type=initialization.device_type,
                signals=set(awg.signals.keys()),
                target_feedback_register=awg.target_feedback_register,
                source_feedback_register=awg.source_feedback_register,
                register_selector_shift=awg.codeword_bitshift,
                register_selector_bitmask=awg.codeword_bitmask,
                command_table_match_offset=awg.command_table_match_offset,
            )
            if awg_config.source_feedback_register not in (None, "local"):
                if devices.has_qhub:
                    raise LabOneQControllerException(
                        "Global feedback over QHub is not implemented."
                    )
                awg_config.fb_reg_source_index = awg.feedback_register_index_select
                assert isinstance(awg.awg, int)
                awg_config.fb_reg_target_index = awg.awg
            if initialization.device_type == "PRETTYPRINTERDEVICE":
                awg_config.result_length = -1  # result shape determined by device
            awg_configs.add(AwgKey(initialization.device_uid, awg.awg), awg_config)

    for integrator_allocation in recipe.integrator_allocations:
        awg_key = AwgKey(integrator_allocation.device_id, integrator_allocation.awg)
        awg_configs[awg_key].acquire_signals.add(integrator_allocation.signal_id)

    acquire_lengths = {al.signal_id: al.acquire_length for al in recipe.acquire_lengths}
    acquire_signals = set(acquire_lengths.keys())

    # Determine the raw acquisition lengths across various acquire events.
    # Will use the maximum length, as scope / monitor can only be configured for one.
    # device_uid + awg_index -> raw acquisition lengths
    raw_acquire_lengths: dict[str, dict[str, int]] = defaultdict(dict)
    raw_acquire_channels: dict[str, set[int]] = defaultdict(set)
    if rt_execution_info.acquisition_type == AcquisitionType.RAW:
        for signal in acquire_signals:
            ac_awg_key, _ = awg_configs.by_signal(signal)
            raw_acquire_lengths[ac_awg_key.device_uid][signal] = acquire_lengths.get(
                signal, 0
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
            result_length = (
                len(any_awg_signal_result_map) * rt_execution_info.mapping_repeats
            )
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
                    enabled_channels = raw_acquire_channels.get(awg_key.device_uid, {0})
                    if len(enabled_channels) < 2:
                        ch_split = 1
                    elif len(enabled_channels) == 2:
                        ch_split = 2
                    else:
                        ch_split = 4
                    max_length = (SCOPE_MEMORY_SIZE // ch_split // result_length) & ~0xF
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
    for _, device in devices.all:
        device.validate_scheduled_experiment(scheduled_experiment)

    recipe = scheduled_experiment.recipe
    assert recipe is not None
    execution = scheduled_experiment.execution
    assert execution is not None

    try:
        if (
            Version(Version(recipe.versions.laboneq).base_version)
            < MIN_LABONEQ_VERSION_FOR_COMPILED_EXPERIMENT
        ):
            raise LabOneQControllerException(
                f"The experiment was compiled with LabOne Q version {recipe.versions.laboneq}, which is not compatible. "
                "Please recompile using the current LabOne Q version."
            )
    except InvalidVersion:
        _logger.warning(
            "The experiment was compiled with an invalid LabOne Q version '%s'. "
            "Passing, as this is only expected to happen in tests. Consider ensuring that the valid version is set.",
            recipe.versions.laboneq,
        )

    if (
        scheduled_experiment.device_setup_fingerprint
        != devices.device_setup_fingerprint
    ):
        raise LabOneQControllerException(
            "The device setup of the compiled experiment is not compatible with the current device setup. "
            "Please recompile using the current device setup."
        )

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
    lp.pipeliner_job_count = scheduled_experiment.chunk_count
    lp.run(execution)
    rt_execution_info = lp.rt_execution_info

    awg_configs = _calculate_awg_configs(rt_execution_info, recipe, devices)

    recipe_data = RecipeData(
        scheduled_experiment=scheduled_experiment,
        recipe=recipe,
        execution=execution,
        result_shapes=lp.result_shapes,
        rt_execution_info=rt_execution_info,
        device_settings=device_settings,
        awg_configs=awg_configs,
        attribute_value_tracker=attribute_value_tracker,
        oscillator_ids=oscillator_ids,
    )

    for _, device in devices.all:
        device.validate_recipe_data(recipe_data)

    return recipe_data


@overload
def get_wave(wave_name: str, waves: dict[str, CodegenWaveform]) -> CodegenWaveform: ...


@overload
def get_wave(
    wave_name: str, waves: dict[str, CodegenWaveform], optional: bool = True
) -> CodegenWaveform | None: ...


def get_wave(
    wave_name: str, waves: dict[str, CodegenWaveform], optional: bool = False
) -> CodegenWaveform | None:
    wave = waves.get(wave_name)
    if wave is None:
        if optional:
            return None
        raise LabOneQControllerException(
            f"Wave '{wave_name}' is not found in the compiled waves collection."
        )
    return wave


def get_marker_samples(
    wave_name: str, waves: dict[str, CodegenWaveform]
) -> NumPyArray | None:
    marker_wave = get_wave(wave_name, waves, optional=True)
    if marker_wave is None:
        return None
    samples = marker_wave.samples
    if samples is not None and np.any(samples * (1 - samples)):
        raise LabOneQControllerException(
            "Marker samples must only contain ones and zeros"
        )
    return np.array(samples, order="C")


def get_iq_marker_samples(
    sig: str, waves: dict[str, CodegenWaveform]
) -> NumPyArray | None:
    marker_samples = get_marker_samples(f"{sig}_marker1.wave", waves)
    marker2_samples = get_marker_samples(f"{sig}_marker2.wave", waves)
    if marker2_samples is not None:
        marker2_len = len(marker2_samples)
        if marker_samples is None:
            marker_samples = np.zeros(marker2_len, dtype=np.int32)
        elif len(marker_samples) != marker2_len:
            raise LabOneQControllerException(
                "Samples for marker1 and marker2 must have the same length"
            )
        # we want marker 2 to be played on output 2, marker 1
        # bits 0/1 = marker 1/2 of output 1, bit 2/4 = marker 1/2 output 2
        # bit 2 is factor 4
        factor = 4
        marker_samples += factor * marker2_samples
    return marker_samples


@dataclass
class WaveformItem:
    index: int
    name: str
    samples: NumPyArray
    hold_start: int | None = None
    hold_length: int | None = None


Waveforms = list[WaveformItem]


def _prepare_wave_iq(
    waves: dict[str, CodegenWaveform], sig: str, index: int
) -> WaveformItem:
    wave_i = get_wave(f"{sig}_i.wave", waves)
    wave_q = get_wave(f"{sig}_q.wave", waves)
    marker_samples = get_iq_marker_samples(sig, waves)
    return WaveformItem(
        index=index,
        name=sig,
        samples=zhinst.utils.convert_awg_waveform(
            np.clip(np.ascontiguousarray(wave_i.samples), -1, 1),
            np.clip(np.ascontiguousarray(wave_q.samples), -1, 1),
            markers=marker_samples,
        ),
    )


def _prepare_wave_single(
    waves: dict[str, CodegenWaveform], sig: str, index: int
) -> WaveformItem:
    wave = get_wave(f"{sig}.wave", waves)
    marker_samples = get_marker_samples(f"{sig}_marker1.wave", waves)
    return WaveformItem(
        index=index,
        name=sig,
        samples=zhinst.utils.convert_awg_waveform(
            np.clip(np.ascontiguousarray(wave.samples), -1, 1),
            markers=marker_samples,
        ),
    )


def _prepare_wave_complex(
    waves: dict[str, CodegenWaveform], sig: str, index: int
) -> WaveformItem:
    wave = get_wave(f"{sig}.wave", waves)
    return WaveformItem(
        index=index,
        name=sig,
        samples=np.ascontiguousarray(wave.samples, dtype=np.complex128),
        hold_start=wave.hold_start,
        hold_length=wave.hold_length,
    )


def prepare_waves(
    artifacts: ArtifactsCodegen,
    wave_indices_ref: str | None,
) -> Waveforms | None:
    if wave_indices_ref is None:
        return None
    if artifacts.wave_indices is None:
        return None
    wave_indices: dict[str, list[int | str]] = next(
        (i for i in artifacts.wave_indices if i["filename"] == wave_indices_ref),
        {"value": {}},
    )["value"]

    waves: Waveforms = []
    for sig, [_index, _sig_type] in wave_indices.items():
        index = cast(int, _index)
        sig_type = cast(str, _sig_type)
        if sig.startswith("precomp_reset"):
            continue  # precomp reset waveform is bundled with ELF
        if sig_type in ("iq", "double", "multi"):
            wave = _prepare_wave_iq(artifacts.waves, sig, index)
        elif sig_type == "single":
            wave = _prepare_wave_single(artifacts.waves, sig, index)
        elif sig_type == "complex":
            wave = _prepare_wave_complex(artifacts.waves, sig, index)
        else:
            raise LabOneQControllerException(
                f"Unexpected signal type for binary wave for '{sig}' in '{wave_indices_ref}' - "
                f"'{sig_type}', should be one of [iq, double, multi, single, complex]"
            )
        waves.append(wave)

    return waves


def prepare_command_table(
    artifacts: ArtifactsCodegen, ct_ref: str | None
) -> dict | None:
    if ct_ref is None:
        return None
    return next(
        (ct["ct"] for ct in artifacts.command_tables if ct["seqc"] == ct_ref),
        None,
    )
