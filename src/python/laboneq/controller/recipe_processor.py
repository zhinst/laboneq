# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    KeysView,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from packaging.version import InvalidVersion, Version
from sortedcollections import NearestDict

from laboneq._utils import cached_method
from laboneq.controller.attribute_value_tracker import AttributeValueTracker
from laboneq.controller.constants import (
    PIPELINER_RELOAD_WORST_CASE,
    RESULT_TRANSFER_RATE_GW_FW,
    adjusted_transfer_rate,
)
from laboneq.controller.utilities.awg_waveform import convert_awg_waveform
from laboneq.controller.utilities.exception import (
    LabOneQControllerException,
    UnknownCallbackError,
)
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.enums.wave_type import WaveType
from laboneq.data.artifacts_qccs import Initialization
from laboneq.executor.executor import (
    ExecutorBase,
    LoopingMode,
)

if TYPE_CHECKING:
    from collections.abc import Callable, ItemsView, Iterator

    from laboneq.controller.devices.device_setup_dao import DeviceUID
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.data.artifacts_qccs import (
        ArtifactsCodegen,
        CodegenWaveform,
        PPChannel,
        WeightInfo,
    )
    from laboneq.data.scheduled_experiment import (
        CompilerArtifact,
        ScheduledExperiment,
    )
    from laboneq.executor.executor import (
        Statement,
    )


_logger = logging.getLogger(__name__)


MIN_LABONEQ_VERSION_FOR_COMPILED_EXPERIMENT = Version("2.52.0")


@dataclass(frozen=True)
class AwgKey:
    device_uid: DeviceUID
    awg_index: int


class AwgType(Enum):
    QA = auto()
    SG = auto()


class AwgSignalType(Enum):
    IQ = auto()
    SINGLE = auto()


@dataclass
class AwgConfig:
    signal_type: AwgSignalType
    signals: set[str]
    # QA
    raw_acquire_length: int | None = None
    result_length: int | None = None
    acquire_signals: set[str] = field(default_factory=set)
    # SG
    command_table_match_offset: int | None = None
    source_feedback_register: int | Literal["local"] | None = None
    fb_reg_source_index: int | None = None
    fb_reg_target_index: int | None = None


class AwgConfigs:
    def __init__(self):
        self._configs: dict[AwgKey, AwgConfig] = {}
        # Used for timeout estimation to account for possible slowdowns in the
        # result delivery path. Use the worst-case (maximum) result length, since
        # faster AWGs will still be blocked by slower ones due to HW synchronization.
        self._max_result_length: int = 0

    @property
    def max_result_length(self) -> int:
        return self._max_result_length

    def __getitem__(self, key: AwgKey) -> AwgConfig:
        return self._configs[key]

    def items(self) -> ItemsView[AwgKey, AwgConfig]:
        return self._configs.items()

    def keys(self) -> KeysView[AwgKey]:
        return self._configs.keys()

    def add(self, key: AwgKey, config: AwgConfig):
        assert key not in self._configs
        if (
            config.result_length is not None
            and config.result_length > self._max_result_length
        ):
            self._max_result_length = config.result_length
        self._configs[key] = config

    def by_signal(self, signal_id: str) -> tuple[AwgKey, AwgConfig]:
        return next(
            (key, config)
            for key, config in self._configs.items()
            if signal_id in config.signals
        )


class HWModulation(Enum):
    NONE = auto()
    OFF = auto()
    SINE = auto()
    IQ = auto()


class TrigOutSource(Enum):
    NONE = auto()
    AWG_TRIGGER = auto()
    MARKER_1 = auto()


@dataclass
class HDAWGOutputRecipeData:
    enable: bool | None = None
    range: float | None = None
    hw_modulation: HWModulation = HWModulation.NONE
    # To keep behavior identical to before, empty dict indicates no precompensation provided,
    # while None means precompensation explicitly disabled.
    precompensation: dict[str, dict] | None = field(default_factory=dict)
    trig_out_source: TrigOutSource = TrigOutSource.NONE


@dataclass
class HDAWGRecipeData:
    outputs: tuple[HDAWGOutputRecipeData, HDAWGOutputRecipeData] = field(
        default_factory=lambda: (HDAWGOutputRecipeData(), HDAWGOutputRecipeData())
    )
    iq_settings: tuple[int, int] | None = None


@dataclass
class UHFQAOutputRecipeData:
    enable: bool = False
    offset: float | None = None
    hw_modulation: HWModulation = HWModulation.NONE
    range: float | None = None


@dataclass
class UHFQARecipeData:
    outputs: tuple[UHFQAOutputRecipeData, UHFQAOutputRecipeData] = field(
        default_factory=lambda: (UHFQAOutputRecipeData(), UHFQAOutputRecipeData())
    )


@dataclass
class QAChannelRecipeData:
    output_enable: bool | None = None  # None means no output config provided
    output_range: float | None = None
    output_mute_enable: bool = False
    input_enable: bool | None = None  # None means no input config provided
    input_range: float | None = None
    input_rf_path: bool = True


@dataclass
class OutputRouteConfig:
    source: int
    fixed_amplitude: float | None = None  # None means parameterized amplitude
    fixed_phase: float | None = None  # None means parameterized phase
    param_amplitude: str | None = None  # None means fixed amplitude
    param_phase: str | None = None  # None means fixed phase


@dataclass
class SGChannelRecipeData:
    output_enable: bool | None = None  # None means no output config provided
    output_range: float | None = None
    output_mute_enable: bool = False
    modulation: bool | None = None
    marker_source_trigger: bool = True
    output_rf_path: bool = True
    router_config: list[OutputRouteConfig] | None = None

    def ensure_router_config(self):
        if self.router_config is None:
            self.router_config = []
        return self.router_config


@dataclass
class PPChannelRecipeData:
    settings: PPChannel


@dataclass
class AllocatedOscillator:
    channels: set[int]
    awg_type: AwgType
    index: int
    id_index: int
    frequency: float | None
    param: str | None


@dataclass
class DeviceRecipeData:
    is_present: bool = True
    start_trigger_repetitions: int = 1
    iq_settings: dict[int, tuple[int, int]] = field(default_factory=dict)
    allocated_oscs: list[AllocatedOscillator] = field(default_factory=list)
    uhfqacore: UHFQARecipeData = field(default_factory=UHFQARecipeData)
    hdawgcores: dict[int, HDAWGRecipeData] = field(default_factory=dict)
    qachannels: dict[int, QAChannelRecipeData] = field(default_factory=dict)
    sgchannels: dict[int, SGChannelRecipeData] = field(default_factory=dict)
    ppchannels: dict[int, PPChannelRecipeData] = field(default_factory=dict)
    awg_configs: dict[int, AwgConfig] = field(default_factory=dict)
    has_feedback: bool = False

    def allocated_awgs(self, default_awg: int | None = None) -> set[int]:
        awgs = set(self.awg_configs.keys())
        if len(awgs) == 0 and default_awg is not None:
            awgs.add(default_awg)
        return awgs


@dataclass
class RtExecutionInfo:
    averages: int
    averaging_mode: AveragingMode
    acquisition_type: AcquisitionType
    chunk_count: int | None

    @property
    def is_chunked(self) -> bool:
        return self.chunk_count is not None

    @property
    def chunk_count_or_1(self) -> int:
        return self.chunk_count or 1

    @property
    def is_raw_acquisition(self) -> bool:
        return self.acquisition_type == AcquisitionType.RAW

    @property
    def effective_averages(self) -> int:
        return 1 if self.averaging_mode == AveragingMode.SINGLE_SHOT else self.averages

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
    artifacts: ArtifactsCodegen | None, device_uid: str
) -> Initialization | None:
    if artifacts is None:
        return None
    for initialization in artifacts.initializations:
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
            "Experiment cannot be executed - compilation artifacts are incomplete. "
            "Possible cause: the experiment was compiled with the IGNORE_RESOURCE_LIMITATION_ERRORS setting."
        )

    return seqc["elf"]


@dataclass
class RecipeData:
    artifacts: CompilerArtifact
    execution: Statement
    rt_execution_info: RtExecutionInfo
    max_step_execution_time: float
    device_settings: dict[DeviceUID, DeviceRecipeData] = field(default_factory=dict)
    awg_configs: AwgConfigs = field(default_factory=AwgConfigs)
    attribute_value_tracker: AttributeValueTracker = field(
        default_factory=AttributeValueTracker
    )
    coprocessor_payloads: dict[str, bytes] = field(default_factory=dict)

    def get_initialization(self, device_uid: DeviceUID) -> Initialization:
        for initialization in self.artifacts.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        return Initialization(device_uid=device_uid)

    def get_artifacts(self, artifacts_class: type[T]) -> T:
        return get_artifacts(self.artifacts, artifacts_class)

    def awgs_producing_results(self) -> Iterator[tuple[AwgKey, AwgConfig]]:
        for awg_key, awg_config in self.awg_configs.items():
            if awg_config.result_length is not None:
                yield awg_key, awg_config

    def awg_by_seqc_name(self, seqc_name: str) -> AwgKey | None:
        for rt_exec_step in self.artifacts.realtime_execution_init:
            if rt_exec_step.program_ref == seqc_name:
                return AwgKey(rt_exec_step.device_id, rt_exec_step.awg_index)
        return None

    def allocated_awgs(
        self, device_uid: str, default_awg: int | None = None
    ) -> set[int]:
        awgs = {
            awg_key.awg_index
            for awg_key in self.awg_configs.keys()
            if awg_key.device_uid == device_uid
        }
        if len(awgs) == 0 and default_awg is not None:
            awgs.add(default_awg)
        return awgs

    @cached_method(1)
    def get_program_refs(self) -> dict[tuple[int, ...], set[str]]:
        refs = NearestDict(rounding=NearestDict.NEAREST_PREV)

        for init in self.artifacts.realtime_execution_init:
            nt_step_program_refs = refs.setdefault(init.nt_step.indices, set())
            nt_step_program_refs.add(init.program_ref)

        return refs


class _NeartimeCallbackCollector(ExecutorBase):
    """Walk the execution tree and collect near-time callback names."""

    def __init__(self) -> None:
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY | LoopingMode.ONCE)
        self.names: set[str] = set()

    def nt_callback_handler(
        self,
        func_name: str,
        args: dict[str, Any],
    ):
        self.names.add(func_name)


def collect_neartime_callback_names(execution: Statement) -> set[str]:
    """Collect the near-time callback names referenced by an execution tree."""
    collector = _NeartimeCallbackCollector()
    collector.run(execution)
    return collector.names


def validate_scheduled_experiment(
    scheduled_experiment: ScheduledExperiment,
    device_setup_fingerprint: str,
    neartime_callbacks: dict[str, Callable[..., Any]],
    allow_client_callbacks: bool = False,
):
    # Compiled experiment version is correct
    try:
        if (
            Version(Version(scheduled_experiment.versions.laboneq).base_version)
            < MIN_LABONEQ_VERSION_FOR_COMPILED_EXPERIMENT
        ):
            raise LabOneQControllerException(
                f"The experiment was compiled with LabOne Q version {scheduled_experiment.versions.laboneq}, which is not compatible. "
                "Please recompile using the current LabOne Q version."
            )
    except InvalidVersion:
        _logger.warning(
            "The experiment was compiled with an invalid LabOne Q version '%s'. "
            "Passing, as this is only expected to happen in tests. Consider ensuring that the valid version is set.",
            scheduled_experiment.versions.laboneq,
        )

    if scheduled_experiment.device_setup_fingerprint != device_setup_fingerprint:
        raise LabOneQControllerException(
            "The device setup of the compiled experiment is not compatible with the current device setup. "
            "Please recompile using the current device setup."
        )

    assert scheduled_experiment.execution is not None

    # Names not registered on the controller are dispatched to the client when a
    # callback channel is available; without one they must all resolve locally.
    names = collect_neartime_callback_names(scheduled_experiment.execution)
    missing = names - set(neartime_callbacks)
    if missing and not allow_client_callbacks:
        raise UnknownCallbackError(
            f"Experiment requires callbacks not registered on the "
            f"controller: {sorted(missing)}. "
            f"Available: {sorted(neartime_callbacks)}."
        )


def _resolve_coprocessor_payloads(
    scheduled_experiment: ScheduledExperiment,
    overrides: dict[str, bytes] | None,
) -> dict[str, bytes]:
    """Merge the initial coprocessor payload and post-compile overrides.

    Returns the resolved payloads, keyed by inventory key.
    """
    payloads = dict(scheduled_experiment.coprocessor_payloads)
    mappings = scheduled_experiment.coprocessor_mappings
    for label, payload in (overrides or {}).items():
        inventory_key = mappings.get(label)
        if inventory_key is None:
            raise LabOneQControllerException(
                f"Experiment provides a payload for '{label}', but '{label}' is not"
                f" a coprocessor of this experiment."
                f" Available: {sorted(mappings)}."
            )
        payloads[inventory_key] = payload
    return payloads


def pre_process_compiled(
    scheduled_experiment: ScheduledExperiment,
    coprocessor_payload_overrides: dict[str, bytes] | None = None,
) -> RecipeData:
    """Returns the recipe data of a compiled experiment, without instrument settings."""
    rt_loop_properties = scheduled_experiment.rt_loop_properties
    return RecipeData(
        artifacts=scheduled_experiment.artifacts,
        execution=scheduled_experiment.execution,
        rt_execution_info=RtExecutionInfo(
            averages=rt_loop_properties.shots,
            averaging_mode=rt_loop_properties.averaging_mode,
            acquisition_type=rt_loop_properties.acquisition_type,
            chunk_count=rt_loop_properties.chunk_count,
        ),
        max_step_execution_time=scheduled_experiment.max_step_execution_time,
        coprocessor_payloads=_resolve_coprocessor_payloads(
            scheduled_experiment, coprocessor_payload_overrides
        ),
    )


def _result_size_per_integrator(recipe_data: RecipeData) -> int:
    # max_result_length is in samples for a single integrator. Each sample is complex128, i.e. 16 bytes.
    return recipe_data.awg_configs.max_result_length * 16


def _result_size_per_job(recipe_data: RecipeData) -> int:
    # The device always returns data for all integrators, even if only a subset is used.
    # TODO(2K): Use the actual number of HW integrators once it is available here.
    # For now, assume the worst case of 16. This only affects timeout estimation, not runtime
    # performance when execution proceeds normally.
    return _result_size_per_integrator(recipe_data) * 16


def get_execution_time(recipe_data: RecipeData) -> tuple[float, float]:
    rt_execution_info = recipe_data.rt_execution_info
    min_wait_time = recipe_data.max_step_execution_time
    if rt_execution_info.is_chunked:
        internal_result_transfer_time = (
            _result_size_per_job(recipe_data) / RESULT_TRANSFER_RATE_GW_FW
        )
        min_wait_time = (
            # Assume reload and result transfer happen simultaneously, and take the worst case of the two.
            min_wait_time
            + max(PIPELINER_RELOAD_WORST_CASE, internal_result_transfer_time)
        ) * rt_execution_info.chunk_count_or_1

    # Add extra guard time for cases where execution state node updates are delayed
    # by an ongoing large result transfer from previously completed jobs. Assume that
    # up to two maximum-size result vectors may need to be transferred before the
    # state update is delivered.
    # TODO(2K): Pass the actual controller connection timeout for a more accurate
    # estimate that reflects the real network speed and load.
    extra_guard_for_results = (
        _result_size_per_integrator(recipe_data) / adjusted_transfer_rate() * 2
    )

    # +10% and fixed 1sec guard time rounded up to next full second
    guarded_wait_time = round((min_wait_time + extra_guard_for_results) * 1.1 + 1.5)

    return min_wait_time, guarded_wait_time


def get_weights_info(
    artifacts: ArtifactsCodegen, kernel_ref: str | None
) -> dict[int, list[WeightInfo]]:
    if kernel_ref is None:
        return {}
    # Group by channel
    weights = artifacts.integration_weights.get(kernel_ref, [])
    out: dict[int, list[WeightInfo]] = {}
    for weight in weights:
        for channel in weight.integration_units:
            out.setdefault(channel, []).append(weight)
    return out


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
        samples=convert_awg_waveform()(
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
        samples=convert_awg_waveform()(
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
    wave_indices: dict[str, tuple[int, WaveType]] = cast(
        "dict[str, tuple[int, WaveType]]",
        next(
            (i for i in artifacts.wave_indices if i["filename"] == wave_indices_ref),
            {},
        )["value"],
    )

    waves: Waveforms = []
    for sig, (index, sig_type) in [
        i for i in wave_indices.items() if i[0] != "filename"
    ]:
        if sig.startswith("precomp_reset"):
            continue  # precomp reset waveform is bundled with ELF
        if sig_type in (WaveType.IQ, WaveType.DOUBLE):
            wave = _prepare_wave_iq(artifacts.waves, sig, index)
        elif sig_type == WaveType.SINGLE:
            wave = _prepare_wave_single(artifacts.waves, sig, index)
        elif sig_type == WaveType.COMPLEX:
            wave = _prepare_wave_complex(artifacts.waves, sig, index)
        else:
            raise LabOneQControllerException(
                f"Unexpected signal type for binary wave for '{sig}' in '{wave_indices_ref}' - "
                f"'{sig_type}', should be one of [iq, double, single, complex]"
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
