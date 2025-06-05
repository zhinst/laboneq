# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import groupby
import math
from numbers import Number
from typing import Any, NamedTuple, Literal, cast
from laboneq._rust import codegenerator as codegen_rs
import numpy as np
from numpy import typing as npt
from engineering_notation import EngNumber
import scipy.signal
from laboneq.core.types.enums import AcquisitionType
from laboneq.compiler.common.compiler_settings import (
    TINYSAMPLE,
    CompilerSettings,
    round_min_playwave_hint,
)
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.resource_usage import (
    ResourceLimitationError,
    ResourceUsage,
)
from laboneq.compiler.common.shfppc_sweeper_config import SHFPPCSweeperConfig
from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout
from laboneq.compiler.ir import IRTree
from laboneq.compiler.seqc import passes, ir as ir_code
from laboneq.compiler.seqc.linker import AwgWeights, SeqCGenOutput, SeqCProgram
from laboneq.compiler.seqc.analyze_events import (
    analyze_loop_times,
    analyze_phase_reset_times,
    analyze_set_oscillator_times,
    analyze_trigger_events,
    analyze_prng_times,
)
from laboneq.compiler.seqc.command_table_tracker import CommandTableTracker
from laboneq.compiler.seqc.feedback_register_allocator import (
    FeedbackRegisterAllocator,
    FeedbackRegisterAllocation,
)
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.seqc import ir_to_event_list
from laboneq.compiler.seqc.measurement_calculator import (
    IntegrationTimes,
    SignalDelays,
)
from laboneq.compiler.seqc.sampled_event_handler import SampledEventHandler
from laboneq.compiler.seqc.shfppc_sweeper_config_tracker import (
    SHFPPCSweeperConfigTracker,
)
from laboneq.compiler.seqc.signatures import PlaybackSignature, SamplesSignatureID
from laboneq.compiler.seqc.utils import normalize_phase
from laboneq.compiler.seqc.wave_compressor import (
    PlayHold,
    PlaySamples,
    WaveCompressor,
)
from laboneq._rust.codegenerator import (
    WaveIndexTracker,
    SeqCGenerator,
    SeqCTracker,
    seqc_generator_from_device_and_signal_type as seqc_generator_from_device_and_signal_type_str,
    merge_generators,
    WaveformSignature,
)
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.event_list.event_type import EventList
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    length_to_samples,
    sample_pulse,
    verify_amplitude_no_clipping,
)
from laboneq.data.compilation_job import PulseDef
from laboneq.data.scheduled_experiment import (
    CodegenWaveform,
    MixerType,
    PulseInstance,
    PulseMapEntry,
    PulseWaveformMap,
    WeightInfo,
    COMPLEX_USAGE,
)
from laboneq.compiler.seqc import compat_rs

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplesSignature:
    """Samples signature.

    The underlying promise is that two keys with the same values are guaranteed to resolve to the same samples
    across a single AWG."""

    samples_i: np.ndarray
    samples_q: np.ndarray | None = None
    samples_marker1: np.ndarray | None = None
    samples_marker2: np.ndarray | None = None

    @staticmethod
    def _compare_maybe_arrays(one: np.ndarray | None, other: np.ndarray | None) -> bool:
        """Compare two arrays or None."""
        if one is None and other is None:
            return True
        if one is None or other is None:
            return False
        return np.array_equal(one, other, equal_nan=True)

    def uid(self) -> int:
        """Return a stable unique identifier for the samples."""
        return hash(self)

    def __eq__(self, other: SamplesSignature):
        if not isinstance(other, SamplesSignature):
            return NotImplemented
        return (
            np.array_equal(self.samples_i, other.samples_i)
            and self._compare_maybe_arrays(self.samples_q, other.samples_q)
            and self._compare_maybe_arrays(self.samples_marker1, other.samples_marker1)
            and self._compare_maybe_arrays(self.samples_marker2, other.samples_marker2)
        )

    def __hash__(self):
        def arr_to_bytes(arr: np.ndarray | None) -> bytes:
            return arr.tobytes() if arr is not None else b""

        h = hashlib.md5()
        h.update(arr_to_bytes(self.samples_i))
        h.update(arr_to_bytes(self.samples_q))
        h.update(arr_to_bytes(self.samples_marker1))
        h.update(arr_to_bytes(self.samples_marker2))
        for arr in (
            self.samples_i,
            self.samples_q,
            self.samples_marker1,
            self.samples_marker2,
        ):
            if arr is not None:
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtype).encode())
        return int.from_bytes(h.digest()[:8], "big", signed=False)


@dataclass
class SampledWaveformSignature:
    samples: SamplesSignature
    # Waveform per pulse def
    pulse_map: dict[str, PulseWaveformMap] = field(default_factory=dict)
    # Compression parameters
    hold_start: int | None = None
    hold_length: int | None = None

    @property
    def samples_i(self) -> np.ndarray:
        """Return the I samples."""
        return self.samples.samples_i

    @property
    def samples_q(self) -> np.ndarray | None:
        """Return the Q samples."""
        return self.samples.samples_q

    @property
    def samples_marker1(self) -> np.ndarray | None:
        """Return the marker1 samples."""
        return self.samples.samples_marker1

    @property
    def samples_marker2(self) -> np.ndarray | None:
        """Return the marker2 samples."""
        return self.samples.samples_marker2


@dataclass(frozen=True)
class _SampledWaveform:
    signals: tuple[str]
    signature: SampledWaveformSignature
    signature_string: str


def generate_sampled_waveform_signature(
    samples_i: np.ndarray,
    samples_q: np.ndarray | None = None,
    samples_marker1: np.ndarray | None = None,
    samples_marker2: np.ndarray | None = None,
) -> tuple[SamplesSignatureID, SampledWaveformSignature]:
    """Generate a `SamplesSignatureID` and `SampledSignature` from the provided samples."""
    label = "compr"
    signature = SamplesSignature(
        samples_i=samples_i,
        samples_q=samples_q,
        samples_marker1=samples_marker1,
        samples_marker2=samples_marker2,
    )
    return SamplesSignatureID(
        label=label,
        uid=signature.uid(),
        has_i=signature.samples_i is not None,
        has_q=signature.samples_q is not None,
        has_marker1=signature.samples_marker1 is not None,
        has_marker2=signature.samples_marker2 is not None,
    ), signature


def seqc_generator_from_device_and_signal_type(
    device_type: DeviceType,
    signal_type: AWGSignalType,
) -> SeqCGenerator:
    return seqc_generator_from_device_and_signal_type_str(
        device_type.value, signal_type.value
    )


def add_wait_trigger_statements(
    awg: AWGInfo,
    init_generator: SeqCGenerator,
    deferred_function_calls: SeqCGenerator,
):
    if awg.trigger_mode == TriggerMode.DIO_TRIGGER:
        # HDAWG+UHFQA connected via DIO, no PQSC
        if awg.awg_id == 0:
            assert awg.reference_clock_source != "internal", (
                "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter."
            )
            init_generator.add_function_call_statement("waitDigTrigger", ["1"])
            init_generator.add_function_call_statement("setDIO", ["0xffffffff"])
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_first_awg_samples = str(
                round(awg.sampling_rate * CodeGenerator.DELAY_FIRST_AWG / 16) * 16
            )
            if int(delay_first_awg_samples) > 0:
                deferred_function_calls.add_function_call_statement(
                    "playZero", [delay_first_awg_samples]
                )
        else:
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_other_awg_samples = str(
                round(awg.sampling_rate * CodeGenerator.DELAY_OTHER_AWG / 16) * 16
            )
            if int(delay_other_awg_samples) > 0:
                deferred_function_calls.add_function_call_statement(
                    "playZero", [delay_other_awg_samples]
                )
    elif awg.trigger_mode == TriggerMode.INTERNAL_READY_CHECK:
        # Standalone HDAWG
        # We don't need to do anything for alignment because ready check
        # mechanism handles that.
        pass

    elif awg.trigger_mode == TriggerMode.DIO_WAIT:
        # UHFQA triggered by HDAWG
        init_generator.add_function_call_statement("waitDIOTrigger")
        delay_uhfqa_samples = str(
            round(awg.sampling_rate * CodeGenerator.DELAY_UHFQA / 8) * 8
        )
        if int(delay_uhfqa_samples) > 0:
            init_generator.add_function_call_statement(
                "playZero", [delay_uhfqa_samples]
            )

    elif awg.trigger_mode == TriggerMode.INTERNAL_TRIGGER_WAIT:
        # SHFQC, internally triggered
        init_generator.add_function_call_statement("waitDigTrigger", ["1"])

    else:
        if CodeGenerator.USE_ZSYNC_TRIGGER and awg.device_type.supports_zsync:
            # Any instrument triggered directly via ZSync
            init_generator.add_function_call_statement("waitZSyncTrigger")
        else:
            # UHFQA triggered by PQSC (forwarded over DIO)
            init_generator.add_function_call_statement("waitDIOTrigger")


def stable_hash(samples_i: np.ndarray, samples_q: np.ndarray):
    """Stable hashing function for generating names for readout weights"""
    return hashlib.sha1(samples_i.tobytes() + samples_q.tobytes())


@dataclass
class IntegrationWeight:
    basename: str
    samples_i: npt.ArrayLike
    samples_q: npt.ArrayLike
    downsampling_factor: int | None


def calculate_integration_weights(
    event_list_acquire: list[AWGEvent],
    signal_obj: SignalObj,
    pulse_defs: dict[str, PulseDef],
    pulse_params: list[passes.PulseParams],
) -> list[IntegrationWeight]:
    integration_weights_by_pulse: dict[str, IntegrationWeight] = {}
    signal_id = signal_obj.id
    already_handled = set()
    nr_of_weights_per_event = None
    for event in event_list_acquire:
        play_wave_ids: list[str] = event.params.get("play_wave_id")
        id_pulse_params: list[int] = event.params["id_pulse_params"]
        oscillator_frequency: float = event.params["oscillator_frequency"]
        key_ = (
            tuple(play_wave_ids),
            tuple(id_pulse_params),
            oscillator_frequency,
        )
        if key_ in already_handled:
            continue
        assert event.params.get("signal_id", signal_id) == signal_id

        filtered_pulses: list[tuple[str, int | None]] = [
            (play_wave_id, param_id)
            for play_wave_id, param_id in zip(
                play_wave_ids, id_pulse_params or [None] * len(play_wave_ids)
            )
            if play_wave_id is not None
            and (pulse_def := pulse_defs.get(play_wave_id)) is not None
            # Not a real pulse, just a placeholder for the length - skip
            and (pulse_def.samples is not None or pulse_def.function is not None)
        ]
        nr_of_weights = len(play_wave_ids)
        if nr_of_weights_per_event is None:
            nr_of_weights_per_event = nr_of_weights
        else:
            assert nr_of_weights == nr_of_weights_per_event
            if any(w not in integration_weights_by_pulse for w, _ in filtered_pulses):
                # Event uses different pulse UIDs than earlier event
                raise LabOneQException(
                    f"Using different integration kernels on a single signal"
                    f" ({signal_id}) is unsupported. Weights: {play_wave_ids} vs"
                    f" {list(integration_weights_by_pulse.keys())}"
                )

        for play_wave_id, pulse_param_id in filtered_pulses:
            pulse_def = pulse_defs[play_wave_id]
            samples = pulse_def.samples
            amplitude: Number = pulse_def.amplitude
            assert isinstance(amplitude, Number)
            length = pulse_def.length
            if length is None:
                length = len(samples) / signal_obj.awg.sampling_rate
            if pulse_param_id is not None:
                params_pulse_combined = pulse_params[pulse_param_id].combined()
            else:
                params_pulse_combined = None
            iw_samples = sample_pulse(
                signal_type="iq",
                sampling_rate=signal_obj.awg.sampling_rate,
                length=length,
                amplitude=amplitude,
                pulse_function=pulse_def.function,
                modulation_frequency=oscillator_frequency,
                samples=samples,
                mixer_type=signal_obj.mixer_type,
                pulse_parameters=params_pulse_combined,
            )

            if (
                signal_obj.awg.device_type == DeviceType.SHFQA
                and (weight_len := len(iw_samples["samples_i"])) > 4096
            ):  # TODO(2K): get via device_type
                downsampling_factor = math.ceil(weight_len / 4096)
                if downsampling_factor > 16:  # TODO(2K): get via device_type
                    raise LabOneQException(
                        "Integration weight length exceeds the maximum supported by HW"
                    )
            else:
                downsampling_factor = None

            verify_amplitude_no_clipping(
                samples_i=iw_samples["samples_i"],
                samples_q=iw_samples["samples_q"],
                pulse_id=pulse_def.uid,
                mixer_type=signal_obj.mixer_type,
                signals=(signal_obj.id,),
            )

            # 128-bit hash as waveform name
            digest = stable_hash(
                samples_i=iw_samples["samples_i"], samples_q=iw_samples["samples_q"]
            ).hexdigest()[:32]
            integration_weight = IntegrationWeight(
                basename=f"kernel_{digest}",
                samples_i=iw_samples["samples_i"],
                samples_q=iw_samples["samples_q"],
                downsampling_factor=downsampling_factor,
            )

            if existing_weight := integration_weights_by_pulse.get(play_wave_id):
                for existing, new in zip(
                    [existing_weight.samples_i, existing_weight.samples_q],
                    [integration_weight.samples_i, integration_weight.samples_q],
                ):
                    if np.any(existing != new):
                        if np.any(np.isnan(new)):
                            # this is because nan != nan is always true
                            raise LabOneQException(
                                "Encountered NaN in an integration kernel."
                            )
                        # Kernel differs even though pulse ID is the same
                        # (e.g. affected by pulse parameters)
                        raise LabOneQException(
                            f"Using different integration kernels on a single signal"
                            f" ({signal_id}) is unsupported."
                        )
            # This relies on the dict remembering insertion order.
            integration_weights_by_pulse[play_wave_id] = integration_weight
        already_handled.add(key_)
    return list(integration_weights_by_pulse.values())


_SEQUENCER_TYPES = {DeviceType.SHFQA: "qa", DeviceType.SHFSG: "sg"}


class CodeGenerator(ICodeGenerator):
    USE_ZSYNC_TRIGGER = True

    DELAY_FIRST_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_OTHER_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_UHFQA = 128 / DeviceType.UHFQA.sampling_rate

    # This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
    # to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
    SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10

    def __init__(
        self,
        ir: IRTree,
        signals: list[SignalObj],
        feedback_register_layout: FeedbackRegisterLayout | None = None,
        settings: CompilerSettings | dict | None = None,
    ):
        if settings is not None:
            if isinstance(settings, CompilerSettings):
                self._settings = settings
            else:
                self._settings = CompilerSettings(**settings)
        else:
            self._settings = CompilerSettings()

        self._ir = ir
        self._awgs: dict[AwgKey, AWGInfo] = {}
        self._signals: dict[str, SignalObj] = {}
        for signal_obj in signals:
            self.add_signal(signal_obj)
        self._src: dict[AwgKey, SeqCProgram] = {}
        self._wave_indices_all: dict[AwgKey, dict] = {}
        self._waves: dict[str, CodegenWaveform] = {}
        self._requires_long_readout: dict[str, list[str]] = defaultdict(list)
        self._command_tables: dict[AwgKey, dict[str, Any]] = {}
        self._pulse_map: dict[str, PulseMapEntry] = {}
        self._parameter_phase_increment_map: dict[
            AwgKey, dict[str, list[int | Literal[COMPLEX_USAGE]]]
        ] = {}
        self._sampled_waveforms: dict[AwgKey, list[_SampledWaveform]] = {}
        self._integration_times: IntegrationTimes | None = None
        self._signal_delays: SignalDelays | None = None
        # awg key -> signal id -> kernel index -> kernel data
        self._integration_weights: dict[AwgKey, dict[str, list[IntegrationWeight]]] = (
            defaultdict(dict)
        )
        self._simultaneous_acquires: list[dict[str, str]] = []
        self._feedback_register_layout = feedback_register_layout or {}
        self._feedback_register_config: dict[AwgKey, FeedbackRegisterConfig] = (
            defaultdict(FeedbackRegisterConfig)
        )
        self._feedback_connections: dict[str, FeedbackConnection] = {}
        self._qa_signals_by_handle: dict[str, SignalObj] = {}
        self._shfppc_sweep_configs: dict[AwgKey, SHFPPCSweeperConfig] = {}
        self._total_execution_time: float | None = None
        self._max_resource_usage: ResourceUsage | None = None

    def generate_code(self):
        passes.inline_sections_in_branch(self._ir)
        measurement_info = passes.collect_measurement_info(self._ir.root, self._signals)
        self.gen_seq_c(
            pulse_defs={p.uid: p for p in self._ir.pulse_defs},
            qa_signals_by_handle=measurement_info.qa_signals_by_handle,
            simultaneous_acquires=measurement_info.simultaneous_acquires,
            feedback_register_allocator=measurement_info.feedback_register_allocator,
            integration_times=measurement_info.integration_times,
            signal_delays=measurement_info.signal_delays,
            total_execution_time=self._ir.root.length * TINYSAMPLE,
        )

    def get_output(self):
        return SeqCGenOutput(
            feedback_connections=self.feedback_connections(),
            signal_delays=self.signal_delays(),
            integration_weights=self.integration_weights(),
            integration_times=self.integration_times(),
            simultaneous_acquires=self.simultaneous_acquires(),
            src=self.src(),
            total_execution_time=self.total_execution_time(),
            waves=self.waves(),
            requires_long_readout=self.requires_long_readout(),
            wave_indices=self.wave_indices(),
            command_tables=self.command_tables(),
            pulse_map=self.pulse_map(),
            parameter_phase_increment_map=self.parameter_phase_increment_map(),
            feedback_register_configurations=self.feedback_register_config(),
            shfppc_sweep_configurations=self.shfppc_sweep_configs(),
        )

    def integration_weights(self) -> dict[AwgKey, AwgWeights]:
        return {
            awg: {
                signal: [
                    WeightInfo(iw.basename, iw.downsampling_factor or 1) for iw in l
                ]
                for signal, l in d.items()
            }
            for awg, d in self._integration_weights.items()
        }

    def simultaneous_acquires(self) -> list[dict[str, str]]:
        return self._simultaneous_acquires

    def total_execution_time(self):
        return self._total_execution_time

    def add_signal(self, signal_obj: SignalObj):
        self._signals[signal_obj.id] = signal_obj
        awg_key = signal_obj.awg.key
        if awg_key not in self._awgs:
            self._awgs[awg_key] = signal_obj.awg

    def sort_signals(self):
        for awg in self._awgs.values():
            awg.signals = list(
                sorted(awg.signals, key=lambda signal: tuple(signal.channels))
            )

    def _append_to_pulse_map(self, signature_pulse_map, sig_string):
        if signature_pulse_map is None:
            return
        for pulse_id, pulse_waveform_map in signature_pulse_map.items():
            pulse_map_entry = self._pulse_map.setdefault(pulse_id, PulseMapEntry())
            pulse_map_entry.waveforms[sig_string] = pulse_waveform_map

    def _save_wave_bin(
        self,
        samples,
        signature_pulse_map,
        sig_string: str,
        suffix: str,
        device_id: str | None = None,
        signal_id: str | None = None,
        hold_start: int | None = None,
        hold_length: int | None = None,
        downsampling_factor: int | None = None,
    ):
        filename = sig_string + suffix + ".wave"
        wave = CodegenWaveform(
            samples=samples,
            hold_start=hold_start,
            hold_length=hold_length,
            downsampling_factor=downsampling_factor,
        )
        assert filename not in self._waves or np.allclose(
            self._waves[filename].samples, wave.samples
        ), filename
        self._waves[filename] = wave
        if (
            hold_start is not None
            or hold_length is not None
            or downsampling_factor is not None
        ):
            assert device_id is not None
            assert signal_id is not None
            device_long_readout_signals = self._requires_long_readout[device_id]
            if signal_id not in device_long_readout_signals:
                device_long_readout_signals.append(signal_id)
        self._append_to_pulse_map(signature_pulse_map, sig_string)

    def _gen_waves(self):
        for awg in self._awgs.values():
            # Handle integration weights separately
            for signal_obj in awg.signals:
                for weight in self._integration_weights[awg.key].get(signal_obj.id, []):
                    if awg.device_type.supports_complex_waves:
                        self._save_wave_bin(
                            CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                            * (weight.samples_i - 1j * weight.samples_q),
                            None,
                            weight.basename,
                            "",
                            device_id=awg.device_id,
                            signal_id=signal_obj.id,
                            downsampling_factor=weight.downsampling_factor,
                        )
                    else:
                        self._save_wave_bin(
                            weight.samples_i, None, weight.basename, "_i"
                        )
                        self._save_wave_bin(
                            weight.samples_q, None, weight.basename, "_q"
                        )

            if awg.signal_type in (AWGSignalType.IQ, AWGSignalType.SINGLE):
                for sampled_waveform in self._sampled_waveforms.get(awg.key, []):
                    assert len(sampled_waveform.signals) == 1
                    signals = sampled_waveform.signals
                    sampled_signature = sampled_waveform.signature
                    sig_string = sampled_waveform.signature_string
                    if awg.device_type.supports_binary_waves:
                        if awg.signal_type == AWGSignalType.SINGLE:
                            self._save_wave_bin(
                                sampled_signature.samples_i,
                                sampled_signature.pulse_map,
                                sig_string,
                                "",
                            )
                            if sampled_signature.samples_marker1 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker1,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker1",
                                )
                            if sampled_signature.samples_marker2 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker2,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker2",
                                )
                        else:
                            self._save_wave_bin(
                                sampled_signature.samples_i,
                                sampled_signature.pulse_map,
                                sig_string,
                                "_i",
                            )
                            if sampled_signature.samples_q is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_q,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_q",
                                )
                            if sampled_signature.samples_marker1 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker1,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker1",
                                )
                            if sampled_signature.samples_marker2 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker2,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker2",
                                )
                    elif awg.device_type.supports_complex_waves:
                        self._save_wave_bin(
                            CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                            * (
                                sampled_signature.samples_i
                                - 1j * sampled_signature.samples_q
                            ),
                            sampled_signature.pulse_map,
                            sig_string,
                            "",
                            device_id=awg.device_id,
                            signal_id=list(signals)[0],
                            hold_start=sampled_signature.hold_start,
                            hold_length=sampled_signature.hold_length,
                        )
                    else:
                        raise RuntimeError(
                            f"Device type {awg.device_type} has invalid supported waves config."
                        )
            else:
                for waveform in self._sampled_waveforms.get(awg.key, []):
                    if not awg.device_type.supports_binary_waves:
                        raise RuntimeError(
                            f"Device type {awg.device_type} has invalid supported waves config."
                        )
                    sampled_signature = waveform.signature
                    sig_string = waveform.signature_string
                    self._save_wave_bin(
                        sampled_signature.samples_i,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_i",
                    )
                    if sampled_signature.samples_q is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_q,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_q",
                        )
                    if sampled_signature.samples_marker1 is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_marker1,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_marker1",
                        )
                    if sampled_signature.samples_marker2 is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_marker2,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_marker2",
                        )

        # check that there are no duplicate filenames in the wave pool (QCSW-1079)
        waves = sorted(
            [(filename, wave.samples) for filename, wave in self._waves.items()],
            key=lambda w: w[0],
        )
        for _, group in groupby(waves, key=lambda w: w[0]):
            group = list(group)
            assert all(np.all(group[0][1] == g[1]) for g in group[1:])

    def gen_seq_c(
        self,
        pulse_defs: dict[str, PulseDef],
        qa_signals_by_handle: dict[str, SignalObj],
        simultaneous_acquires: list[dict[str, str]],
        feedback_register_allocator: FeedbackRegisterAllocator,
        integration_times: IntegrationTimes,
        signal_delays: SignalDelays,
        total_execution_time: float,
    ):
        # TODO: Remove state
        self._integration_times = integration_times
        self._signal_delays = signal_delays
        self._simultaneous_acquires = simultaneous_acquires
        self._qa_signals_by_handle = qa_signals_by_handle
        self._total_execution_time = total_execution_time

        self.sort_signals()
        awgs_sorted = sorted(
            self._awgs.values(),
            key=lambda item: item.key,
        )
        feedback_register_alloc = passes.allocate_feedback_registers(
            awgs=awgs_sorted,
            signal_to_handle={
                sig.id: handle for handle, sig in self._qa_signals_by_handle.items()
            },
            feedback_register_allocator=feedback_register_allocator,
        )
        # The pass must happen after delay calculation as it relies on section information.
        # TODO: Fix that this pass can happen as early as possible.
        passes.inline_sections(self._ir.root)
        # Detach must happen before creating the event list
        user_pulse_params = passes.detach_pulse_params(self._ir.root)
        ir_tree = passes.fanout_awgs(self._ir, self._awgs.values())
        ir_awgs: dict[AwgKey, ir_code.SingleAwgIR] = {}
        for child in ir_tree.root.children:
            awg = cast(ir_code.SingleAwgIR, child)
            ir_awgs[awg.awg.key] = awg
        events_per_awg, _ = ir_to_event_list.event_list_per_awg(ir_tree, self._settings)

        for signal_id, signal_obj in self._signals.items():
            code_generation_delay = self._signal_delays.get(signal_id)
            if code_generation_delay is not None:
                signal_obj.total_delay = (
                    signal_obj.start_delay
                    + signal_obj.delay_signal
                    + code_generation_delay.code_generation
                )
                signal_obj.on_device_delay = code_generation_delay.on_device
            else:
                signal_obj.total_delay = (
                    signal_obj.start_delay + signal_obj.delay_signal
                )

            _logger.debug(
                "Processed signal obj %s signal_obj.start_delay=%s  signal_obj.delay_signal=%s signal_obj.total_delay=%s  signal_obj.on_device_delay=%s",
                signal_id,
                EngNumber(signal_obj.start_delay),
                EngNumber(signal_obj.delay_signal),
                EngNumber(signal_obj.total_delay),
                EngNumber(signal_obj.on_device_delay),
            )

        self._integration_weights.clear()
        for awg in awgs_sorted:
            if awg.key not in ir_awgs:
                continue
            self._gen_seq_c_per_awg(
                awg_ir=ir_awgs[awg.key],
                events=events_per_awg.get(awg.key, []),
                pulse_defs=pulse_defs,
                user_pulse_params=user_pulse_params,
                feedback_register=feedback_register_alloc.get(awg.key),
                acquisition_type=self._ir.root.acquisition_type,
            )
        if (
            self._max_resource_usage is not None
            and self._max_resource_usage.usage > 1.0
        ):
            raise ResourceLimitationError(
                f"Exceeded resource limitation: {self._max_resource_usage}.\n"
            )

        for awg_key, seqc_program in self._src.items():
            awg_info = self._awgs[awg_key]
            assert isinstance(awg_info.awg_id, int)
            seqc_program.dev_type = awg_info.dev_type
            seqc_program.dev_opts = awg_info.dev_opts
            seqc_program.awg_index = awg_info.awg_id
            seqc_program.sequencer = _SEQUENCER_TYPES.get(awg_info.device_type, "auto")
            seqc_program.sampling_rate = (
                awg_info.sampling_rate
                if awg_info.device_type == DeviceType.HDAWG
                else None
            )

        tgt_feedback_regs = feedback_register_allocator.target_feedback_registers
        for awg, target_fb_register in tgt_feedback_regs.items():
            feedback_reg_config = self._feedback_register_config[awg]
            feedback_reg_config.target_feedback_register = target_fb_register
        self._gen_waves()

    @staticmethod
    def _calc_global_awg_params(awg: AWGInfo) -> tuple[float, float]:
        global_sampling_rate = None
        global_delay = None
        signals_so_far = set()
        all_relevant_delays = {}
        for signal_obj in awg.signals:
            _logger.debug(f"considering signal {signal_obj.id}")
            if awg.device_type == DeviceType.UHFQA:
                # on the UHFQA, we allow an individual delay_signal on the measure (play) line, even though we can't
                # shift the play time with a node on the device
                # for this to work, we need to ignore the play delay when generating code for loop events
                # and we use the start delay (lead time) to calculate the global delay
                relevant_delay = signal_obj.start_delay
            else:
                relevant_delay = signal_obj.total_delay

            if (
                round(relevant_delay * signal_obj.awg.sampling_rate)
                % signal_obj.awg.device_type.sample_multiple
                != 0
            ):
                raise RuntimeError(
                    f"Delay {relevant_delay} s = {round(relevant_delay * signal_obj.awg.sampling_rate)} samples on signal {signal_obj.id} is not compatible with the sample multiple of {signal_obj.awg.device_type.sample_multiple} on {signal_obj.awg.device_type}"
                )
            all_relevant_delays[signal_obj.id] = relevant_delay

            if signal_obj.signal_type != "integration":
                if awg.signal_type != AWGSignalType.IQ and global_delay is not None:
                    if global_delay != relevant_delay:
                        raise RuntimeError(
                            f"Delay {relevant_delay * 1e9:.2f} ns on signal "
                            f"{signal_obj.id} is different from other delays "
                            f"({global_delay * 1e9:.2f} ns) on the same AWG, on signals {signals_so_far}"
                        )
                if global_delay is not None:
                    if relevant_delay < global_delay:
                        # use minimum delay as global delay
                        # this makes sure that loop start events happen first and are not shifted beyond loop body events
                        global_delay = relevant_delay
                else:
                    global_delay = relevant_delay

            global_sampling_rate = signal_obj.awg.sampling_rate
            signals_so_far.add(signal_obj.id)

        if global_delay is None:
            global_delay = 0

        if (
            0
            < global_delay
            < awg.device_type.min_play_wave / awg.device_type.sampling_rate
        ):
            global_delay = 0

        _logger.debug(
            "Global delay for %s awg %s: %ss, calculated from all_relevant_delays %s",
            awg.device_id,
            awg.awg_id,
            EngNumber(global_delay or 0),
            [(s, EngNumber(d)) for s, d in all_relevant_delays.items()],
        )

        return global_sampling_rate, global_delay

    @staticmethod
    def _emit_new_awg_events(
        old_event: AWGEvent,
        new_events: list[Any],
        sampled_signatures: dict[
            codegen_rs.WaveformSignature, SampledWaveformSignature
        ],
    ) -> list[AWGEvent]:
        new_awg_events: list[AWGEvent] = []
        time = old_event.start
        for new_event in new_events:
            if isinstance(new_event, PlayHold):
                new_awg_events.append(
                    AWGEvent(
                        type=AWGEventType.PLAY_HOLD,
                        start=time,
                        end=time + new_event.num_samples,
                    )
                )
                time += new_event.num_samples
            if isinstance(new_event, PlaySamples):
                new_params = copy.deepcopy(old_event.params)
                new_length = len(next(iter(new_event.samples.values())))

                playback_signature: PlaybackSignature = new_params["playback_signature"]
                samples_signature_id, samples_signature = (
                    generate_sampled_waveform_signature(
                        samples_i=new_event.samples.get("samples_i"),
                        samples_q=new_event.samples.get("samples_q"),
                        samples_marker1=new_event.samples.get("samples_marker1"),
                        samples_marker2=new_event.samples.get("samples_marker2"),
                    )
                )
                waveform = codegen_rs.WaveformSignature.from_samples_id(
                    length=new_length,
                    uid=samples_signature_id.uid,
                    label=samples_signature_id.label,
                    has_i=samples_signature_id.has_i,
                    has_q=samples_signature_id.has_q,
                    has_marker1=samples_signature_id.has_marker1,
                    has_marker2=samples_signature_id.has_marker2,
                )
                playback_signature.waveform = waveform
                new_awg_events.append(
                    AWGEvent(
                        type=AWGEventType.PLAY_WAVE,
                        start=time,
                        end=time + new_length,
                        params=new_params,
                    )
                )
                sampled_signature_new = copy.deepcopy(
                    sampled_signatures[old_event.signature.waveform]
                )
                sampled_signature_new.samples = samples_signature
                # update 3 things in samples signatures
                #   - name of pulses that have been compressed
                #   - length_samples entry in signature pulse map
                #   - length entry in the instances stored in the signature pulse map
                pulse_map = sampled_signature_new.pulse_map
                for pulse_name in list(pulse_map.keys()):
                    pulse_map[pulse_name + f"_compr_{new_event.label}"] = pulse_map.pop(
                        pulse_name
                    )
                for sp_map in pulse_map.values():
                    sp_map.length_samples = new_length
                    for signature in sp_map.instances:
                        signature.length = new_length
                sampled_signatures[playback_signature.waveform] = sampled_signature_new
                time += new_length
        return new_awg_events

    @staticmethod
    def _pulses_compatible_for_compression(pulses: list[NamedTuple]) -> bool:
        sorted_pulses = sorted(pulses, key=lambda x: x.start)
        n = len(sorted_pulses)

        for i in range(n - 1):
            pi = sorted_pulses[i]
            pj = sorted_pulses[i + 1]

            if pi.end > pj.start and pi.can_compress != pj.can_compress:
                return False

        return True

    @staticmethod
    def _compress_waves(
        sampled_events: AWGSampledEventSequence,
        sampled_signatures: dict[WaveformSignature, SampledWaveformSignature],
        pulse_defs: dict[str, PulseDef],
        device_type: DeviceType,
    ):
        """Compress waves.

        Modifies `sampled_signatures` in-place.
        """
        wave_compressor = WaveCompressor()
        compressed_waveform_signatures = set()
        for event_group in sampled_events.sequence.values():
            event_replacement = {}
            for i, event in enumerate(event_group):
                if event.type == AWGEventType.ACQUIRE:
                    if any(
                        pulse_defs[id].can_compress
                        for id in event.params["play_wave_id"]
                    ):
                        _logger.warning(
                            "Compression for integration pulses is not supported. %s, for which compression has been requested, will not be compressed.",
                            event.params["play_wave_id"],
                        )
                if event.type == AWGEventType.PLAY_WAVE:
                    wave_form = event.signature.waveform
                    pulses_not_in_pulsedef = [
                        pulse.pulse
                        for pulse in wave_form.pulses
                        if pulse.pulse not in pulse_defs
                    ]
                    assert all(pulse is None for pulse in pulses_not_in_pulsedef)
                    if len(pulses_not_in_pulsedef) > 0:
                        continue

                    # SHFQA long readout measure pulses
                    if (
                        any(
                            pulse_defs[pulse.pulse].can_compress
                            for pulse in wave_form.pulses
                        )
                        and device_type.is_qa_device
                    ):
                        sampled_signature = sampled_signatures.get(wave_form)
                        if sampled_signature is None:  # Non-matching channel
                            continue
                        orig_i = sampled_signature.samples_i
                        if len(orig_i) <= 4096:  # TODO(2K): get via device_type
                            # Measure pulse is fitting into memory, no additional processing needed
                            continue
                        orig_q = sampled_signature.samples_q
                        sample_dict = {
                            "i": orig_i,
                            "q": orig_q,
                        }
                        new_events = wave_compressor.compress_wave(
                            sample_dict, sample_multiple=4, threshold=12
                        )
                        if new_events is None:
                            raise LabOneQException(
                                "SHFQA measure pulse exceeds 4096 samples and is not compressible."
                            )
                        lead_and_hold = len(new_events) == 2
                        lead_hold_tail = len(new_events) == 3
                        if (
                            (
                                lead_and_hold
                                or lead_hold_tail
                                and isinstance(new_events[2], PlaySamples)
                            )
                            and isinstance(new_events[0], PlaySamples)
                            and isinstance(new_events[1], PlayHold)
                        ):
                            if lead_hold_tail:
                                new_i = np.concatenate(
                                    [
                                        new_events[0].samples["i"],
                                        [new_events[0].samples["i"][-1]] * 4,
                                        new_events[2].samples["i"],
                                    ]
                                )
                                new_q = np.concatenate(
                                    [
                                        new_events[0].samples["q"],
                                        [new_events[0].samples["q"][-1]] * 4,
                                        new_events[2].samples["q"],
                                    ]
                                )
                            else:
                                new_i = np.concatenate(
                                    [
                                        new_events[0].samples["i"],
                                        [new_events[0].samples["i"][-1]] * 4,
                                    ]
                                )
                                new_q = np.concatenate(
                                    [
                                        new_events[0].samples["q"],
                                        [new_events[0].samples["q"][-1]] * 4,
                                    ]
                                )
                            if len(new_i) > 4096:  # TODO(2K): get via device_type
                                raise LabOneQException(
                                    "SHFQA measure pulse exceeds 4096 samples after compression."
                                )
                            new_samples_signature = SamplesSignature(
                                samples_i=new_i,
                                samples_q=new_q,
                                # No markers for SHFQA measure pulses
                                samples_marker1=None,
                                samples_marker2=None,
                            )
                            sampled_signature.samples = new_samples_signature
                            sampled_signature.hold_start = len(
                                new_events[0].samples["i"]
                            )
                            assert (
                                new_events[1].num_samples >= 12
                            )  # Ensured by previous conditions
                            sampled_signature.hold_length = (
                                new_events[1].num_samples - 4
                            )
                            continue
                        raise LabOneQException(
                            "Unexpected SHFQA long measure pulse: only a single const region is allowed."
                        )

                    if all(
                        not pulse_defs[pulse.pulse].can_compress
                        for pulse in wave_form.pulses
                    ):
                        continue
                    sampled_signature = sampled_signatures[wave_form]
                    compressor_input_samples = {
                        k: getattr(sampled_signature, k)
                        for k in (
                            "samples_i",
                            "samples_q",
                            "samples_marker1",
                            "samples_marker2",
                        )
                        if getattr(sampled_signature, k) is not None
                    }
                    pulse_compr_info = namedtuple(
                        "PulseComprInfo", ["start", "end", "can_compress"]
                    )
                    pulse_compr_infos = [
                        pulse_compr_info(
                            start=pulse.start,
                            end=pulse.start + pulse.length,
                            can_compress=pulse_defs[pulse.pulse].can_compress,
                        )
                        for pulse in wave_form.pulses
                    ]
                    if not CodeGenerator._pulses_compatible_for_compression(
                        pulse_compr_infos
                    ):
                        raise LabOneQException(
                            "overlapping pulses need to either all have can_compress=True or can_compress=False"
                        )
                    compressible_segments = [
                        (pulse.start, pulse.start + pulse.length)
                        for pulse in wave_form.pulses
                        if pulse_defs[pulse.pulse].can_compress
                    ]
                    # remove duplicates, keep order
                    compressible_segments = [*dict.fromkeys(compressible_segments)]
                    new_events = wave_compressor.compress_wave(
                        compressor_input_samples,
                        device_type.min_play_wave,
                        compressible_segments,
                    )
                    pulse_names = [pulse.pulse for pulse in wave_form.pulses]
                    if new_events is None:
                        _logger.info(
                            "Requested to compress pulse(s) %s which has(have) either no, or too short, constant sections. Skipping compression",
                            ",".join(pulse_names),
                        )
                        continue
                    event_replacement[i] = new_events
                    _logger.debug(
                        "Compressing pulse(s) %s using %d PlayWave and %d PlayHold events",
                        ",".join(pulse_names),
                        sum(
                            1 for event in new_events if isinstance(event, PlaySamples)
                        ),
                        sum(1 for event in new_events if isinstance(event, PlayHold)),
                    )
            for idx, new_events in event_replacement.items():
                new_awg_events = CodeGenerator._emit_new_awg_events(
                    event_group[idx],
                    new_events,
                    sampled_signatures=sampled_signatures,
                )
                old_event = event_group[idx]
                event_group[idx : idx + 1] = new_awg_events
                compressed_waveform_signatures.add(old_event.signature.waveform)

        # evict waveforms that have been compressed, and thus replaced with one or more, shorter, waves
        for waveform_signature in compressed_waveform_signatures:
            sampled_signatures.pop(waveform_signature)

    def _gen_seq_c_per_awg(
        self,
        awg_ir: ir_code.SingleAwgIR,
        events: EventList,
        pulse_defs: dict[str, PulseDef],
        user_pulse_params: list[passes.PulseParams],
        feedback_register: FeedbackRegisterAllocation,
        acquisition_type: AcquisitionType | None,
    ):
        awg = awg_ir.awg
        _logger.debug("Generating seqc for awg %d of %s", awg.awg_id, awg.device_id)
        _logger.debug("AWG Object = \n%s", awg)
        awg_compile_info = passes.analyze_awg_ir(awg_ir)
        global_sampling_rate, global_delay = self._calc_global_awg_params(awg)
        global_delay_samples = length_to_samples(global_delay, global_sampling_rate)
        use_command_table = awg.device_type in (DeviceType.HDAWG, DeviceType.SHFSG)

        _logger.debug(
            "Analyzing initialization events for awg %d of %s",
            awg.awg_id,
            awg.device_id,
        )
        sampled_events = AWGSampledEventSequence()
        # Sequencer start
        cut_points = {global_delay_samples}

        # Gather events sorted by time (in samples) in a dict of lists with time as key

        phase_reset_events = analyze_phase_reset_times(
            events, awg.device_id, global_sampling_rate, global_delay
        )
        sampled_events.merge(phase_reset_events)

        loop_events = analyze_loop_times(
            awg, events, global_sampling_rate, global_delay
        )
        loop_events.sort()
        sampled_events.merge(loop_events)
        prng_setup = analyze_prng_times(events, global_sampling_rate, global_delay)
        sampled_events.merge(prng_setup)

        trigger_events = analyze_trigger_events(events, awg.signals, loop_events)
        sampled_events.merge(trigger_events)

        for signal_obj in awg.signals:
            set_oscillator_events = analyze_set_oscillator_times(
                events=events,
                signal_obj=signal_obj,
                global_delay=global_delay,
            )
            sampled_events.merge(set_oscillator_events)

        cut_points = set(sampled_events.sequence) | {
            length_to_samples(global_delay, global_sampling_rate)
        }
        play_wave_size_hint, play_zero_size_hint = self.waveform_size_hints(
            awg.device_type
        )
        awg_code_output = codegen_rs.generate_code_for_awg(
            ob=awg_ir,
            signals=self._ir.signals,
            cut_points=cut_points,
            play_wave_size_hint=play_wave_size_hint,
            play_zero_size_hint=play_zero_size_hint,
            amplitude_resolution_range=self.amplitude_resolution_range(),
            use_amplitude_increment=self._settings.USE_AMPLITUDE_INCREMENT,
            phase_resolution_range=self.phase_resolution_range(),
            global_delay_samples=global_delay_samples,
        )
        awg_events = compat_rs.transform_rs_events_to_awg_events(
            awg_code_output.awg_events
        )
        for event in awg_events:
            if event.type in [
                AWGEventType.RESET_PRECOMPENSATION_FILTERS,
                AWGEventType.PPC_SWEEP_STEP_START,
                AWGEventType.PPC_SWEEP_STEP_END,
            ]:
                sampled_events.add(event.start, event)

        for signal_obj in awg.signals:
            if signal_obj.signal_type != "integration":
                continue
            events_acquire = [
                e
                for e in awg_events
                if e.type == AWGEventType.ACQUIRE
                and e.params["signal_id"] == signal_obj.id
            ]
            self._integration_weights[awg.key][signal_obj.id] = (
                calculate_integration_weights(
                    events_acquire,
                    signal_obj,
                    pulse_defs,
                    pulse_params=user_pulse_params,
                )
            )
            for e in events_acquire:
                sampled_events.add(e.start, e)
        # Perform the actual downsampling of integration weights using a common factor for the AWG
        common_downsampling_factor = max(
            (
                iw.downsampling_factor or 1
                for iws in self._integration_weights[awg.key].values()
                for iw in iws
            ),
            default=1,
        )
        if common_downsampling_factor > 1:
            for iws in self._integration_weights[awg.key].values():
                for iw in iws:
                    iw.samples_i = scipy.signal.resample(
                        iw.samples_i, len(iw.samples_i) // common_downsampling_factor
                    )
                    iw.samples_q = scipy.signal.resample(
                        iw.samples_q, len(iw.samples_q) // common_downsampling_factor
                    )
                    iw.downsampling_factor = common_downsampling_factor

        # Signatures of this AWG
        waveforms: dict[
            WaveformSignature, list[tuple[tuple[str], SampledWaveformSignature | None]]
        ] = defaultdict(list)
        if awg.signal_type == AWGSignalType.MULTI:
            _logger.debug("Multi signal %s", awg)
            assert len(awg.signals) > 0
            mixer_type = awg.signals[0].mixer_type
            for signal_obj in awg.signals:
                if signal_obj.signal_type != "integration":
                    _logger.debug(
                        "Non-integration signal in multi signal: %s", signal_obj.id
                    )

            signals = {s.id: s for s in awg.signals if s.signal_type != "integration"}
            virtual_signal = tuple(signals)
            events_play_wave = AWGSampledEventSequence()
            # Original insertion order kept for the event types below
            for event in awg_events:
                if event.type == AWGEventType.PLAY_WAVE and event.params[
                    "signal_ids"
                ] == set(signals.keys()):
                    events_play_wave.add(event.start, event)
                elif event.type == AWGEventType.MATCH:
                    events_play_wave.add(event.start, event)
                elif event.type == AWGEventType.INIT_AMPLITUDE_REGISTER:
                    events_play_wave.add(event.start, event)
                elif (
                    event.type == AWGEventType.CHANGE_OSCILLATOR_PHASE
                    and event.params["signal_id"] in signals
                ):
                    events_play_wave.add(event.start, event)
            sampled_signatures = self._sample_pulses(
                virtual_signal,
                events_play_wave,
                pulse_defs=pulse_defs,
                sampling_rate=awg.sampling_rate,
                signal_type="iq",
                device_type=awg.device_type,
                multi_iq_signal=True,
                mixer_type=mixer_type,
                params=user_pulse_params,
            )

            sampled_events.merge(events_play_wave)

            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                pulse_defs=pulse_defs,
                device_type=awg.device_type,
            )
            for waveform, sampled_signature in sampled_signatures.items():
                waveforms[waveform].append(
                    (
                        virtual_signal,
                        sampled_signature,
                    )
                )
        elif awg.signal_type != AWGSignalType.DOUBLE:
            # AWGSignalType.IQ / AWGSignalType.SINGLE
            for signal_obj in awg.signals:
                events_play_wave = AWGSampledEventSequence()
                # Original insertion order kept for the event types below
                for event in awg_events:
                    if event.type == AWGEventType.PLAY_WAVE and event.params[
                        "signal_ids"
                    ] == {signal_obj.id}:
                        events_play_wave.add(event.start, event)
                    elif event.type == AWGEventType.MATCH:
                        events_play_wave.add(event.start, event)
                    elif event.type == AWGEventType.INIT_AMPLITUDE_REGISTER:
                        events_play_wave.add(event.start, event)
                    elif (
                        event.type == AWGEventType.CHANGE_OSCILLATOR_PHASE
                        and event.params["signal_id"] == signal_obj.id
                    ):
                        events_play_wave.add(event.start, event)
                virtual_signal = (signal_obj.id,)
                sampled_signatures = self._sample_pulses(
                    virtual_signal,
                    events_play_wave,
                    pulse_defs=pulse_defs,
                    sampling_rate=awg.sampling_rate,
                    signal_type=signal_obj.signal_type,
                    device_type=awg.device_type,
                    mixer_type=signal_obj.mixer_type,
                    params=user_pulse_params,
                )
                sampled_events.merge(events_play_wave)

                self._compress_waves(
                    sampled_events=sampled_events,
                    sampled_signatures=sampled_signatures,
                    pulse_defs=pulse_defs,
                    device_type=awg.device_type,
                )
                for waveform, sampled_signature in sampled_signatures.items():
                    waveforms[waveform].append(
                        (
                            virtual_signal,
                            sampled_signature,
                        )
                    )
        else:  # awg.signal_type == AWGSignalType.DOUBLE
            assert len(awg.signals) == 2
            signal_a, signal_b = awg.signals
            virtual_signal = (signal_a.id, signal_b.id)
            events_play_wave = AWGSampledEventSequence()
            # Original insertion order kept for the event types below
            for event in awg_events:
                if event.type == AWGEventType.PLAY_WAVE and event.params[
                    "signal_ids"
                ] == {signal_a.id, signal_b.id}:
                    events_play_wave.add(event.start, event)
                elif event.type == AWGEventType.MATCH:
                    events_play_wave.add(event.start, event)
                elif event.type == AWGEventType.INIT_AMPLITUDE_REGISTER:
                    events_play_wave.add(event.start, event)
                elif (
                    event.type == AWGEventType.CHANGE_OSCILLATOR_PHASE
                    and event.params["signal_id"] in {signal_a.id, signal_b.id}
                ):
                    events_play_wave.add(event.start, event)
            sampled_signatures = self._sample_pulses(
                virtual_signal,
                events_play_wave,
                pulse_defs=pulse_defs,
                sampling_rate=awg.sampling_rate,
                signal_type=signal_a.signal_type,
                device_type=awg.device_type,
                mixer_type=signal_a.mixer_type,
                params=user_pulse_params,
            )

            sampled_events.merge(events_play_wave)

            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                pulse_defs=pulse_defs,
                device_type=awg.device_type,
            )
            for waveform, sampled_signature in sampled_signatures.items():
                waveforms[waveform].append(
                    (
                        virtual_signal,
                        sampled_signature,
                    )
                )

        self.post_process_sampled_events(awg, sampled_events)
        # Handle sampled waveforms for output
        signature_strings = {}
        sampled_waveforms: list[_SampledWaveform] = []
        signature_infos = []
        for wf, sampled in waveforms.items():
            for signals, sampled_signature in sampled:
                if not sampled_signature:
                    continue
                sig_string = signature_strings.setdefault(wf, wf.signature_string())
                sampled_waveforms.append(
                    _SampledWaveform(
                        signals=signals,
                        signature=sampled_signature,
                        signature_string=sig_string,
                    )
                )
                has_marker1 = sampled_signature.samples_marker1 is not None
                has_marker2 = sampled_signature.samples_marker2 is not None
                signature_infos.append(
                    (sig_string, wf.length, (has_marker1, has_marker2))
                )
        self._sampled_waveforms[awg.key] = sampled_waveforms
        _logger.debug(
            "** Start processing events for awg %d of %s",
            awg.awg_id,
            awg.device_id,
        )
        emit_timing_comments = self._settings.EMIT_TIMING_COMMENTS
        function_defs_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        declarations_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        if emit_timing_comments:
            declarations_generator.add_comment(
                f"{awg.device_type}/{awg.awg_id} global delay {EngNumber(global_delay)} sampling_rate: {EngNumber(global_sampling_rate)}Sa/s "
            )
        has_readout_feedback = awg_compile_info.has_readout_feedback
        if has_readout_feedback:
            declarations_generator.add_variable_declaration("current_seq_step", 0)
        for siginfo in sorted(signature_infos):
            declarations_generator.add_wave_declaration(
                siginfo[0],
                siginfo[1],
                siginfo[2][0],
                siginfo[2][1],
            )
        command_table_tracker = CommandTableTracker(awg.device_type)
        deferred_function_calls = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        init_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        add_wait_trigger_statements(awg, init_generator, deferred_function_calls)
        seqc_tracker = SeqCTracker(
            init_generator=init_generator,
            deferred_function_calls=deferred_function_calls,
            sampling_rate=global_sampling_rate,
            delay=global_delay,
            device_type=awg.device_type.name,
            signal_type=awg.signal_type.name,
            emit_timing_comments=emit_timing_comments,
            automute_playzeros_min_duration=self._settings.SHF_OUTPUT_MUTE_MIN_DURATION,
            automute_playzeros=any([sig.automute for sig in awg.signals]),
        )
        shfppc_sweeper_config_tracker = SHFPPCSweeperConfigTracker(
            awg_compile_info.ppc_device, awg_compile_info.ppc_channel
        )
        handler = SampledEventHandler(
            seqc_tracker=seqc_tracker,
            command_table_tracker=command_table_tracker,
            shfppc_sweeper_config_tracker=shfppc_sweeper_config_tracker,
            function_defs_generator=function_defs_generator,
            declarations_generator=declarations_generator,
            wave_indices=WaveIndexTracker(),
            feedback_register=feedback_register,
            feedback_connections=self._feedback_connections,
            feedback_register_config=self._feedback_register_config[awg.key],
            qa_signal_by_handle=self._qa_signals_by_handle,
            feedback_register_layout=self._feedback_register_layout,
            awg=awg,
            device_type=awg.device_type,
            channels=awg.signals[0].channels,
            use_command_table=use_command_table,
            emit_timing_comments=emit_timing_comments,
            use_current_sequencer_step=has_readout_feedback,
            acquisition_type=acquisition_type,
        )

        handler.handle_sampled_events(sampled_events)
        for res_usage in handler.resource_usage():
            if (
                self._max_resource_usage is None
                or res_usage.usage > self._max_resource_usage.usage
            ):
                self._max_resource_usage = res_usage

        seq_c_generators: list[SeqCGenerator] = []
        while (part := seqc_tracker.pop_loop_stack_generators()) is not None:
            seq_c_generators = part + seq_c_generators
        _logger.debug(
            "***  collected generators, seq_c_generators: %s", seq_c_generators
        )

        main_generator = merge_generators(seq_c_generators, True)

        seq_c_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        if function_defs_generator.num_statements() > 0:
            seq_c_generator.append_statements_from(function_defs_generator)
            seq_c_generator.add_comment("=== END-OF-FUNCTION-DEFS ===")
        seq_c_generator.append_statements_from(declarations_generator)
        seq_c_generator.append_statements_from(main_generator)

        seq_c_text = seq_c_generator.generate_seq_c()

        for line in seq_c_text.splitlines():
            _logger.debug(line)

        awg_key = AwgKey(awg.device_id, awg.awg_id)

        self._src[awg_key] = SeqCProgram(src=seq_c_text)
        self._wave_indices_all[awg_key] = {"value": handler.wave_indices.wave_indices()}
        if use_command_table:
            self._command_tables[awg_key] = {
                "ct": handler.command_table_tracker.command_table()
            }
            self._parameter_phase_increment_map[awg_key] = (
                handler.command_table_tracker.parameter_phase_increment_map()
            )

        if shfppc_sweeper_config_tracker.has_sweep_commands():
            shfppc_config = shfppc_sweeper_config_tracker.finish()
            self._shfppc_sweep_configs[awg_key] = shfppc_config

    def waveform_size_hints(self, device: DeviceType):
        settings = self._settings

        def sanitize_min_playwave_hint(n: int, multiple: int) -> int:
            if n % multiple != 0:
                n2 = round_min_playwave_hint(n, multiple)
                _logger.warning(
                    f"Compiler setting `MIN_PLAYWAVE_HINT`={n} for device {device.name} is not multiple of {multiple} and is rounded to {n2}."
                )
                return n2
            return n

        min_pw_hint = None
        min_pz_hint = None
        if device == DeviceType.HDAWG:
            min_pw_hint = settings.HDAWG_MIN_PLAYWAVE_HINT
            min_pz_hint = settings.HDAWG_MIN_PLAYZERO_HINT
        elif device == DeviceType.UHFQA:
            min_pw_hint = settings.UHFQA_MIN_PLAYWAVE_HINT
            min_pz_hint = settings.UHFQA_MIN_PLAYZERO_HINT
        elif device == DeviceType.SHFQA:
            min_pw_hint = settings.SHFQA_MIN_PLAYWAVE_HINT
            min_pz_hint = settings.SHFQA_MIN_PLAYZERO_HINT
        elif device == DeviceType.SHFSG:
            min_pw_hint = settings.SHFSG_MIN_PLAYWAVE_HINT
            min_pz_hint = settings.SHFSG_MIN_PLAYZERO_HINT
        if min_pw_hint is not None and min_pz_hint is not None:
            return (
                sanitize_min_playwave_hint(min_pw_hint, device.sample_multiple),
                min_pz_hint,
            )

    def _sample_pulses(
        self,
        signals: tuple[str],
        interval_events: AWGSampledEventSequence,
        pulse_defs: dict[str, PulseDef],
        sampling_rate: float,
        signal_type: str,
        device_type: DeviceType,
        mixer_type: MixerType | None,
        params: list[passes.PulseParams],
        multi_iq_signal=False,
    ):
        params = params or {}
        sampled_signatures: dict[
            WaveformSignature, SampledWaveformSignature | None
        ] = {}
        # To produce deterministic SeqC, we use a dict for signatures
        # as an easy insertion-ordered set.
        signatures: dict[PlaybackSignature, type[None]] = {}
        for interval_event_list in interval_events.sequence.values():
            for interval_event in interval_event_list:
                if interval_event.type != AWGEventType.PLAY_WAVE:
                    continue
                signature = interval_event.signature
                _logger.debug("Signature found %s in %s", signature, interval_event)
                if any(p.pulse for p in signature.waveform.pulses):
                    signatures[signature] = None
                else:
                    sampled_signatures[signature.waveform] = None
        _logger.debug("Signatures: %s", signatures.keys())

        needs_conjugate = device_type == DeviceType.SHFSG
        for signature in signatures:
            length = signature.waveform.length
            if length % device_type.sample_multiple != 0:
                raise Exception(
                    f"Length of signature {signature.waveform.signature_string()} is not divisible by {device_type.sample_multiple}, which it needs to be for {device_type.value}"
                )

            signature_pulse_map: dict[str, PulseWaveformMap] = {}
            samples_i = np.zeros(length)
            samples_q = np.zeros(length)
            samples_marker1 = np.zeros(length, dtype=np.int16)
            samples_marker2 = np.zeros(length, dtype=np.int16)
            has_marker1 = False
            has_marker2 = False

            has_q = False

            assert signature.waveform is not None
            assert signature.waveform.pulses is not None

            for pulse_part in signature.waveform.pulses:
                _logger.debug(" Sampling pulse part %s", pulse_part)
                pulse_def = pulse_defs[pulse_part.pulse]

                if pulse_def.amplitude is None:
                    pulse_def = copy.deepcopy(pulse_def)
                    pulse_def.amplitude = 1.0
                _logger.debug(" Pulse def: %s", pulse_def)

                sampling_signal_type = signal_type
                if pulse_part.channel is not None:
                    sampling_signal_type = "single"
                if multi_iq_signal:
                    sampling_signal_type = "iq"
                if pulse_part.sub_channel is not None:
                    sampling_signal_type = "iq"

                amplitude = pulse_def.amplitude
                if pulse_part.amplitude is not None:
                    amplitude *= pulse_part.amplitude

                used_oscillator_frequency = pulse_part.oscillator_frequency

                _logger.debug(
                    " Sampling pulse %s using oscillator frequency %s",
                    pulse_part,
                    used_oscillator_frequency,
                )

                iq_phase = 0.0

                if pulse_part.phase is not None:
                    # According to "LabOne Q Software: Signal, channel and oscillator concept" REQ 1.3
                    iq_phase += pulse_part.phase

                iq_phase = normalize_phase(iq_phase)

                samples = pulse_def.samples
                if pulse_part.id_pulse_params is not None:
                    pulse_params = params[pulse_part.id_pulse_params]
                    params_pulse_pulse = pulse_params.pulse_params
                    params_pulse_play = pulse_params.play_params
                    params_pulse_combined = pulse_params.combined()
                else:
                    params_pulse_pulse = None
                    params_pulse_play = None
                    params_pulse_combined = None
                sampled_pulse = sample_pulse(
                    signal_type=sampling_signal_type,
                    sampling_rate=sampling_rate,
                    amplitude=amplitude,
                    length=pulse_part.length / sampling_rate,
                    pulse_function=pulse_def.function,
                    modulation_frequency=used_oscillator_frequency,
                    phase=iq_phase,
                    samples=samples,
                    mixer_type=mixer_type,
                    pulse_parameters=params_pulse_combined,
                    markers=[m for m in pulse_part.markers]
                    if pulse_part.markers
                    else None,
                    pulse_defs=pulse_defs,
                )
                sampled_pulse = SampledWaveformSignature(
                    samples=SamplesSignature(
                        samples_i=sampled_pulse["samples_i"],
                        samples_q=sampled_pulse["samples_q"],
                        samples_marker1=sampled_pulse.get("samples_marker1"),
                        samples_marker2=sampled_pulse.get("samples_marker2"),
                    )
                )
                verify_amplitude_no_clipping(
                    samples_i=sampled_pulse.samples_i,
                    samples_q=sampled_pulse.samples_q,
                    pulse_id=pulse_def.uid,
                    mixer_type=mixer_type,
                    signals=signals,
                )

                if sampled_pulse.samples_q is not None and len(
                    sampled_pulse.samples_i
                ) != len(sampled_pulse.samples_q):
                    _logger.warning(
                        "Expected samples_q and samples_i to be of equal length"
                    )
                len_i = len(sampled_pulse.samples_i)
                if not len_i == pulse_part.length and samples is None:
                    num_samples = length_to_samples(pulse_def.length, sampling_rate)
                    _logger.warning(
                        "Pulse part %s: Expected %d samples but got %d; length = %f num samples=%d length in samples=%d",
                        repr(pulse_part),
                        pulse_part.length,
                        len_i,
                        pulse_def.length,
                        num_samples,
                        pulse_def.length * sampling_rate,
                    )
                    raise Exception("Len mismatch")

                if (
                    pulse_part.channel == 0
                    and not multi_iq_signal
                    and not device_type == DeviceType.SHFQA
                ):
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse.samples_i, samples_i
                    )
                    has_q = True
                elif (
                    pulse_part.channel == 1
                    and not multi_iq_signal
                    and not device_type == DeviceType.SHFQA
                ):
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse.samples_i, samples_q
                    )
                    has_q = True
                else:
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse.samples_i, samples_i
                    )
                    if sampled_pulse.samples_q is not None:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse.samples_q,
                            samples_q,
                        )
                        has_q = True

                # RF case
                if pulse_part.channel is not None and device_type == DeviceType.HDAWG:
                    if (
                        sampled_pulse.samples_marker1 is not None
                        and pulse_part.channel == 0
                    ):
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse.samples_marker1,
                            samples_marker1,
                        )
                        has_marker1 = True

                    # map user facing marker1 to "internal" marker 2
                    if (
                        sampled_pulse.samples_marker1 is not None
                        and pulse_part.channel == 1
                    ):
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse.samples_marker1,
                            samples_marker2,
                        )
                        has_marker2 = True

                    if (
                        sampled_pulse.samples_marker2 is not None
                        and pulse_part.channel == 1
                    ):
                        raise LabOneQException(
                            f"Marker 2 not supported on channel 1 of multiplexed RF signal {signals}. Please use marker 1"
                        )
                else:
                    if sampled_pulse.samples_marker1 is not None:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse.samples_marker1,
                            samples_marker1,
                        )
                        has_marker1 = True

                    if sampled_pulse.samples_marker2 is not None:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse.samples_marker2,
                            samples_marker2,
                        )
                        has_marker2 = True

                pm = signature_pulse_map.get(pulse_def.uid)
                if pm is None:
                    pm = PulseWaveformMap(
                        sampling_rate=sampling_rate,
                        length_samples=pulse_part.length,
                        signal_type=sampling_signal_type,
                        mixer_type=mixer_type,
                    )
                    signature_pulse_map[pulse_def.uid] = pm
                pulse_amplitude = pulse_def.amplitude
                amplitude_multiplier = (
                    amplitude / pulse_amplitude if pulse_amplitude else 0.0
                )
                pm.instances.append(
                    PulseInstance(
                        offset_samples=pulse_part.start,
                        amplitude=amplitude_multiplier,
                        length=pulse_part.length,
                        modulation_frequency=used_oscillator_frequency,
                        iq_phase=iq_phase,
                        channel=pulse_part.channel,
                        needs_conjugate=needs_conjugate,
                        play_pulse_parameters={}
                        if params_pulse_play is None
                        else params_pulse_play,
                        pulse_pulse_parameters={}
                        if params_pulse_pulse is None
                        else params_pulse_pulse,
                        has_marker1=has_marker1,
                        has_marker2=has_marker2,
                        can_compress=pulse_def.can_compress,
                    )
                )
            if len(samples_i) != length:
                _logger.warning(
                    "Num samples does not match. Expected %d but got %d",
                    length,
                    len(samples_i),
                )
            if has_q:
                if needs_conjugate:
                    samples_q = -samples_q
                samples_q = samples_q
            sampled_signature = SampledWaveformSignature(
                samples=SamplesSignature(
                    samples_i=samples_i,
                    samples_q=samples_q,
                    samples_marker1=samples_marker1 if has_marker1 else None,
                    samples_marker2=samples_marker2 if has_marker2 else None,
                ),
                pulse_map=signature_pulse_map,
            )
            sampled_signatures[signature.waveform] = sampled_signature
            verify_amplitude_no_clipping(
                samples_i=sampled_signature.samples_i,
                samples_q=sampled_signature.samples_q,
                pulse_id=None,
                mixer_type=mixer_type,
                signals=signals,
            )

        return sampled_signatures

    def waves(self) -> dict[str, CodegenWaveform]:
        return self._waves

    def requires_long_readout(self) -> dict[str, list[str]]:
        return self._requires_long_readout

    def src(self) -> dict[AwgKey, SeqCProgram]:
        return self._src

    def wave_indices(self):
        return self._wave_indices_all

    def command_tables(self):
        return self._command_tables

    def pulse_map(self) -> dict[str, PulseMapEntry]:
        return self._pulse_map

    def parameter_phase_increment_map(
        self,
    ) -> dict[AwgKey, dict[str, list[int | Literal[COMPLEX_USAGE]]]]:
        return self._parameter_phase_increment_map

    def integration_times(self) -> IntegrationTimes:
        return self._integration_times

    def signal_delays(self) -> SignalDelays:
        return self._signal_delays

    def feedback_register_config(self) -> dict[AwgKey, FeedbackRegisterConfig]:
        # convert defaultdict to dict
        return dict(self._feedback_register_config)

    def feedback_connections(self) -> dict[str, FeedbackConnection]:
        return self._feedback_connections

    def shfppc_sweep_configs(self) -> dict[AwgKey, SHFPPCSweeperConfig]:
        return self._shfppc_sweep_configs

    @staticmethod
    def stencil_samples(start, source, target):
        source_start = 0
        target_start = start
        if start < 0:
            source_start = -start
            target_start = 0

        source_end = len(source)
        if source_end - source_start + target_start > len(target):
            source_end = source_start + len(target) - target_start
        target_end = target_start + source_end - source_start
        if target_end >= 0 and target_start < len(target):
            to_insert = source[source_start:source_end]
            target[target_start:target_end] += to_insert
        else:
            _logger.debug(
                "Not inserting: %d:%d , %d:%d",
                source_start,
                source_end,
                target_start,
                target_end,
            )

    def phase_resolution_range(self) -> int:
        if self._settings.PHASE_RESOLUTION_BITS < 0:
            # disable quantization
            return 0
        return 1 << self._settings.PHASE_RESOLUTION_BITS

    def amplitude_resolution_range(self) -> int:
        if self._settings.AMPLITUDE_RESOLUTION_BITS < 0:
            # disable quantization
            return 0
        return 1 << self._settings.AMPLITUDE_RESOLUTION_BITS

    @staticmethod
    def post_process_sampled_events(
        awg: AWGInfo, sampled_events: AWGSampledEventSequence
    ):
        if awg.device_type == DeviceType.SHFQA:
            has_acquire = sampled_events.has_matching_event(
                lambda ev: ev.type == AWGEventType.ACQUIRE
            )

            for sampled_event_list in sampled_events.sequence.values():
                start = None
                end = None
                play: list[AWGEvent] = []
                acquire: list[AWGEvent] = []
                for x in sampled_event_list:
                    if x.type == AWGEventType.PLAY_WAVE:
                        play.append(x)
                        start = x.start
                        end = x.end if end is None else max(end, x.end)
                    elif x.type == AWGEventType.ACQUIRE:
                        start = x.start
                        acquire.append(x)
                        end = x.end if end is None else max(end, x.end)
                if len(play) > 0 and len(acquire) == 0 and has_acquire:
                    raise Exception("Play and acquire must happen at the same time")
                if len(play) > 0 or len(acquire) > 0:
                    assert end is not None
                    end = (
                        round(end / DeviceType.SHFQA.sample_multiple)
                        * DeviceType.SHFQA.sample_multiple
                    )
                    qa_event = AWGEvent(
                        type=AWGEventType.QA_EVENT,
                        start=start,
                        end=end,
                        params={
                            "acquire_events": acquire,
                            "play_events": play,
                        },
                    )

                    for to_delete in play + acquire:
                        sampled_event_list.remove(to_delete)
                    sampled_event_list.append(qa_event)
