# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
import math
from numbers import Number
from typing import Any, Literal, cast
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
from laboneq._rust.codegenerator import (
    WaveIndexTracker,
    SeqCGenerator,
    SeqCTracker,
    seqc_generator_from_device_and_signal_type as seqc_generator_from_device_and_signal_type_str,
    merge_generators,
    SampledWaveform,
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
    PulseMapEntry,
    COMPLEX_USAGE,
    WeightInfo,
)
from laboneq.compiler.seqc import compat_rs
from .waveform_sampler import WaveformSampler

_logger = logging.getLogger(__name__)


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
        play_wave_ids: list[str] = event.params.get("play_wave_id", [])
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
        self._sampled_waveforms: dict[AwgKey, list[SampledWaveform]] = {}
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
            waveform_sampler=WaveformSampler(
                pulse_defs=pulse_defs, pulse_def_params=user_pulse_params
            ),
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
        # NOTE: Add playwave related events after the rest to mimic the original event insertion order
        # Can be removed once the event insertion order is not important anymore (all events generated in Rust)
        for event in awg_events:
            if event.type in [
                AWGEventType.PLAY_WAVE,
                AWGEventType.MATCH,
                AWGEventType.INIT_AMPLITUDE_REGISTER,
                AWGEventType.CHANGE_OSCILLATOR_PHASE,
                AWGEventType.PLAY_HOLD,
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
        self.post_process_sampled_events(awg, sampled_events)
        signature_infos = [
            (
                wave_declaration.signature_string,
                wave_declaration.length,
                (wave_declaration.has_marker1, wave_declaration.has_marker2),
            )
            for wave_declaration in awg_code_output.wave_declarations
        ]
        self._sampled_waveforms[awg.key] = awg_code_output.sampled_waveforms
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

        for siginfo in sorted(list(signature_infos)):
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
