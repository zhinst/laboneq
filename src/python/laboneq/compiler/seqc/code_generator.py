# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import groupby
import math
from numbers import Number
from typing import Any, NamedTuple, Literal

import numpy as np
from numpy import typing as npt
from engineering_notation import EngNumber
import scipy.signal

from laboneq.compiler.common.compiler_settings import (
    TINYSAMPLE,
    CompilerSettings,
    round_min_playwave_hint,
)
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.shfppc_sweeper_config import SHFPPCSweeperConfig
from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout
from laboneq.compiler.ir.ir import IRTree, PulseDefIR
from laboneq.compiler.seqc import passes, ir as ir_code
from laboneq.compiler.seqc.linker import AwgWeights, SeqCGenOutput
from laboneq._utils import ensure_list
from laboneq.compiler.seqc.analyze_events import (
    analyze_acquire_times,
    analyze_init_times,
    analyze_loop_times,
    analyze_phase_reset_times,
    analyze_precomp_reset_times,
    analyze_set_oscillator_times,
    analyze_trigger_events,
    analyze_prng_times,
    analyze_ppc_sweep_events,
)
from laboneq.compiler.seqc.analyze_playback import analyze_play_wave_times
from laboneq.compiler.seqc.command_table_tracker import (
    CommandTableTracker,
    EntryLimitExceededError,
)
from laboneq.compiler.seqc.feedback_register_allocator import (
    FeedbackRegisterAllocator,
)
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.seqc import ir_to_event_list
from laboneq.compiler.seqc.measurement_calculator import (
    IntegrationTimes,
    MeasurementCalculator,
    SignalDelays,
)
from laboneq.compiler.seqc.sampled_event_handler import SampledEventHandler
from laboneq.compiler.seqc.seqc_generator import (
    SeqCGenerator,
    merge_generators,
)
from laboneq.compiler.seqc.seqc_tracker import SeqCTracker
from laboneq.compiler.seqc.shfppc_sweeper_config_tracker import (
    SHFPPCSweeperConfigTracker,
)
from laboneq.compiler.seqc.signatures import (
    PlaybackSignature,
    SamplesSignature,
    WaveformSignature,
    FrozenWaveformSignature,
)
from laboneq.compiler.seqc.utils import normalize_phase
from laboneq.compiler.seqc.wave_compressor import (
    PlayHold,
    PlaySamples,
    WaveCompressor,
)
from laboneq.compiler.seqc.wave_index_tracker import WaveIndexTracker
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.event_list.event_type import EventList, EventType
from laboneq.compiler.common.pulse_parameters import decode_pulse_parameters
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    combine_pulse_parameters,
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


_logger = logging.getLogger(__name__)


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


def stable_hash(integration_weight):
    """Stable hashing function for generating names for readout weights"""
    return hashlib.sha1(
        integration_weight["samples_i"].tobytes()
        + integration_weight["samples_q"].tobytes()
    )


@dataclass
class IntegrationWeight:
    basename: str
    samples_i: npt.ArrayLike
    samples_q: npt.ArrayLike
    downsampling_factor: int | None


def calculate_integration_weights(
    acquire_events: AWGSampledEventSequence,
    signal_obj: SignalObj,
    pulse_defs: dict[str, PulseDef],
) -> list[IntegrationWeight]:
    integration_weights_by_pulse: dict[str, IntegrationWeight] = {}
    signal_id = signal_obj.id

    nr_of_weights_per_event = None
    for event_list in acquire_events.sequence.values():
        for event in event_list:
            _logger.debug("For weights, look at %s", event)

            assert event.params.get("signal_id", signal_id) == signal_id
            play_wave_ids: list[str] = ensure_list(event.params.get("play_wave_id"))
            n = len(play_wave_ids)
            play_pars = ensure_list(
                event.params.get("play_pulse_parameters") or [None] * n
            )
            pulse_pars = ensure_list(
                event.params.get("pulse_pulse_parameters") or [None] * n
            )

            oscillator_frequency = event.params.get("oscillator_frequency", 0.0)
            if isinstance(oscillator_frequency, list):
                [oscillator_frequency, *rest] = oscillator_frequency
                assert all(oscillator_frequency == f for f in rest)

            filtered_pulses: list[tuple[str, Any, Any]] = [
                (play_wave_id, play_par, pulse_par)
                for play_wave_id, play_par, pulse_par in zip(
                    play_wave_ids, play_pars, pulse_pars
                )
                if play_wave_id is not None
                and (pulse_def := pulse_defs.get(play_wave_id)) is not None
                # Not a real pulse, just a placeholder for the length - skip
                and (pulse_def.samples is not None or pulse_def.function is not None)
            ]

            assert n == len(play_pars) == len(pulse_pars)
            if nr_of_weights_per_event is None:
                nr_of_weights_per_event = n
            else:
                assert n == nr_of_weights_per_event
                if any(
                    w not in integration_weights_by_pulse for w, _, _ in filtered_pulses
                ):
                    # Event uses different pulse UIDs than earlier event
                    raise LabOneQException(
                        f"Using different integration kernels on a single signal"
                        f" ({signal_id}) is unsupported. Weights: {play_wave_ids} vs"
                        f" {list(integration_weights_by_pulse.keys())}"
                    )

            for play_wave_id, play_par, pulse_par in filtered_pulses:
                pulse_def = pulse_defs[play_wave_id]
                samples = pulse_def.samples
                pulse_amplitude: Number = pulse_def.amplitude
                assert isinstance(pulse_amplitude, Number)

                play_amplitude = event.params.get("amplitude", 1.0)
                if isinstance(play_amplitude, list):
                    [play_amplitude, *rest] = play_amplitude
                    assert all(a == play_amplitude for a in rest)
                amplitude = pulse_amplitude * play_amplitude

                length = pulse_def.length
                if length is None:
                    length = len(samples) / signal_obj.awg.sampling_rate

                pulse_parameters = combine_pulse_parameters(pulse_par, None, play_par)
                pulse_parameters = decode_pulse_parameters(pulse_parameters)
                iw_samples = sample_pulse(
                    signal_type="iq",
                    sampling_rate=signal_obj.awg.sampling_rate,
                    length=length,
                    amplitude=amplitude,
                    pulse_function=pulse_def.function,
                    modulation_frequency=oscillator_frequency,
                    samples=samples,
                    mixer_type=signal_obj.mixer_type,
                    pulse_parameters=pulse_parameters,
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
                    iw_samples,
                    pulse_def.uid,
                    signal_obj.mixer_type,
                    signal_obj.id,
                )

                # 128-bit hash as waveform name
                digest = stable_hash(iw_samples).hexdigest()[:32]
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

    return list(integration_weights_by_pulse.values())


class CodeGenerator(ICodeGenerator):
    USE_ZSYNC_TRIGGER = True

    DELAY_FIRST_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_OTHER_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_UHFQA = 128 / DeviceType.UHFQA.sampling_rate

    # This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
    # to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
    SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10

    _measurement_calculator = MeasurementCalculator

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
        self._src: dict[AwgKey, dict[str, str]] = {}
        self._wave_indices_all: dict[AwgKey, dict] = {}
        self._waves: dict[str, CodegenWaveform] = {}
        self._requires_long_readout: dict[str, list[str]] = defaultdict(list)
        self._command_tables: dict[AwgKey, dict[str, Any]] = {}
        self._pulse_map: dict[str, PulseMapEntry] = {}
        self._parameter_phase_increment_map: dict[
            AwgKey, dict[str, list[int | Literal[COMPLEX_USAGE]]]
        ] = {}
        self._sampled_signatures: dict[str, dict[FrozenWaveformSignature, dict]] = {}
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
        self._feedback_register_allocator: FeedbackRegisterAllocator | None = None
        self._total_execution_time: float | None = None

        self.EMIT_TIMING_COMMENTS = self._settings.EMIT_TIMING_COMMENTS

    def collect_acquisition_data(
        self, events_or_ir: EventList | IRTree
    ) -> tuple[
        dict[str, SignalObj],
        list[dict[str, str]],
        FeedbackRegisterAllocator,
        IntegrationTimes,
        SignalDelays,
        float,
    ]:
        """Collect data from the event list/IR used to generate seqc code."""
        # This is a temporary hack to obtain some acquisiton related
        # data from the event list/IR used to generate seqc code. As we
        # progress in converting these passes to use the IR instead of the
        # event list, the calls in this method will be replaced piece by
        # piece, until the whole code is converted to rust.

        # TODO: Remove 'events' API in favor of IR, requires fixing many tests
        if isinstance(events_or_ir, IRTree):
            assert events_or_ir.root is not None
            total_execution_time = events_or_ir.root.length * TINYSAMPLE
            measurement_info = passes.collect_measurement_info(
                events_or_ir.root, self._signals
            )
            integration_times = measurement_info.integration_times
            signal_delays = measurement_info.signal_delays
            feedback_register_allocator = measurement_info.feedback_register_allocator
            qa_signals_by_handle = measurement_info.qa_signals_by_handle
            simultaneous_acquires = measurement_info.simultaneous_acquires
        else:
            # This branch should only be encountered via tests
            # TODO: Remove this code and the called functions when the event list is gone
            events = events_or_ir
            total_execution_time = (
                max([e.get("time", 0) for e in events]) if len(events) > 0 else None
            )
            (
                integration_times,
                signal_delays,
            ) = self._measurement_calculator.calculate_integration_times(
                self._signals, events
            )
            feedback_register_allocator = FeedbackRegisterAllocator(self._signals)
            feedback_register_allocator.set_feedback_paths_from_events(events)
            qa_signals_by_handle = self._collect_qa_signals_by_handle(
                events, self._signals
            )
            simultaneous_acquires = self._gen_acquire_map(events)

        return (
            qa_signals_by_handle,
            simultaneous_acquires,
            feedback_register_allocator,
            integration_times,
            signal_delays,
            total_execution_time,
        )

    def generate_code(self):
        passes.inline_sections_in_branch(self._ir)

        (
            qa_signals_by_handle,
            simultaneous_acquires,
            feedback_register_allocator,
            integration_times,
            signal_delays,
            total_execution_time,
        ) = self.collect_acquisition_data(self._ir)

        self.gen_seq_c(
            self._ir,
            pulse_defs={p.uid: p for p in self._ir.pulse_defs},
            qa_signals_by_handle=qa_signals_by_handle,
            simultaneous_acquires=simultaneous_acquires,
            feedback_register_allocator=feedback_register_allocator,
            integration_times=integration_times,
            signal_delays=signal_delays,
            total_execution_time=total_execution_time,
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
        )
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

            signals_to_process = awg.signals
            virtual_signal_id = None
            if awg.signal_type == AWGSignalType.MULTI:
                iq_signal_ids = []
                signals_to_process = []
                for signal_obj in awg.signals:
                    if signal_obj.signal_type != "integration":
                        _logger.debug(
                            "Non-integration signal in multi signal: %s", signal_obj
                        )
                        iq_signal_ids.append(signal_obj.id)
                    else:
                        signals_to_process.append(signal_obj)
                virtual_signal_id = "_".join(iq_signal_ids)
            elif awg.signal_type == AWGSignalType.DOUBLE:
                virtual_signal_id = (
                    signals_to_process[0].id + "_" + signals_to_process[1].id
                )

            if virtual_signal_id is None:
                for signal_obj in signals_to_process:
                    if signal_obj.id in self._sampled_signatures:
                        for (
                            signature_key,
                            sampled_signature,
                        ) in self._sampled_signatures[signal_obj.id].items():
                            if not sampled_signature:
                                continue
                            sig_string = signature_key.signature_string()
                            if signal_obj.awg.device_type.supports_binary_waves:
                                if awg.signal_type == AWGSignalType.SINGLE:
                                    self._save_wave_bin(
                                        sampled_signature["samples_i"],
                                        sampled_signature["signature_pulse_map"],
                                        sig_string,
                                        "",
                                    )
                                    if "samples_marker1" in sampled_signature:
                                        self._save_wave_bin(
                                            sampled_signature["samples_marker1"],
                                            sampled_signature["signature_pulse_map"],
                                            sig_string,
                                            "_marker1",
                                        )
                                    if "samples_marker2" in sampled_signature:
                                        self._save_wave_bin(
                                            sampled_signature["samples_marker2"],
                                            sampled_signature["signature_pulse_map"],
                                            sig_string,
                                            "_marker2",
                                        )
                                else:
                                    self._save_wave_bin(
                                        sampled_signature["samples_i"],
                                        sampled_signature["signature_pulse_map"],
                                        sig_string,
                                        "_i",
                                    )
                                    if "samples_q" in sampled_signature:
                                        self._save_wave_bin(
                                            sampled_signature["samples_q"],
                                            sampled_signature["signature_pulse_map"],
                                            sig_string,
                                            "_q",
                                        )
                                    if "samples_marker1" in sampled_signature:
                                        self._save_wave_bin(
                                            sampled_signature["samples_marker1"],
                                            sampled_signature["signature_pulse_map"],
                                            sig_string,
                                            "_marker1",
                                        )
                                    if "samples_marker2" in sampled_signature:
                                        self._save_wave_bin(
                                            sampled_signature["samples_marker2"],
                                            sampled_signature["signature_pulse_map"],
                                            sig_string,
                                            "_marker2",
                                        )

                            elif signal_obj.awg.device_type.supports_complex_waves:
                                self._save_wave_bin(
                                    CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                                    * (
                                        sampled_signature["samples_i"]
                                        - 1j * sampled_signature["samples_q"]
                                    ),
                                    sampled_signature["signature_pulse_map"],
                                    sig_string,
                                    "",
                                    device_id=awg.device_id,
                                    signal_id=signal_obj.id,
                                    hold_start=sampled_signature.get("hold_start"),
                                    hold_length=sampled_signature.get("hold_length"),
                                )
                            else:
                                raise RuntimeError(
                                    f"Device type {signal_obj.awg.device_type} has invalid supported waves config."
                                )
            else:
                signal_obj = awg.signals[0]
                for signature_key, sampled_signature in self._sampled_signatures[
                    virtual_signal_id
                ].items():
                    if not sampled_signature:
                        continue
                    sig_string = signature_key.signature_string()
                    if signal_obj.awg.device_type.supports_binary_waves:
                        self._save_wave_bin(
                            sampled_signature["samples_i"],
                            sampled_signature["signature_pulse_map"],
                            sig_string,
                            "_i",
                        )
                        if "samples_q" in sampled_signature:
                            self._save_wave_bin(
                                sampled_signature["samples_q"],
                                sampled_signature["signature_pulse_map"],
                                sig_string,
                                "_q",
                            )
                        if "samples_marker1" in sampled_signature:
                            self._save_wave_bin(
                                sampled_signature["samples_marker1"],
                                sampled_signature["signature_pulse_map"],
                                sig_string,
                                "_marker1",
                            )
                        if "samples_marker2" in sampled_signature:
                            self._save_wave_bin(
                                sampled_signature["samples_marker2"],
                                sampled_signature["signature_pulse_map"],
                                sig_string,
                                "_marker2",
                            )

                    else:
                        raise RuntimeError(
                            f"Device type {signal_obj.awg.device_type} has invalid supported waves config."
                        )

        # check that there are no duplicate filenames in the wave pool (QCSW-1079)
        waves = sorted(
            [(filename, wave.samples) for filename, wave in self._waves.items()],
            key=lambda w: w[0],
        )
        for _, group in groupby(waves, key=lambda w: w[0]):
            group = list(group)
            assert all(np.all(group[0][1] == g[1]) for g in group[1:])

    def _unroll_acquisition_in_prng_loops(self, events: EventList):
        @dataclass
        class LoopInfo:
            name: str
            start: float
            iterations: int = 0

        prng_loop_sections = {
            e["section_name"]
            for e in events
            if e["event_type"] == EventType.DRAW_PRNG_SAMPLE
        }

        loop_stack: list[LoopInfo] = []
        open_acquisitions: list[EventList] = [[]]

        for e in events:
            if e["event_type"] == EventType.ACQUIRE_START:
                open_acquisitions[-1].append(e)
            elif (
                e["event_type"] == EventType.LOOP_START
                and (section := e["section_name"]) in prng_loop_sections
            ):
                # entering new nested loop, push onto stack
                loop_stack.append(
                    LoopInfo(name=section, start=e["time"], iterations=e["iterations"])
                )
                open_acquisitions.append([])
            elif (
                e["event_type"] == EventType.LOOP_END
                and (section := e["section_name"]) in prng_loop_sections
            ):
                # exiting loop, pop from stack and replicate any open acquisitions N times
                loop = loop_stack.pop()
                end = e["time"]
                assert loop.name == section

                length = (end - loop.start) / loop.iterations

                unrolled_acquires = []
                for original_acquire in open_acquisitions.pop():
                    extra_acquires = [
                        original_acquire.copy() for _ in range(loop.iterations)
                    ]
                    for i in range(1, loop.iterations):
                        extra_acquires[i]["time"] += i * length
                    unrolled_acquires.extend(extra_acquires)

                open_acquisitions[-1].extend(unrolled_acquires)

        [unrolled_acquires] = open_acquisitions
        return unrolled_acquires

    def _gen_acquire_map(self, events_or_ir: EventList) -> list[dict[str, str]]:
        # timestamp -> map[signal -> handle]
        simultaneous_acquires: dict[int, dict[str, str]] = {}

        events = self._unroll_acquisition_in_prng_loops(events_or_ir)

        for e in events:
            time_events = simultaneous_acquires.setdefault(e["time"], {})
            time_events[e["signal"]] = e["acquire_handle"]

        return list(simultaneous_acquires.values())

    @staticmethod
    def _collect_qa_signals_by_handle(events: EventList, signals: dict[str, SignalObj]):
        # TODO: Remove once the event list is gone for good
        qa_signals_by_handle: dict[str, SignalObj] = {}
        for event in events:
            if event["event_type"] != EventType.ACQUIRE_START:
                continue
            handle = event["acquire_handle"]
            if handle is None:
                # Some unit tests omit the handle
                # this cannot be used for feedback anyway, so ignore
                continue
            signal_obj = signals[event["signal"]]

            if handle not in qa_signals_by_handle:
                qa_signals_by_handle[handle] = signal_obj
            else:
                assert qa_signals_by_handle[handle] == signal_obj
        return qa_signals_by_handle

    def gen_seq_c(
        self,
        events_or_ir: EventList | IRTree,
        pulse_defs: dict[str, PulseDefIR],
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
        self._feedback_register_allocator = feedback_register_allocator
        self._total_execution_time = total_execution_time

        if isinstance(events_or_ir, IRTree):
            # The pass must happen after delay calculation as it relies on section information.
            # TODO: Fix that this pass can happen as early as possible.
            passes.inline_sections(self._ir.root)
            osc_params = passes.calculate_oscillator_parameters(events_or_ir.root)
            ir_tree = passes.fanout_awgs(events_or_ir, self._awgs.values())
            ir_awgs = {awg.awg.key: awg for awg in ir_tree.root.children}
            events_per_awg = ir_to_event_list.event_list_per_awg(
                ir_tree, self._settings, osc_params
            )
        else:
            # This branch should only be encountered via tests
            events_per_awg = {awg: events_or_ir for awg in self._awgs}
            ir_awgs = {
                awg_key: ir_code.SingleAwgIR(awg=awg)
                for awg_key, awg in self._awgs.items()
            }

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

        self.sort_signals()
        self._integration_weights.clear()

        for awg_key, ir_awg in sorted(
            ir_awgs.items(),
            key=lambda item: item[0],
        ):
            self._gen_seq_c_per_awg(
                ir_awg,
                events_per_awg.get(awg_key, []),
                pulse_defs,
            )

        tgt_feedback_regs = self._feedback_register_allocator.target_feedback_registers
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
        old_event, new_events
    ) -> tuple[list[AWGEvent], dict[str, str]]:
        new_awg_events = []
        pulse_name_mapping = {}
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

                for pulse in playback_signature.waveform.pulses:
                    pulse_name_mapping[pulse.pulse] = (
                        pulse.pulse + f"_compr_{new_event.label}"
                    )

                playback_signature.waveform.length = new_length
                playback_signature.waveform.pulses = None
                playback_signature.waveform.samples = SamplesSignature(
                    "compr", new_event.samples
                )

                new_awg_events.append(
                    AWGEvent(
                        type=AWGEventType.PLAY_WAVE,
                        start=time,
                        end=time + new_length,
                        params=new_params,
                    )
                )

                time += new_length
        return new_awg_events, pulse_name_mapping

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
        sampled_signatures: dict[WaveformSignature, dict],
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
                        for id in ensure_list(event.params["play_wave_id"])
                    ):
                        _logger.warning(
                            "Compression for integration pulses is not supported. %s, for which compression has been requested, will not be compressed.",
                            event.params["play_wave_id"],
                        )
                if event.type == AWGEventType.PLAY_WAVE:
                    playback_signature: PlaybackSignature = event.params[
                        "playback_signature"
                    ]
                    wave_form = playback_signature.waveform
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
                        orig_i = sampled_signature["samples_i"]
                        if len(orig_i) <= 4096:  # TODO(2K): get via device_type
                            # Measure pulse is fitting into memory, no additional processing needed
                            continue
                        orig_q = sampled_signature["samples_q"]
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
                            sampled_signature["samples_i"] = new_i
                            sampled_signature["samples_q"] = new_q
                            sampled_signature["hold_start"] = len(
                                new_events[0].samples["i"]
                            )
                            assert (
                                new_events[1].num_samples >= 12
                            )  # Ensured by previous conditions
                            sampled_signature["hold_length"] = (
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
                    sample_dict = {
                        k: sampled_signature[k]
                        for k in (
                            "samples_i",
                            "samples_q",
                            "samples_marker1",
                            "samples_marker2",
                        )
                        if k in sampled_signature
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
                        sample_dict, device_type.min_play_wave, compressible_segments
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
                new_awg_events, pulse_name_mapping = CodeGenerator._emit_new_awg_events(
                    event_group[idx], new_events
                )

                old_event = event_group[idx]
                event_group[idx : idx + 1] = new_awg_events

                playback_signature: PlaybackSignature = old_event.params[
                    "playback_signature"
                ]
                old_waveform = playback_signature.waveform
                for new_awg_event in new_awg_events:
                    if new_awg_event.type == AWGEventType.PLAY_WAVE:
                        new_playback_signature: PlaybackSignature = (
                            new_awg_event.params["playback_signature"]
                        )
                        new_waveform = new_playback_signature.waveform
                        new_length = len(
                            next(iter(new_waveform.samples.samples_map.values()))
                        )

                        old_sampled_signature = sampled_signatures[old_waveform]
                        new_sampled_signature = copy.deepcopy(old_sampled_signature)

                        # save old waveform such that we can clean the sampled signatures later on
                        compressed_waveform_signatures.add(old_waveform)

                        new_sampled_signature = (
                            new_sampled_signature | new_waveform.samples.samples_map
                        )
                        new_signature_pulse_map = new_sampled_signature[
                            "signature_pulse_map"
                        ]

                        # update 3 things in samples signatures
                        #   - name of pulses that have been compressed
                        #   - length_samples entry in signature pulse map
                        #   - length entry in the instances stored in the signature pulse map

                        for old_name, new_name in pulse_name_mapping.items():
                            new_signature_pulse_map[new_name] = (
                                new_signature_pulse_map.pop(old_name)
                            )

                        for sp_map in new_signature_pulse_map.values():
                            sp_map.length_samples = new_length
                            for signature in sp_map.instances:
                                signature.length = new_length

                        sampled_signatures[new_waveform] = new_sampled_signature

        # evict waveforms that have been compressed, and thus replaced with one or more, shorter, waves
        for waveform_signature in compressed_waveform_signatures:
            sampled_signatures.pop(waveform_signature)

    def _gen_seq_c_per_awg(
        self,
        awg_ir: ir_code.SingleAwgIR,
        events: EventList,
        pulse_defs: dict[str, PulseDef],
    ):
        awg = awg_ir.awg
        empty_intervals = passes.collect_empty_intervals(awg_ir)
        # TODO: Ensure AWG does not get any irrelevant events, it will result in extra looping.
        #       There exists e.g Section, loop, subsection starts and ends which are
        #       effectively always dropped.
        function_defs_generator = SeqCGenerator()
        declarations_generator = SeqCGenerator()
        _logger.debug("Generating seqc for awg %d of %s", awg.awg_id, awg.device_id)
        _logger.debug("AWG Object = \n%s", awg)
        sampled_events = AWGSampledEventSequence()
        # Signatures of this AWG
        waveform_signatures: dict[str, dict[FrozenWaveformSignature, dict]] = {}

        global_sampling_rate, global_delay = self._calc_global_awg_params(awg)
        if self.EMIT_TIMING_COMMENTS:
            declarations_generator.add_comment(
                f"{awg.device_type}/{awg.awg_id} global delay {EngNumber(global_delay)} sampling_rate: {EngNumber(global_sampling_rate)}Sa/s "
            )

        use_command_table = awg.device_type in (DeviceType.HDAWG, DeviceType.SHFSG)
        signal_ids = set(signal.id for signal in awg.signals)
        own_sections = set(
            event["section_name"]
            for event in events
            if event.get("signal") in signal_ids
            or event["event_type"]
            in (
                EventType.PARAMETER_SET,
                EventType.SET_OSCILLATOR_FREQUENCY_START,
            )
        )
        for event in events:
            if (
                event["event_type"] == EventType.SUBSECTION_END
                and event["subsection_name"] in own_sections
            ):
                # by looking at SUBSECTION_END, we'll always walk towards the tree root
                own_sections.add(event["section_name"])
        has_readout_feedback = False
        for event in events:
            if (
                event["event_type"] == EventType.SECTION_START
                and event.get("handle") is not None
                and event["section_name"] in own_sections
            ):
                has_readout_feedback = True
        if has_readout_feedback:
            declarations_generator.add_variable_declaration("current_seq_step", 0)
        _logger.debug(
            "Analyzing initialization events for awg %d of %s",
            awg.awg_id,
            awg.device_id,
        )

        # Gather events sorted by time (in samples) in a dict of lists with time as key
        init_events = analyze_init_times(
            awg.device_id, global_sampling_rate, global_delay
        )
        sampled_events.merge(init_events)

        precomp_reset_events = analyze_precomp_reset_times(
            events, [s.id for s in awg.signals], global_sampling_rate, global_delay
        )
        sampled_events.merge(precomp_reset_events)

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

        ppc_sweep_events = analyze_ppc_sweep_events(events, awg, global_delay)
        sampled_events.merge(ppc_sweep_events)

        for signal_obj in awg.signals:
            set_oscillator_events = analyze_set_oscillator_times(
                events, signal_obj, global_delay
            )
            sampled_events.merge(set_oscillator_events)

            trigger_events = analyze_trigger_events(events, signal_obj, loop_events)
            sampled_events.merge(trigger_events)

            acquire_events = analyze_acquire_times(
                events,
                signal_obj,
                feedback_register_allocator=self._feedback_register_allocator,
            )

            if signal_obj.signal_type == "integration":
                _logger.debug(
                    "Calculating integration weights for signal %s. There are %d acquire events.",
                    signal_obj.id,
                    len(acquire_events.sequence),
                )
                self._integration_weights[awg.key][signal_obj.id] = (
                    calculate_integration_weights(
                        acquire_events, signal_obj, pulse_defs
                    )
                )

            sampled_events.merge(acquire_events)

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

        if awg.signal_type == AWGSignalType.MULTI:
            _logger.debug("Multi signal %s", awg)
            assert len(awg.signals) > 0
            delay = 0.0
            mixer_type = awg.signals[0].mixer_type
            for signal_obj in awg.signals:
                if signal_obj.signal_type != "integration":
                    _logger.debug(
                        "Non-integration signal in multi signal: %s", signal_obj
                    )
                    delay = max(delay, signal_obj.total_delay)

            signals = {s.id: s for s in awg.signals if s.signal_type != "integration"}
            virtual_signal_id = "_".join(signals.keys())

            interval_events = analyze_play_wave_times(
                events=events,
                signals=signals,
                device_type=awg.device_type,
                sampling_rate=awg.sampling_rate,
                delay=delay,
                other_events=sampled_events,
                phase_resolution_range=self.phase_resolution_range(),
                amplitude_resolution_range=self.amplitude_resolution_range(),
                waveform_size_hints=self.waveform_size_hints(awg.device_type),
                use_command_table=use_command_table,
                use_amplitude_increment=self._settings.USE_AMPLITUDE_INCREMENT,
                empty_intervals=empty_intervals,
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=awg.sampling_rate,
                signal_type="iq",
                device_type=awg.device_type,
                multi_iq_signal=True,
                mixer_type=mixer_type,
            )

            sampled_events.merge(interval_events)

            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                pulse_defs=pulse_defs,
                device_type=awg.device_type,
            )
            waveform_signatures[virtual_signal_id] = {
                k.to_frozen(): v for k, v in sampled_signatures.items()
            }
            for sig, sampled in waveform_signatures[virtual_signal_id].items():
                if not sampled:
                    continue
                declarations_generator.add_wave_declaration(
                    awg.device_type,
                    awg.signal_type.value,
                    sig.signature_string(),
                    sig.length,
                    "samples_marker1" in sampled,
                    "samples_marker2" in sampled,
                )

        elif awg.signal_type != AWGSignalType.DOUBLE:
            # AWGSignalType.IQ / AWGSignalType.SINGLE
            for signal_obj in awg.signals:
                if awg.device_type == DeviceType.SHFQA:
                    sub_channel = signal_obj.channels[0]
                else:
                    sub_channel = None
                interval_events = analyze_play_wave_times(
                    events=events,
                    signals={signal_obj.id: signal_obj},
                    device_type=awg.device_type,
                    sampling_rate=awg.sampling_rate,
                    delay=signal_obj.total_delay,
                    other_events=sampled_events,
                    phase_resolution_range=self.phase_resolution_range(),
                    amplitude_resolution_range=self.amplitude_resolution_range(),
                    waveform_size_hints=self.waveform_size_hints(awg.device_type),
                    sub_channel=sub_channel,
                    use_command_table=use_command_table,
                    use_amplitude_increment=self._settings.USE_AMPLITUDE_INCREMENT,
                    empty_intervals=empty_intervals,
                )

                sampled_signatures = self._sample_pulses(
                    signal_obj.id,
                    interval_events,
                    pulse_defs=pulse_defs,
                    sampling_rate=awg.sampling_rate,
                    signal_type=signal_obj.signal_type,
                    device_type=awg.device_type,
                    mixer_type=signal_obj.mixer_type,
                )

                sampled_events.merge(interval_events)

                self._compress_waves(
                    sampled_events=sampled_events,
                    sampled_signatures=sampled_signatures,
                    pulse_defs=pulse_defs,
                    device_type=awg.device_type,
                )
                waveform_signatures[signal_obj.id] = {
                    k.to_frozen(): v for k, v in sampled_signatures.items()
                }
                signature_infos = []
                for sig, sampled in waveform_signatures[signal_obj.id].items():
                    if not sampled:
                        continue
                    has_marker1 = False
                    has_marker2 = False
                    if "samples_marker1" in sampled:
                        has_marker1 = True
                    if "samples_marker2" in sampled:
                        has_marker2 = True
                    signature_infos.append(
                        (sig.signature_string(), sig.length, (has_marker1, has_marker2))
                    )
                for siginfo in sorted(signature_infos):
                    declarations_generator.add_wave_declaration(
                        awg.device_type,
                        signal_obj.signal_type,
                        siginfo[0],
                        siginfo[1],
                        siginfo[2][0],
                        siginfo[2][1],
                    )
        else:  # awg.signal_type == AWGSignalType.DOUBLE
            assert len(awg.signals) == 2
            signal_a, signal_b = awg.signals
            virtual_signal_id = signal_a.id + "_" + signal_b.id
            interval_events = analyze_play_wave_times(
                events=events,
                signals={s.id: s for s in awg.signals},
                device_type=awg.device_type,
                sampling_rate=awg.sampling_rate,
                delay=signal_a.total_delay,
                other_events=sampled_events,
                phase_resolution_range=self.phase_resolution_range(),
                amplitude_resolution_range=self.amplitude_resolution_range(),
                waveform_size_hints=self.waveform_size_hints(awg.device_type),
                use_command_table=False,  # do not set amplitude/oscillator phase via CT
                empty_intervals=empty_intervals,
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=awg.sampling_rate,
                signal_type=signal_a.signal_type,
                device_type=awg.device_type,
                mixer_type=signal_a.mixer_type,
            )

            sampled_events.merge(interval_events)

            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                pulse_defs=pulse_defs,
                device_type=awg.device_type,
            )
            waveform_signatures[virtual_signal_id] = {
                k.to_frozen(): v for k, v in sampled_signatures.items()
            }
            for sig, sampled in waveform_signatures[virtual_signal_id].items():
                if not sampled:
                    continue
                declarations_generator.add_wave_declaration(
                    awg.device_type,
                    awg.signal_type.value,
                    sig.signature_string(),
                    sig.length,
                    "samples_marker1" in sampled,
                    "samples_marker2" in sampled,
                )
        self._sampled_signatures.update(waveform_signatures)
        self.post_process_sampled_events(awg, sampled_events)
        _logger.debug(
            "** Start processing events for awg %d of %s",
            awg.awg_id,
            awg.device_id,
        )
        command_table_tracker = CommandTableTracker(awg.device_type)
        deferred_function_calls = SeqCGenerator()
        init_generator = SeqCGenerator()
        add_wait_trigger_statements(awg, init_generator, deferred_function_calls)
        seqc_tracker = SeqCTracker(
            init_generator=init_generator,
            deferred_function_calls=deferred_function_calls,
            sampling_rate=global_sampling_rate,
            delay=global_delay,
            device_type=awg.device_type,
            emit_timing_comments=self.EMIT_TIMING_COMMENTS,
            logger=_logger,
            automute_playzeros_min_duration=self._settings.SHF_OUTPUT_MUTE_MIN_DURATION,
            automute_playzeros=any([sig.automute for sig in awg.signals]),
        )
        shfppc_sweeper_config_tracker = SHFPPCSweeperConfigTracker()
        handler = SampledEventHandler(
            seqc_tracker=seqc_tracker,
            command_table_tracker=command_table_tracker,
            shfppc_sweeper_config_tracker=shfppc_sweeper_config_tracker,
            function_defs_generator=function_defs_generator,
            declarations_generator=declarations_generator,
            wave_indices=WaveIndexTracker(),
            feedback_connections=self._feedback_connections,
            feedback_register_config=self._feedback_register_config[awg.key],
            qa_signal_by_handle=self._qa_signals_by_handle,
            feedback_register_layout=self._feedback_register_layout,
            awg=awg,
            device_type=awg.device_type,
            channels=awg.signals[0].channels,
            use_command_table=use_command_table,
            emit_timing_comments=self.EMIT_TIMING_COMMENTS,
            use_current_sequencer_step=has_readout_feedback,
        )

        # Modify signatures in place to replace Signature waveforms with frozen waveforms.
        # The frozen waveforms have precalculated signatures strings, which
        # is costly to compute for each pulse.
        frozen_wf_signatures = {
            signature: signature
            for wf in waveform_signatures.values()
            for signature in wf.keys()
        }
        for awg_events in sampled_events.sequence.values():
            for awg_event in awg_events:
                if maybe_signature := awg_event.params.get("playback_signature"):
                    if maybe_signature.waveform is None:
                        continue
                    maybe_signature.waveform = frozen_wf_signatures[
                        maybe_signature.waveform
                    ]
        try:
            sampled_events.sort()
            handler.handle_sampled_events(sampled_events)
        except EntryLimitExceededError as error:
            msg = (
                f"Device '{awg.device_id}', AWG({awg.awg_id}): "
                "Exceeded maximum number of command table entries.\n"
                "Try the following:\n"
                "- Reduce the number of sweep steps\n"
                "- Reduce the number of variations in the pulses that are being played\n"
            )
            raise LabOneQException(f"Compiler error. {msg}") from error

        _logger.debug(
            "***  Finished event processing, loop_stack_generators: %s",
            seqc_tracker.loop_stack_generators,
        )
        seq_c_generators = []
        for part in seqc_tracker.loop_stack_generators:
            seq_c_generators.extend(part)
        _logger.debug(
            "***  collected generators, seq_c_generators: %s", seq_c_generators
        )

        main_generator = merge_generators(seq_c_generators)

        seq_c_generator = SeqCGenerator()
        if function_defs_generator.num_statements() > 0:
            seq_c_generator.append_statements_from(function_defs_generator)
            seq_c_generator.add_comment("=== END-OF-FUNCTION-DEFS ===")
        seq_c_generator.append_statements_from(declarations_generator)
        seq_c_generator.append_statements_from(main_generator)

        seq_c_text = seq_c_generator.generate_seq_c()

        for line in seq_c_text.splitlines():
            _logger.debug(line)

        awg_key = AwgKey(awg.device_id, awg.awg_id)

        self._src[awg_key] = {"text": seq_c_text}
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
        signal_id: str,
        interval_events: AWGSampledEventSequence,
        pulse_defs: dict[str, PulseDef],
        sampling_rate: float,
        signal_type: str,
        device_type: DeviceType,
        mixer_type: MixerType | None,
        multi_iq_signal=False,
    ):
        sampled_signatures: dict[WaveformSignature, dict] = {}
        # To produce deterministic SeqC, we use a dict for signatures
        # as an easy insertion-ordered set.
        signatures: dict[PlaybackSignature, type[None]] = {}
        for interval_event_list in interval_events.sequence.values():
            for interval_event in interval_event_list:
                if interval_event.type != AWGEventType.PLAY_WAVE:
                    continue
                signature: PlaybackSignature = interval_event.params[
                    "playback_signature"
                ]
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

            for pulse_part, (play_pulse_parameters, pulse_pulse_parameters) in zip(
                signature.waveform.pulses, signature.pulse_parameters
            ):
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

                assert not pulse_part.oscillator_phase, (
                    "oscillator phase should have been baked into pulse phase"
                )

                baseband_phase = pulse_part.increment_oscillator_phase
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

                # In case oscillator phase can't be set at runtime (e.g. HW oscillator without
                # phase control from a sequencer), apply oscillator phase on a baseband (iq) signal
                iq_phase += baseband_phase or 0.0

                iq_phase = normalize_phase(iq_phase)

                samples = pulse_def.samples

                decoded_pulse_parameters = decode_pulse_parameters(
                    {}
                    if pulse_part.pulse_parameters is None
                    else {k: v for k, v in pulse_part.pulse_parameters}
                )

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
                    pulse_parameters=decoded_pulse_parameters,
                    markers=[{k: v for k, v in m} for m in pulse_part.markers]
                    if pulse_part.markers
                    else None,
                    pulse_defs=pulse_defs,
                )

                verify_amplitude_no_clipping(
                    sampled_pulse, pulse_def.uid, mixer_type, signal_id
                )

                if "samples_q" in sampled_pulse and len(
                    sampled_pulse["samples_i"]
                ) != len(sampled_pulse["samples_q"]):
                    _logger.warning(
                        "Expected samples_q and samples_i to be of equal length"
                    )
                len_i = len(sampled_pulse["samples_i"])
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
                        pulse_part.start, sampled_pulse["samples_i"], samples_i
                    )
                    has_q = True
                elif (
                    pulse_part.channel == 1
                    and not multi_iq_signal
                    and not device_type == DeviceType.SHFQA
                ):
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse["samples_i"], samples_q
                    )
                    has_q = True
                else:
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse["samples_i"], samples_i
                    )
                    if "samples_q" in sampled_pulse:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse["samples_q"],
                            samples_q,
                        )
                        has_q = True

                # RF case
                if pulse_part.channel is not None and device_type == DeviceType.HDAWG:
                    if "samples_marker1" in sampled_pulse and pulse_part.channel == 0:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse["samples_marker1"],
                            samples_marker1,
                        )
                        has_marker1 = True

                    # map user facing marker1 to "internal" marker 2
                    if "samples_marker1" in sampled_pulse and pulse_part.channel == 1:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse["samples_marker1"],
                            samples_marker2,
                        )
                        has_marker2 = True

                    if "samples_marker2" in sampled_pulse and pulse_part.channel == 1:
                        raise LabOneQException(
                            f"Marker 2 not supported on channel 1 of multiplexed RF signal {signal_id}. Please use marker 1"
                        )
                else:
                    if "samples_marker1" in sampled_pulse:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse["samples_marker1"],
                            samples_marker1,
                        )
                        has_marker1 = True

                    if "samples_marker2" in sampled_pulse:
                        self.stencil_samples(
                            pulse_part.start,
                            sampled_pulse["samples_marker2"],
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
                        play_pulse_parameters=None
                        if play_pulse_parameters is None
                        else {k: v for k, v in play_pulse_parameters},
                        pulse_pulse_parameters=None
                        if pulse_pulse_parameters is None
                        else {k: v for k, v in pulse_pulse_parameters},
                        has_marker1=has_marker1,
                        has_marker2=has_marker2,
                        can_compress=pulse_def.can_compress,
                    )
                )

            sampled_pulse_obj = {
                "signature_pulse_map": signature_pulse_map,
                "samples_i": samples_i,
            }
            if len(samples_i) != length:
                _logger.warning(
                    "Num samples does not match. Expected %d but got %d",
                    length,
                    len(samples_i),
                )
            if has_q:
                if needs_conjugate:
                    samples_q = -samples_q
                sampled_pulse_obj["samples_q"] = samples_q

            if has_marker1:
                sampled_pulse_obj["samples_marker1"] = samples_marker1
            if has_marker2:
                sampled_pulse_obj["samples_marker2"] = samples_marker2

            sampled_signatures[signature.waveform] = sampled_pulse_obj

            verify_amplitude_no_clipping(
                sampled_pulse_obj,
                None,
                mixer_type,
                signal_id,
            )

        return sampled_signatures

    def waves(self) -> dict[str, CodegenWaveform]:
        return self._waves

    def requires_long_readout(self) -> dict[str, list[str]]:
        return self._requires_long_readout

    def src(self):
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
                acquisition_types = set()
                feedback_register: None | int = None
                play: list[AWGEvent] = []
                acquire: list[AWGEvent] = []
                for x in sampled_event_list:
                    if "signal_id" in x.params and x.type != AWGEventType.ACQUIRE:
                        play.append(x)
                        start = x.start
                        end = x.end if end is None else max(end, x.end)
                    elif x.type == AWGEventType.ACQUIRE:
                        start = x.start
                        if "acquisition_type" in x.params:
                            acquisition_types.update(x.params["acquisition_type"])
                        this_feedback_register = x.params.get("feedback_register")
                        if this_feedback_register is not None:
                            if (
                                feedback_register is not None
                                and feedback_register != this_feedback_register
                            ):
                                raise LabOneQException(
                                    "Conflicting feedback register allocation detected, please contact development."
                                )
                            feedback_register = this_feedback_register
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
                            "acquisition_type": list(acquisition_types),
                            "acquire_handles": [
                                a.params["acquire_handles"][0] for a in acquire
                            ],
                            "feedback_register": feedback_register,
                        },
                    )

                    for to_delete in play + acquire:
                        sampled_event_list.remove(to_delete)
                    sampled_event_list.append(qa_event)
