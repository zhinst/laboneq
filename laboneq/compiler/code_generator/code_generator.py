# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict, namedtuple
from itertools import groupby
from numbers import Number
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
from engineering_notation import EngNumber

from laboneq._utils import ensure_list
from laboneq.compiler.code_generator.analyze_events import (
    analyze_acquire_times,
    analyze_init_times,
    analyze_loop_times,
    analyze_phase_reset_times,
    analyze_precomp_reset_times,
    analyze_set_oscillator_times,
    analyze_trigger_events,
    analyze_prng_setup_times,
)
from laboneq.compiler.code_generator.analyze_playback import analyze_play_wave_times
from laboneq.compiler.code_generator.command_table_tracker import (
    CommandTableTracker,
    EntryLimitExceededError,
)
from laboneq.compiler.code_generator.feedback_register_allocator import (
    FeedbackRegisterAllocator,
)
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.code_generator.ir_to_event_list import generate_event_list_from_ir
from laboneq.compiler.code_generator.measurement_calculator import (
    IntegrationTimes,
    MeasurementCalculator,
    SignalDelays,
)
from laboneq.compiler.code_generator.sampled_event_handler import SampledEventHandler
from laboneq.compiler.code_generator.seq_c_generator import (
    SeqCGenerator,
    merge_generators,
)
from laboneq.compiler.code_generator.seqc_tracker import SeqCTracker
from laboneq.compiler.code_generator.signatures import (
    PlaybackSignature,
    SamplesSignature,
    WaveformSignature,
)
from laboneq.compiler.code_generator.utils import normalize_phase
from laboneq.compiler.code_generator.wave_compressor import (
    PlayHold,
    PlaySamples,
    WaveCompressor,
)
from laboneq.compiler.code_generator.wave_index_tracker import WaveIndexTracker
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.common.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.compiler_settings import (
    CompilerSettings,
    round_min_playwave_hint,
)
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventList, EventType
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.pulse_parameters import decode_pulse_parameters
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.compiler.workflow.compiler_output import CodegenOutput
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    combine_pulse_parameters,
    length_to_samples,
    sample_pulse,
    verify_amplitude_no_clipping,
)
from laboneq.data.compilation_job import ParameterInfo, PulseDef
from laboneq.data.scheduled_experiment import (
    PulseInstance,
    PulseMapEntry,
    PulseWaveformMap,
)

_logger = logging.getLogger(__name__)


def add_wait_trigger_statements(
    awg: AWGInfo,
    init_generator: SeqCGenerator,
    deferred_function_calls: SeqCGenerator,
):
    if awg.trigger_mode == TriggerMode.DIO_TRIGGER:
        if awg.awg_number == 0:
            init_generator.add_function_call_statement("setDIO", ["0"])
            init_generator.add_function_call_statement("wait", ["2000000"])
            init_generator.add_function_call_statement("playZero", ["512"])
            if awg.reference_clock_source != "internal":
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
                deferred_function_calls.add_function_call_statement("waitWave")
        else:
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_other_awg_samples = str(
                round(awg.sampling_rate * CodeGenerator.DELAY_OTHER_AWG / 16) * 16
            )
            if int(delay_other_awg_samples) > 0:
                deferred_function_calls.add_function_call_statement(
                    "playZero", [delay_other_awg_samples]
                )
                deferred_function_calls.add_function_call_statement("waitWave")

    elif awg.trigger_mode == TriggerMode.DIO_WAIT:
        init_generator.add_variable_declaration("dio", "0xffffffff")
        body = SeqCGenerator()
        body.add_function_call_statement("getDIO", args=None, assign_to="dio")
        init_generator.add_do_while("dio & 0x0001", body)
        init_generator.add_function_call_statement("waitDIOTrigger")
        delay_uhfqa_samples = str(
            round(awg.sampling_rate * CodeGenerator.DELAY_UHFQA / 8) * 8
        )
        if int(delay_uhfqa_samples) > 0:
            init_generator.add_function_call_statement(
                "playZero", [delay_uhfqa_samples]
            )
            init_generator.add_function_call_statement("waitWave")

    elif awg.trigger_mode == TriggerMode.INTERNAL_TRIGGER_WAIT:
        init_generator.add_function_call_statement("waitDigTrigger", ["1"])

    else:
        if CodeGenerator.USE_ZSYNC_TRIGGER and awg.device_type.supports_zsync:
            init_generator.add_function_call_statement("waitZSyncTrigger")
        else:
            init_generator.add_function_call_statement("waitDIOTrigger")


def setup_sine_phase(
    awg: AWGInfo,
    command_table_tracker: CommandTableTracker,
    init_generator: SeqCGenerator,
    use_ct: bool,
):
    if awg.device_type != DeviceType.HDAWG or not use_ct:
        return

    index = command_table_tracker.create_entry(
        PlaybackSignature(
            waveform=None,
            hw_oscillator=None,
            pulse_parameters=(),
            state=None,
            set_phase=0,
            increment_phase=None,
        ),
        None,
    )
    init_generator.add_command_table_execution(index)


def calculate_integration_weights(
    acquire_events: AWGSampledEventSequence, signal_obj, pulse_defs
):
    integration_weights = {}
    signal_id = signal_obj.id
    nr_of_weights_per_event = None
    for event_list in acquire_events.sequence.values():
        for event in event_list:
            _logger.debug("For weights, look at %s", event)

            assert event.params.get("signal_id", signal_id) == signal_id
            play_wave_ids = ensure_list(event.params.get("play_wave_id"))
            n = len(play_wave_ids)
            play_pars = ensure_list(
                event.params.get("play_pulse_parameters") or [None] * n
            )
            pulse_pars = ensure_list(
                event.params.get("pulse_pulse_parameters") or [None] * n
            )

            filtered_pulses = [
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
                if any(w not in integration_weights for w, _, _ in filtered_pulses):
                    # Event uses different pulse UIDs than earlier event
                    raise LabOneQException(
                        f"Using different integration kernels on a single signal"
                        f" ({signal_id}) is unsupported. Weights: {play_wave_ids} vs"
                        f" {list(integration_weights.keys())}"
                    )
            for play_wave_id, play_par, pulse_par in filtered_pulses:
                pulse_def = pulse_defs[play_wave_id]
                samples = pulse_def.samples
                pulse_amplitude: Number | ParameterInfo = pulse_def.amplitude
                if pulse_amplitude is None:
                    pulse_amplitude = 1.0
                elif isinstance(pulse_amplitude, ParameterInfo):
                    pulse_amplitude = (
                        event.params.get(pulse_def.amplitude.uid) or pulse_amplitude
                    )
                assert isinstance(pulse_amplitude, Number)

                length = pulse_def.length
                if length is None:
                    length = len(samples) / signal_obj.awg.sampling_rate

                _logger.debug(
                    "Sampling integration weights for %s with modulation_frequency %s",
                    signal_obj.id,
                    signal_obj.oscillator_frequency,
                )

                pulse_parameters = combine_pulse_parameters(pulse_par, None, play_par)
                pulse_parameters = decode_pulse_parameters(pulse_parameters)
                integration_weight = sample_pulse(
                    signal_type="iq",
                    sampling_rate=signal_obj.awg.sampling_rate,
                    length=length,
                    amplitude=pulse_amplitude,
                    pulse_function=pulse_def.function,
                    modulation_frequency=signal_obj.oscillator_frequency,
                    samples=samples,
                    mixer_type=signal_obj.mixer_type,
                    pulse_parameters=pulse_parameters,
                )

                verify_amplitude_no_clipping(
                    integration_weight,
                    pulse_def.uid,
                    signal_obj.mixer_type,
                    signal_obj.id,
                )

                integration_weight["basename"] = (
                    f"{signal_obj.awg.device_id}"
                    f"_{signal_obj.awg.awg_number}"
                    f"_{min(signal_obj.channels)}"
                    f"_{play_wave_id}"
                )
                if existing_weight := integration_weights.get(play_wave_id):
                    for key in ["samples_i", "samples_q"]:
                        if np.any(existing_weight[key] != integration_weight[key]):
                            # Kernel differs even though pulse ID is the same
                            # (e.g. affected by pulse parameters)
                            raise LabOneQException(
                                f"Using different integration kernels on a single signal"
                                f" ({signal_id}) is unsupported."
                            )
                integration_weights[play_wave_id] = integration_weight

    return integration_weights


class CodeGenerator:
    USE_ZSYNC_TRIGGER = True

    DELAY_FIRST_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_OTHER_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_UHFQA = 140 / DeviceType.HDAWG.sampling_rate

    # This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
    # to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
    SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10

    _measurement_calculator = MeasurementCalculator

    def __init__(self, ir=None, settings: CompilerSettings | dict | None = None):
        if settings is not None:
            if isinstance(settings, CompilerSettings):
                self._settings = settings
            else:
                self._settings = CompilerSettings(**settings)
        else:
            self._settings = CompilerSettings()

        self._ir = ir
        self._signals: dict[str, SignalObj] = {}
        self._code = {}
        self._src: dict[AwgKey, dict[str, str]] = {}
        self._wave_indices_all: dict[AwgKey, dict] = {}
        self._waves: dict[str, Any] = {}
        self._command_tables: dict[AwgKey, dict[str, Any]] = {}
        self._pulse_map: Dict[str, PulseMapEntry] = {}
        self._sampled_signatures: Dict[str, Dict[WaveformSignature, Dict]] = {}
        self._awgs: Dict[AwgKey, AWGInfo] = {}
        self._events_in_samples = {}
        self._integration_times: IntegrationTimes = None
        self._signal_delays: SignalDelays = None
        self._integration_weights = None
        self._simultaneous_acquires: Dict[float, Dict[str, str]] = None
        self._feedback_register_config: Dict[
            AwgKey, FeedbackRegisterConfig
        ] = defaultdict(FeedbackRegisterConfig)
        self._feedback_connections: Dict[str, FeedbackConnection] = {}
        self._feedback_register_allocator: FeedbackRegisterAllocator = None
        self._total_execution_time = None

        self.EMIT_TIMING_COMMENTS = self._settings.EMIT_TIMING_COMMENTS

    def generate_code(self, signal_objs: list[SignalObj]):
        event_list = generate_event_list_from_ir(
            self._ir, self._settings, expand_loops=False, max_events=float("inf")
        )
        for signal_obj in signal_objs:
            self.add_signal(signal_obj)
        self.gen_acquire_map(event_list)
        self.gen_seq_c(
            event_list,
            {p.uid: p for p in self._ir.pulse_defs},
        )
        self.gen_waves()

    def fill_output(self):
        return CodegenOutput(
            feedback_connections=self.feedback_connections(),
            signal_delays=self.signal_delays(),
            integration_weights=self.integration_weights(),
            integration_times=self.integration_times(),
            simultaneous_acquires=self.simultaneous_acquires(),
            src=self.src(),
            total_execution_time=self.total_execution_time(),
            waves=self.waves(),
            wave_indices=self.wave_indices(),
            command_tables=self.command_tables(),
            pulse_map=self.pulse_map(),
            feedback_register_configurations=self.feedback_register_config(),
        )

    def integration_weights(self):
        return self._integration_weights

    def simultaneous_acquires(self) -> Dict[float, Dict[str, str]]:
        return self._simultaneous_acquires

    def total_execution_time(self):
        return self._total_execution_time

    def _add_signal_to_awg(self, signal_obj: SignalObj):
        awg_key = signal_obj.awg.key
        if awg_key not in self._awgs:
            self._awgs[awg_key] = copy.deepcopy(signal_obj.awg)
        self._awgs[awg_key].signals.append(signal_obj)

    def add_signal(self, signal: SignalObj):
        signal_obj = copy.deepcopy(signal)
        signal_obj.pulses = []
        _logger.debug(signal_obj)
        self._signals[signal.id] = signal_obj
        self._add_signal_to_awg(signal_obj)

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
        self, samples, signature_pulse_map, sig_string: str, suffix: str
    ):
        filename = sig_string + suffix + ".wave"
        wave = {"filename": filename, "samples": samples}
        assert filename not in self._waves or np.allclose(
            self._waves[filename]["samples"], wave["samples"]
        )
        self._waves[filename] = wave
        self._append_to_pulse_map(signature_pulse_map, sig_string)

    def gen_waves(self):
        for awg in self._awgs.values():
            # Handle integration weights separately
            for signal_obj in awg.signals:
                assert self._integration_weights is not None
                if signal_obj.id in self._integration_weights:
                    signal_weights = self._integration_weights[signal_obj.id]
                    for weight in signal_weights.values():
                        basename = weight["basename"]
                        if signal_obj.awg.device_type.supports_complex_waves:
                            self._save_wave_bin(
                                CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                                * (weight["samples_i"] - 1j * weight["samples_q"]),
                                None,
                                basename,
                                "",
                            )
                        else:
                            self._save_wave_bin(
                                weight["samples_i"], None, basename, "_i"
                            )
                            self._save_wave_bin(
                                weight["samples_q"], None, basename, "_q"
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
            [(filename, wave["samples"]) for filename, wave in self._waves.items()],
            key=lambda w: w[0],
        )
        for _, group in groupby(waves, key=lambda w: w[0]):
            group = list(group)
            assert all(np.all(group[0][1] == g[1]) for g in group[1:])

    def gen_acquire_map(self, events: EventList):
        # timestamp -> map[signal -> handle]
        self._simultaneous_acquires: Dict[float, Dict[str, str]] = {}
        for e in (e for e in events if e["event_type"] == "ACQUIRE_START"):
            time_events = self._simultaneous_acquires.setdefault(e["time"], {})
            time_events[e["signal"]] = e["acquire_handle"]

    def gen_seq_c(self, events: List[Any], pulse_defs: Dict[str, PulseDef]):
        signal_info_map = {
            signal_id: {"id": s.id, "delay_signal": s.delay_signal}
            for signal_id, s in self._signals.items()
        }

        for k, s in signal_info_map.items():
            signal_obj = self._signals[k]
            s["sampling_rate"] = signal_obj.awg.sampling_rate
            s["awg_number"] = signal_obj.awg.awg_number
            s["device_id"] = signal_obj.awg.device_id
            s["device_type"] = signal_obj.awg.device_type

        (
            self._integration_times,
            self._signal_delays,
        ) = self._measurement_calculator.calculate_integration_times(
            signal_info_map, events
        )

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

        self._total_execution_time = events[-1].get("time") if len(events) > 0 else None
        self.sort_signals()
        self._integration_weights = {}

        self._feedback_register_allocator = FeedbackRegisterAllocator(
            self._signals, events
        )

        for _, awg in sorted(
            self._awgs.items(),
            key=lambda item: item[0].device_id + str(item[0].awg_number),
        ):
            self._gen_seq_c_per_awg(awg, events, pulse_defs)

        for (
            awg,
            target_fb_register,
        ) in self._feedback_register_allocator.target_feedback_registers.items():
            self._feedback_register_config[
                awg
            ].target_feedback_register = target_fb_register

    @staticmethod
    def _calc_global_awg_params(awg: AWGInfo) -> Tuple[float, float]:
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
                    f"Delay {relevant_delay} s = {round(relevant_delay*signal_obj.awg.sampling_rate)} samples on signal {signal_obj.id} is not compatible with the sample multiple of {signal_obj.awg.device_type.sample_multiple} on {signal_obj.awg.device_type}"
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
            global_delay > 0
            and global_delay
            < awg.device_type.min_play_wave / awg.device_type.sampling_rate
        ):
            global_delay = 0

        _logger.debug(
            "Global delay for %s awg %s: %ss, calculated from all_relevant_delays %s",
            awg.device_id,
            awg.awg_number,
            EngNumber(global_delay or 0),
            [(s, EngNumber(d)) for s, d in all_relevant_delays.items()],
        )

        return global_sampling_rate, global_delay

    def _emit_new_awg_events(self, old_event, new_events):
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

                for pulse in new_params["playback_signature"].waveform.pulses:
                    pulse_name_mapping[pulse.pulse] = (
                        pulse.pulse + f"_compr_{new_event.label}"
                    )

                new_params["playback_signature"].waveform.length = new_length
                new_params["playback_signature"].waveform.pulses = None
                new_params["playback_signature"].waveform.samples = SamplesSignature(
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

    def _pulses_compatible_for_compression(self, pulses: List[NamedTuple]):
        sorted_pulses = sorted(pulses, key=lambda x: x.start)
        n = len(sorted_pulses)

        for i in range(n - 1):
            pi = sorted_pulses[i]
            pj = sorted_pulses[i + 1]

            if pi.end > pj.start and pi.can_compress != pj.can_compress:
                return False

        return True

    def _compress_waves(
        self,
        sampled_events,
        sampled_signatures,
        signal_id,
        min_play_wave,
        pulse_defs,
        device_type,
    ):
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
                    wave_form = event.params["playback_signature"].waveform
                    pulses_not_in_pulsedef = [
                        pulse.pulse
                        for pulse in wave_form.pulses
                        if pulse.pulse not in pulse_defs
                    ]
                    assert all(pulse is None for pulse in pulses_not_in_pulsedef)
                    if len(pulses_not_in_pulsedef) > 0:
                        continue
                    if (
                        any(
                            pulse_defs[pulse.pulse].can_compress
                            for pulse in wave_form.pulses
                        )
                        and device_type.is_qa_device
                    ):
                        pulse_names = [
                            pulse.pulse
                            for pulse in wave_form.pulses
                            if pulse_defs[pulse.pulse].can_compress
                        ]
                        _logger.warning(
                            "Requested to compress pulse(s) %s which are to be played on a QA device, which does not support playHold",
                            ",".join(pulse_names),
                        )
                        continue
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
                    if not self._pulses_compatible_for_compression(pulse_compr_infos):
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
                        sample_dict, min_play_wave, compressible_segments
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
                new_awg_events, pulse_name_mapping = self._emit_new_awg_events(
                    event_group[idx], new_events
                )

                old_event = event_group[idx]
                event_group[idx : idx + 1] = new_awg_events

                old_waveform = old_event.params["playback_signature"].waveform
                for new_awg_event in new_awg_events:
                    if new_awg_event.type == AWGEventType.PLAY_WAVE:
                        new_waveform = new_awg_event.params[
                            "playback_signature"
                        ].waveform
                        new_length = len(
                            next(iter(new_waveform.samples.samples_map.values()))
                        )

                        old_sampled_signature = self._sampled_signatures[signal_id][
                            old_waveform
                        ]
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
                            new_signature_pulse_map[
                                new_name
                            ] = new_signature_pulse_map.pop(old_name)

                        for sp_map in new_signature_pulse_map.values():
                            sp_map.length_samples = new_length
                            for signature in sp_map.instances:
                                signature.length = new_length

                        self._sampled_signatures[signal_id][
                            new_waveform
                        ] = new_sampled_signature

        # evict waveforms that have been compressed, and thus replaced with one or more, shorter, waves
        for waveform_signature in compressed_waveform_signatures:
            self._sampled_signatures[signal_id].pop(waveform_signature)

    def _gen_seq_c_per_awg(
        self,
        awg: AWGInfo,
        events: List[Any],
        pulse_defs: Dict[str, PulseDef],
    ):
        function_defs_generator = SeqCGenerator()
        declarations_generator = SeqCGenerator()
        _logger.debug("Generating seqc for awg %d of %s", awg.awg_number, awg.device_id)
        _logger.debug("AWG Object = \n%s", awg)
        sampled_events = AWGSampledEventSequence()

        global_sampling_rate, global_delay = self._calc_global_awg_params(awg)
        if self.EMIT_TIMING_COMMENTS:
            declarations_generator.add_comment(
                f"{awg.device_type}/{awg.awg_number} global delay {EngNumber(global_delay)} sampling_rate: {EngNumber(global_sampling_rate)}Sa/s "
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
                EventType.RESET_SW_OSCILLATOR_PHASE,
                EventType.INCREMENT_OSCILLATOR_PHASE,
                EventType.SET_OSCILLATOR_FREQUENCY_START,
            )
            or set(
                to_item["signal_id"] for to_item in event.get("trigger_output", [])
            ).intersection(signal_ids)
        )
        has_match_case = False
        for event in events:
            if (
                event["event_type"] == EventType.SECTION_START
                and event.get("state") is not None
                and event["section_name"] in own_sections
            ):
                has_match_case = True
                use_command_table = True
        for event in events:
            if (
                event["event_type"] == EventType.SUBSECTION_END
                and event["subsection_name"] in own_sections
            ):
                # by looking at SUBSECTION_END, we'll always walk towards the tree root
                own_sections.add(event["section_name"])

        if has_match_case:
            declarations_generator.add_variable_declaration("current_seq_step", 0)

        if awg.device_type not in [DeviceType.HDAWG, DeviceType.SHFSG]:
            use_command_table = False

        # filter the event list
        events = [
            event
            for event in events
            if "section_name" not in event or event.get("section_name") in own_sections
        ]

        _logger.debug(
            "Analyzing initialization events for awg %d of %s",
            awg.awg_number,
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
        sampled_events.merge(loop_events)

        prng_setup_events = analyze_prng_setup_times(
            events, global_sampling_rate, global_delay
        )
        sampled_events.merge(prng_setup_events)

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

            if (
                trigger_events.sequence
                and signal_obj.awg.device_type == DeviceType.SHFQA
                and acquire_events.has_matching_event(
                    lambda ev: "spectroscopy" in ev.get("acquisition_type")
                )
            ):
                raise LabOneQException(
                    "Trigger signals cannot be used on SHFQA in spectroscopy mode"
                )

            if signal_obj.signal_type == "integration":
                assert self._integration_weights is not None
                _logger.debug(
                    "Calculating integration weights for signal %s. There are %d acquire events.",
                    signal_obj.id,
                    len(acquire_events.sequence),
                )
                self._integration_weights[
                    signal_obj.id
                ] = calculate_integration_weights(
                    acquire_events, signal_obj, pulse_defs
                )

            sampled_events.merge(acquire_events)

        if awg.signal_type == AWGSignalType.MULTI:
            _logger.debug("Multi signal %s", awg)
            assert len(awg.signals) > 0
            delay = 0.0
            device_type = awg.signals[0].awg.device_type
            sampling_rate = awg.signals[0].awg.sampling_rate
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
                device_type=device_type,
                sampling_rate=sampling_rate,
                delay=delay,
                other_events=sampled_events,
                phase_resolution_range=self.phase_resolution_range(),
                amplitude_resolution_range=self.amplitude_resolution_range(),
                waveform_size_hints=self.waveform_size_hints(device_type),
                use_command_table=use_command_table,
                use_amplitude_increment=self._settings.USE_AMPLITUDE_INCREMENT,
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=sampling_rate,
                signal_type="iq",
                device_type=device_type,
                multi_iq_signal=True,
                mixer_type=mixer_type,
            )

            self._sampled_signatures[virtual_signal_id] = sampled_signatures
            sampled_events.merge(interval_events)

            min_play_waves = [
                signal.awg.device_type.min_play_wave for signal in awg.signals
            ]
            assert all(
                min_play_wave == min_play_waves[0] for min_play_wave in min_play_waves
            )
            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                signal_id=virtual_signal_id,
                min_play_wave=min_play_waves[0],
                pulse_defs=pulse_defs,
                device_type=device_type,
            )

            if virtual_signal_id in self._sampled_signatures:
                for sig, sampled in self._sampled_signatures[virtual_signal_id].items():
                    if not sampled:
                        continue
                    sig_string = sig.signature_string()

                    length = sig.length
                    declarations_generator.add_wave_declaration(
                        device_type,
                        awg.signal_type.value,
                        sig_string,
                        length,
                        False,
                        False,
                    )

        elif awg.signal_type != AWGSignalType.DOUBLE:
            for signal_obj in awg.signals:
                if signal_obj.awg.device_type == DeviceType.SHFQA:
                    sub_channel = signal_obj.channels[0]
                else:
                    sub_channel = None
                interval_events = analyze_play_wave_times(
                    events=events,
                    signals={signal_obj.id: signal_obj},
                    device_type=signal_obj.awg.device_type,
                    sampling_rate=signal_obj.awg.sampling_rate,
                    delay=signal_obj.total_delay,
                    other_events=sampled_events,
                    phase_resolution_range=self.phase_resolution_range(),
                    amplitude_resolution_range=self.amplitude_resolution_range(),
                    waveform_size_hints=self.waveform_size_hints(
                        signal_obj.awg.device_type
                    ),
                    sub_channel=sub_channel,
                    use_command_table=use_command_table,
                    use_amplitude_increment=self._settings.USE_AMPLITUDE_INCREMENT,
                )

                sampled_signatures = self._sample_pulses(
                    signal_obj.id,
                    interval_events,
                    pulse_defs=pulse_defs,
                    sampling_rate=signal_obj.awg.sampling_rate,
                    signal_type=signal_obj.signal_type,
                    device_type=signal_obj.awg.device_type,
                    mixer_type=signal_obj.mixer_type,
                )

                self._sampled_signatures[signal_obj.id] = sampled_signatures
                sampled_events.merge(interval_events)

                self._compress_waves(
                    sampled_events=sampled_events,
                    sampled_signatures=sampled_signatures,
                    signal_id=signal_obj.id,
                    min_play_wave=signal_obj.awg.device_type.min_play_wave,
                    pulse_defs=pulse_defs,
                    device_type=signal_obj.awg.device_type,
                )

                if signal_obj.id in self._sampled_signatures:
                    signature_infos = []
                    for sig, sampled in self._sampled_signatures[signal_obj.id].items():
                        has_marker1 = False
                        has_marker2 = False
                        if sampled:
                            if "samples_marker1" in sampled:
                                has_marker1 = True
                            if "samples_marker2" in sampled:
                                has_marker2 = True

                            sig_string = sig.signature_string()
                            length = sig.length
                            signature_infos.append(
                                (sig_string, length, (has_marker1, has_marker2))
                            )

                    for siginfo in sorted(signature_infos):
                        declarations_generator.add_wave_declaration(
                            signal_obj.awg.device_type,
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
                device_type=signal_a.awg.device_type,
                sampling_rate=signal_a.awg.sampling_rate,
                delay=signal_a.total_delay,
                other_events=sampled_events,
                phase_resolution_range=self.phase_resolution_range(),
                amplitude_resolution_range=self.amplitude_resolution_range(),
                waveform_size_hints=self.waveform_size_hints(signal_a.awg.device_type),
                use_command_table=False,  # do not set amplitude/oscillator phase via CT
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=signal_a.awg.sampling_rate,
                signal_type=signal_a.signal_type,
                device_type=signal_a.awg.device_type,
                mixer_type=signal_a.mixer_type,
            )

            self._sampled_signatures[virtual_signal_id] = sampled_signatures
            sampled_events.merge(interval_events)

            assert (
                signal_a.awg.device_type.min_play_wave
                == signal_b.awg.device_type.min_play_wave
            )

            self._compress_waves(
                sampled_events=sampled_events,
                sampled_signatures=sampled_signatures,
                signal_id=virtual_signal_id,
                min_play_wave=signal_a.awg.device_type.min_play_wave,
                pulse_defs=pulse_defs,
                device_type=awg.device_type,
            )

            if virtual_signal_id in self._sampled_signatures:
                for sig, sampled in self._sampled_signatures[virtual_signal_id].items():
                    if not sampled:
                        continue
                    sig_string = sig.signature_string()
                    length = sig.length
                    declarations_generator.add_wave_declaration(
                        awg.device_type,
                        awg.signal_type.value,
                        sig_string,
                        length,
                        False,
                        False,
                    )
        self.post_process_sampled_events(awg, sampled_events)

        _logger.debug(
            "** Start processing events for awg %d of %s",
            awg.awg_number,
            awg.device_id,
        )
        command_table_tracker = CommandTableTracker(awg.device_type)

        deferred_function_calls = SeqCGenerator()
        init_generator = SeqCGenerator()

        seqc_tracker = SeqCTracker(
            init_generator=init_generator,
            deferred_function_calls=deferred_function_calls,
            sampling_rate=global_sampling_rate,
            delay=global_delay,
            device_type=awg.device_type,
            emit_timing_comments=self.EMIT_TIMING_COMMENTS,
            logger=_logger,
        )

        handler = SampledEventHandler(
            seqc_tracker=seqc_tracker,
            command_table_tracker=command_table_tracker,
            function_defs_generator=function_defs_generator,
            declarations_generator=declarations_generator,
            wave_indices=WaveIndexTracker(),
            feedback_connections=self._feedback_connections,
            awg=awg,
            device_type=awg.device_type,
            channels=awg.signals[0].channels,
            sampled_signatures=self._sampled_signatures,
            use_command_table=use_command_table,
            emit_timing_comments=self.EMIT_TIMING_COMMENTS,
            use_current_sequencer_step=has_match_case,
        )

        setup_sine_phase(awg, command_table_tracker, init_generator, use_command_table)

        add_wait_trigger_statements(awg, init_generator, deferred_function_calls)
        try:
            handler.handle_sampled_events(sampled_events)
        except EntryLimitExceededError as error:
            msg = (
                f"Device '{awg.device_id}', AWG({awg.awg_number}): "
                "Exceeded maximum number of command table entries.\n"
                "Try the following:\n"
                "- Reduce the number of sweep steps\n"
                "- Reduce the number of variations in the pulses that are being played\n"
            )
            raise LabOneQException(f"Compiler error. {msg}") from error

        self._feedback_register_config[
            awg.key
        ].command_table_offset = handler.command_table_match_offset

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

        if awg.trigger_mode == TriggerMode.DIO_TRIGGER and awg.awg_number == 0:
            seq_c_generator.add_function_call_statement("setDIO", ["0"])

        seq_c_text = seq_c_generator.generate_seq_c()

        for line in seq_c_text.splitlines():
            _logger.debug(line)

        awg_key = AwgKey(awg.device_id, awg.awg_number)

        self._src[awg_key] = {"text": seq_c_text}
        self._wave_indices_all[awg_key] = {"value": handler.wave_indices.wave_indices()}
        if use_command_table:
            self._command_tables[awg_key] = {
                "ct": handler.command_table_tracker.command_table()
            }

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
        signal_id,
        interval_events: AWGSampledEventSequence,
        pulse_defs: Dict[str, PulseDef],
        sampling_rate,
        signal_type,
        device_type,
        mixer_type,
        multi_iq_signal=False,
    ):
        sampled_signatures: Dict[WaveformSignature, Dict] = {}
        signatures = set()
        for interval_event_list in interval_events.sequence.values():
            for interval_event in interval_event_list:
                if interval_event.type != AWGEventType.PLAY_WAVE:
                    continue
                signature: PlaybackSignature = interval_event.params[
                    "playback_signature"
                ]
                _logger.debug("Signature found %s in %s", signature, interval_event)
                if any(p.pulse for p in signature.waveform.pulses):
                    signatures.add(signature)
                else:
                    sampled_signatures[signature.waveform] = None
        _logger.debug("Signatures: %s", signatures)

        needs_conjugate = device_type == DeviceType.SHFSG
        for signature in signatures:
            length = signature.waveform.length
            _logger.debug(
                "Sampling pulses for signature %s for signal %s, length %d device type %s",
                signature.waveform.signature_string(),
                signal_id,
                length,
                device_type.value,
            )

            if length % device_type.sample_multiple != 0:
                raise Exception(
                    f"Length of signature {signature.waveform.signature_string()} is not divisible by {device_type.sample_multiple}, which it needs to be for {device_type.value}"
                )

            signature_pulse_map: Dict[str, PulseWaveformMap] = {}
            samples_i = np.zeros(length)
            samples_q = np.zeros(length)
            samples_marker1 = np.zeros(length, dtype=np.int16)
            samples_marker2 = np.zeros(length, dtype=np.int16)
            has_marker1 = False
            has_marker2 = False

            has_q = False

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

                oscillator_phase = pulse_part.oscillator_phase

                baseband_phase = pulse_part.increment_oscillator_phase
                used_oscillator_frequency = pulse_part.oscillator_frequency

                _logger.debug(
                    " Sampling pulse %s using oscillator frequency %s",
                    pulse_part,
                    used_oscillator_frequency,
                )

                if used_oscillator_frequency and device_type == DeviceType.SHFSG:
                    amplitude /= math.sqrt(2)

                iq_phase = 0.0

                if pulse_part.phase is not None:
                    # According to "LabOne Q Software: Signal, channel and oscillator concept" REQ 1.3
                    iq_phase += pulse_part.phase

                # In case oscillator phase can't be set at runtime (e.g. HW oscillator without
                # phase control from a sequencer), apply oscillator phase on a baseband (iq) signal
                iq_phase += baseband_phase or 0.0

                iq_phase += oscillator_phase or 0.0

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
                    markers=None
                    if not pulse_part.markers
                    else [{k: v for k, v in m} for m in pulse_part.markers],
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

                if "samples_marker1" in sampled_pulse:
                    if (
                        pulse_part.channel == 1
                        and not multi_iq_signal
                        and not device_type == DeviceType.SHFQA
                    ):
                        raise LabOneQException(
                            f"Marker 1 not supported on channel 2 of multiplexed RF signal {signal_id}"
                        )

                    self.stencil_samples(
                        pulse_part.start,
                        sampled_pulse["samples_marker1"],
                        samples_marker1,
                    )
                    has_marker1 = True

                if "samples_marker2" in sampled_pulse:
                    if (
                        pulse_part.channel == 0
                        and not multi_iq_signal
                        and not device_type == DeviceType.SHFQA
                    ):
                        raise LabOneQException(
                            f"Marker 2 not supported on channel 1 of multiplexed RF signal {signal_id}"
                        )

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
                        modulation_phase=oscillator_phase,
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

    def waves(self):
        return self._waves

    def src(self):
        return self._src

    def wave_indices(self):
        return self._wave_indices_all

    def command_tables(self):
        return self._command_tables

    def pulse_map(self) -> Dict[str, PulseMapEntry]:
        return self._pulse_map

    def integration_times(self) -> IntegrationTimes:
        return self._integration_times

    def signal_delays(self) -> SignalDelays:
        return self._signal_delays

    def feedback_register_config(self) -> Dict[AwgKey, FeedbackRegisterConfig]:
        # convert defaultdict to dict
        return dict(self._feedback_register_config)

    def feedback_connections(self) -> Dict[str, FeedbackConnection]:
        return self._feedback_connections

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
                feedback_registers = list()
                play: List[AWGEvent] = []
                acquire: List[AWGEvent] = []
                for x in sampled_event_list:
                    if "signal_id" in x.params and x.type != AWGEventType.ACQUIRE:
                        play.append(x)
                        start = x.start
                        end = x.end if end is None else max(end, x.end)
                    elif x.type == AWGEventType.ACQUIRE:
                        start = x.start
                        if "acquisition_type" in x.params:
                            acquisition_types.update(x.params["acquisition_type"])
                        feedback_register = x.params.get("feedback_register")
                        if feedback_register is not None:
                            feedback_registers.append(feedback_register)
                        acquire.append(x)
                        end = x.end if end is None else max(end, x.end)
                if len(play) > 0 and len(acquire) == 0 and has_acquire:
                    _logger.warning("Problem:")
                    for log_event in sampled_event_list:
                        _logger.warning("  %s", log_event)
                    raise Exception("Play and acquire must happen at the same time")
                if len(feedback_registers) > 1:
                    _logger.warning(
                        "Conflicting feedback register allocation detected, please contact development."
                    )
                if len(play) > 0 or len(acquire) > 0:
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
                            "feedback_register": None
                            if len(feedback_registers) == 0
                            else feedback_registers[0],
                        },
                    )

                    for to_delete in play + acquire:
                        sampled_event_list.remove(to_delete)
                    sampled_event_list.append(qa_event)
