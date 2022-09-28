# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from engineering_notation import EngNumber
from intervaltree import IntervalTree
from sortedcontainers import SortedDict
from laboneq.compiler.section_graph import SectionGraph

from laboneq.core.types.compiled_experiment import (
    PulseInstance,
    PulseMapEntry,
    PulseWaveformMap,
)
from laboneq.compiler.event_graph import EventType
from .compiler_settings import CompilerSettings
from .device_type import DeviceType
from .fastlogging import NullLogger
from .interval_calculator import (
    calculate_intervals,
    MinimumWaveformLengthViolation,
)
from .seq_c_generator import SeqCGenerator, string_sanitize
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    length_to_samples,
    interval_to_samples,
    sample_pulse,
)

_logger = logging.getLogger(__name__)

if _logger.getEffectiveLevel() == logging.DEBUG:
    _dlogger = _logger
else:
    _logger.info("Debug logging disabled for %s", __name__)
    _dlogger = NullLogger()


@dataclass(init=True, repr=True, order=True, frozen=True)
class AwgKey:
    device_id: str
    awg_number: int


class TriggerMode(Enum):
    NONE = "none"
    DIO_TRIGGER = "dio_trigger"
    DIO_WAIT = "dio_wait"
    INTERNAL_TRIGGER_WAIT = "internal_trigger_wait"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class AWGSignalType(Enum):
    SINGLE = "single"  # Only one channel is played
    DOUBLE = "double"  # Two independent channels
    IQ = "iq"  # Two channels form an I/Q signal
    MULTI = "multi"  # Multiple logical channels mixed


@dataclass
class AWGInfo:
    device_id: str
    signal_type: AWGSignalType
    awg_number: int
    seqc: str
    signal_channels: List[Tuple[str, int]] = field(default_factory=list)
    device_type: DeviceType = None
    signals: List[SignalObj] = field(default_factory=list)


@dataclass(init=True, repr=True, order=True)
class SignalObj:
    id: str
    sampling_rate: float
    delay: float
    signal_type: str
    device_id: str
    device_type: DeviceType
    oscillator_frequency: float = None  # for software modulation only
    trigger_mode: TriggerMode = TriggerMode.NONE
    reference_clock_source: Optional[str] = None
    pulse_defs: Dict = field(default_factory=dict)
    pulses: List = field(default_factory=list)
    channels: List = field(default_factory=list)
    awg: AWGInfo = None


class WaveIndexTracker:
    def __init__(self):
        self._wave_indices: Dict[str, Tuple[int, Any]] = {}
        self._next_wave_index: int = 0
        self._numbered_waves = {}

    def lookup_index(self, wave_id: str, signal_type: str) -> int:
        if signal_type == "csv":
            # For CSV store only the wave_id, do not allocate an index
            self._wave_indices[wave_id] = [-1, signal_type]
            return None
        wave_index = self._wave_indices.get(wave_id)
        if wave_index is None:
            index = self._next_wave_index
            self._next_wave_index += 1
            self._wave_indices[wave_id] = [index, signal_type]
            return index
        return None

    def add_numbered_wave(self, wave_id: str, signal_type: str, index):
        self._wave_indices[wave_id] = [index, signal_type]

    def wave_indices(self) -> Dict[str, Tuple[int, Any]]:
        return self._wave_indices


class CodeGenerator:

    EMIT_TIMING_COMMENTS = False
    USE_ZSYNC_TRIGGER = True

    DELAY_FIRST_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_OTHER_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_UHFQA = 140 / DeviceType.HDAWG.sampling_rate

    PHASE_FIXED_SCALE = 1000000000

    PHASE_RESOLUTION_BITS = 7

    # This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
    # to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
    SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10

    def __init__(self, compiler_settings: CompilerSettings = None):
        self._compiler_settings = compiler_settings or CompilerSettings()
        self._signals: Dict[Any, SignalObj] = {}
        self._code = {}
        self._src = []
        self._wave_indices_all = []
        self._waves = []
        self._pulse_map: Dict[str, PulseMapEntry] = {}
        self._sampled_signatures = {}
        self._sampled_events = None
        self._awgs: Dict[AwgKey, AWGInfo] = {}
        self._events_in_samples = {}
        self._integration_weights = None
        self._long_signatures = {}
        self._simultaneous_acquires = None
        self._total_execution_time = None

    def integration_weights(self):
        return self._integration_weights

    def simultaneous_acquires(self):
        return self._simultaneous_acquires

    def total_execution_time(self):
        return self._total_execution_time

    def _add_signal_to_awg(self, signal_obj: SignalObj):
        awg_key = AwgKey(signal_obj.device_id, signal_obj.awg.awg_number)
        if awg_key not in self._awgs:
            self._awgs[awg_key] = copy.deepcopy(signal_obj.awg)
            self._awgs[awg_key].device_type = signal_obj.device_type
        self._awgs[awg_key].signals.append(signal_obj)

    def add_signal(self, signal: SignalObj):
        signal_obj = copy.deepcopy(signal)
        signal_obj.pulse_defs = {}
        signal_obj.pulses = []
        _dlogger.debug(signal_obj)
        self._signals[signal.id] = signal_obj
        self._add_signal_to_awg(signal_obj)

    def sort_signals(self):
        for awg in self._awgs.values():
            awg.signals = list(
                sorted(awg.signals, key=lambda signal: tuple(signal.channels))
            )

    def _add_timing_comment(
        self, generator, start_samples, end_samples, sampling_rate, delay
    ):
        if CodeGenerator.EMIT_TIMING_COMMENTS:
            start_time_ns = round((start_samples / sampling_rate - delay) * 1e10) / 10
            end_time_ns = round(((end_samples / sampling_rate) - delay) * 1e10) / 10
            generator.add_comment(
                f"{start_samples} - {end_samples} , {start_time_ns} ns - {end_time_ns} ns "
            )

    def _append_to_pulse_map(self, signature_pulse_map, sig_string):
        if signature_pulse_map is None:
            return
        for pulse_id, pulse_waveform_map in signature_pulse_map.items():
            pulse_map_entry = self._pulse_map.setdefault(pulse_id, PulseMapEntry())
            pulse_map_entry.waveforms[sig_string] = pulse_waveform_map

    def _save_wave_bin(
        self, samples, signature_pulse_map, sig_string: str, suffix: str,
    ):
        filename = sig_string + suffix + ".wave"
        self._waves.append({"filename": filename, "samples": samples})
        self._append_to_pulse_map(signature_pulse_map, sig_string)

    def gen_waves(self):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            _dlogger.debug("Sampled signatures: %s", self._sampled_signatures)
        for awg in self._awgs.values():
            # Handle integration weights separately
            for signal_obj in awg.signals:
                if signal_obj.id in self._integration_weights:
                    signal_weights = self._integration_weights[signal_obj.id]
                    for weight in signal_weights.values():
                        basename = weight["basename"]
                        if signal_obj.device_type.supports_complex_waves:
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
                        _dlogger.debug(
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
                            sig_string = self._signature_string(signature_key)
                            if signal_obj.device_type.supports_binary_waves:
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
                            elif signal_obj.device_type.supports_complex_waves:
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
                                    f"Device type {signal_obj.device_type} has invalid supported waves config."
                                )
            else:
                signal_obj = awg.signals[0]
                for signature_key, sampled_signature in self._sampled_signatures[
                    virtual_signal_id
                ].items():
                    sig_string = self._signature_string(signature_key)
                    if signal_obj.device_type.supports_binary_waves:
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
                    else:
                        raise RuntimeError(
                            f"Device type {signal_obj.device_type} has invalid supported waves config."
                        )

        if _logger.getEffectiveLevel() == logging.DEBUG:
            _dlogger.debug("Sampled signatures: %s", self._sampled_signatures)

            _dlogger.debug(self._waves)

    @staticmethod
    def _add_to_dict_list(dictlist, key, item):
        if not key in dictlist:
            dictlist[key] = []
        dictlist[key].append(item)

    @staticmethod
    def _merge_dict_list(dl1, dl2):
        for k, v in dl2.items():
            if k in dl1:
                dl1[k].extend(v)
            else:
                dl1[k] = v

    @staticmethod
    def _clear_deferred_function_calls(
        deferred_function_calls, event, loop_stack_generators
    ):
        if len(deferred_function_calls["calls"]) > 0:
            _dlogger.debug(
                "  Emitting deferred function calls: %s at event %s",
                deferred_function_calls,
                event or "None",
            )
            for call in deferred_function_calls["calls"]:
                if isinstance(call, dict):
                    loop_stack_generators[-1][-1].add_function_call_statement(
                        call["name"], call["args"]
                    )
                else:
                    loop_stack_generators[-1][-1].add_function_call_statement(call)
            deferred_function_calls["calls"] = []

    def _advance_current_time(
        self,
        current_time,
        sampled_event,
        deferred_function_calls,
        loop_stack_generators,
        signal_obj,
    ) -> int:
        """If `current_time` precedes the scheduled start of the event, emit playZero to catch up.

        Also clears deferred function calls within the context of the new playZero."""
        start = sampled_event["start"]
        signature = sampled_event["signature"]

        if start > current_time:
            play_zero_samples = start - current_time
            _dlogger.debug(
                "  Emitting %d play zero samples before signature %s for event %s",
                play_zero_samples,
                signature,
                sampled_event,
            )

            self._add_timing_comment(
                loop_stack_generators[-1][-1],
                current_time,
                current_time + play_zero_samples,
                signal_obj.sampling_rate,
                signal_obj.delay,
            )
            loop_stack_generators[-1][-1].add_play_zero_statement(
                play_zero_samples, signal_obj.device_type
            )
            self._clear_deferred_function_calls(
                deferred_function_calls, sampled_event, loop_stack_generators
            )
            current_time += play_zero_samples
        return current_time

    def gen_acquire_map(self, events: List[Any], sections: SectionGraph):
        loop_events = [
            e
            for e in events
            if e["event_type"] == "LOOP_ITERATION_END" and not e.get("shadow")
        ]
        averaging_loop_info = None
        innermost_loop: Dict[str, Any] = None
        outermost_loop: Dict[str, Any] = None
        for e in loop_events:
            section_info = sections.section_info(e["section_name"])
            if section_info["averaging_type"] == "hardware":
                averaging_loop_info = section_info
            if (
                innermost_loop is None
                or e["nesting_level"] > innermost_loop["nesting_level"]
            ):
                innermost_loop = e
            if (
                outermost_loop is None
                or e["nesting_level"] < outermost_loop["nesting_level"]
            ):
                outermost_loop = e
        averaging_loop = (
            None
            if averaging_loop_info is None
            else innermost_loop
            if averaging_loop_info["averaging_mode"] == "sequential"
            else outermost_loop
        )
        if (
            averaging_loop is not None
            and averaging_loop["section_name"] != averaging_loop_info["section_id"]
        ):
            raise RuntimeError(
                f"Internal error: couldn't unambiguously determine the hardware averaging loop - "
                f"innermost '{innermost_loop['section_name']}', outermost '{outermost_loop['section_name']}', "
                f"hw avg '{averaging_loop_info['section_id']}' with mode '{averaging_loop_info['averaging_mode']}' "
                f"expected to match '{averaging_loop['section_name']}'"
            )
        unrolled_avg_matcher = re.compile(
            "(?!)"  # Never match anything
            if averaging_loop is None
            else f"{averaging_loop['section_name']}_[0-9]+"
        )
        # timestamp -> map[signal -> handle]
        self._simultaneous_acquires: Dict[float, Dict[str, str]] = {}
        for e in events:
            if (
                e["event_type"] == "ACQUIRE_START"
                # Skip injected spectroscopy events (obsolete hack)
                and e["acquisition_type"] != ["spectroscopy"]
            ):
                if e.get("shadow") and unrolled_avg_matcher.match(
                    e.get("loop_iteration")
                ):
                    continue  # Skip events for unrolled averaging loop
                time_events = self._simultaneous_acquires.setdefault(e["time"], {})
                time_events[e["signal"]] = e["acquire_handle"]

    def gen_seq_c(self, events: List[Any], pulse_defs):
        self._total_execution_time = events[-1].get("time") if len(events) > 0 else None
        self.sort_signals()
        self._integration_weights = {}

        ignore_pulses = set()
        for pulse_id, pulse_def in pulse_defs.items():
            if abs(pulse_def["amplitude"]) < 1e-12:  # ignore zero amplitude pulses
                ignore_pulses.add(pulse_id)
        if len(ignore_pulses) > 0:
            _logger.debug(
                "Ignoring pulses because of zero amplitude: %s", ignore_pulses
            )

        filtered_events: List[Any] = [
            event
            for event in events
            if "play_wave_id" not in event or event["play_wave_id"] not in ignore_pulses
        ]

        for _, awg in sorted(
            self._awgs.items(),
            key=lambda item: item[0].device_id + str(item[0].awg_number),
        ):
            self._gen_seq_c_per_awg(awg, filtered_events, pulse_defs)

    def _calc_global_awg_params(self, awg: AWGInfo) -> Tuple[float, float]:
        global_sampling_rate = None
        global_delay = None
        for signal_obj in awg.signals:
            if (
                round(signal_obj.delay * signal_obj.sampling_rate)
                % signal_obj.device_type.sample_multiple
                != 0
            ):
                raise RuntimeError(
                    f"Delay {signal_obj.delay} s = {round(signal_obj.delay*signal_obj.sampling_rate)} samples on signal {signal_obj.id} is not compatible with the sample multiple of {signal_obj.device_type.sample_multiple} on {signal_obj.device_type}"
                )
            if awg.signal_type != AWGSignalType.IQ and global_delay is not None:
                if global_delay != signal_obj.delay:
                    raise RuntimeError(
                        f"Delay {signal_obj.delay * 1e9:.2f} ns on signal "
                        f"{signal_obj.id} is different from other delays "
                        f"({global_delay * 1e9:.2f} ns) on the same AWG."
                    )
            global_sampling_rate = signal_obj.sampling_rate
            global_delay = signal_obj.delay
        return global_sampling_rate, global_delay

    def _gen_seq_c_per_awg(self, awg: AWGInfo, events: List[Any], pulse_defs):
        wave_indices = WaveIndexTracker()
        declarations_generator = SeqCGenerator()
        declared_variables = set()
        _logger.debug("Generating seqc for awg %d of %s", awg.awg_number, awg.device_id)
        _dlogger.debug("AWG Object = \n%s", awg)
        sampled_events = SortedDict()
        filename = awg.seqc

        global_sampling_rate, global_delay = self._calc_global_awg_params(awg)

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
        )
        for event in events:
            if (
                event["event_type"] == EventType.SUBSECTION_END
                and event["subsection_name"] in own_sections
            ):
                # by looking at SUBSECTION_END, we'll always walk towards the tree root
                own_sections.add(event["section_name"])

        # filter the event list
        events = [
            event
            for event in events
            if "section_name" not in event or event.get("section_name") in own_sections
        ]

        init_events = self._analyze_init_times(
            awg.device_id, global_sampling_rate, global_delay
        )
        self._merge_dict_list(sampled_events, init_events)

        phase_reset_events = self._analyze_phase_reset_times(
            events, awg.device_id, global_sampling_rate, global_delay
        )
        self._merge_dict_list(sampled_events, phase_reset_events)

        loop_events = self._analyze_loop_times(
            awg, events, global_sampling_rate, global_delay,
        )
        self._merge_dict_list(sampled_events, loop_events)

        for signal_obj in awg.signals:

            set_oscillator_events = self._analyze_set_oscillator_times(
                events,
                signal_obj.id,
                device_id=signal_obj.device_id,
                device_type=signal_obj.device_type,
                sampling_rate=signal_obj.sampling_rate,
                delay=signal_obj.delay,
            )
            self._merge_dict_list(sampled_events, set_oscillator_events)

            acquire_events = self._analyze_acquire_times(
                events,
                signal_obj.id,
                sampling_rate=signal_obj.sampling_rate,
                delay=signal_obj.delay,
                sample_multiple=signal_obj.device_type.sample_multiple,
                channels=signal_obj.channels,
            )

            if signal_obj.signal_type == "integration":
                _logger.debug(
                    "Calculating integration weights for signal %s. There are %d acquire events.",
                    signal_obj.id,
                    len(acquire_events.values()),
                )
                integration_weights = {}
                for event_list in acquire_events.values():
                    for event in event_list:
                        _dlogger.debug("For weights, look at %s", event)
                        if "play_wave_id" in event:
                            play_wave_id = event["play_wave_id"]
                            _dlogger.debug(
                                "Event %s has play wave id %s", event, play_wave_id
                            )
                            if (
                                play_wave_id in pulse_defs
                                and play_wave_id not in integration_weights
                            ):
                                pulse_def = pulse_defs[play_wave_id]
                                _dlogger.debug("Pulse def: %s", pulse_def)

                                function = pulse_def.get("function")
                                samples = pulse_def.get("samples")

                                if function is None and samples is None:
                                    # Not a real pulse, just a placeholder for the length - skip
                                    continue

                                amplitude = 1.0
                                if "amplitude" in pulse_def:
                                    amplitude = pulse_def["amplitude"]

                                if samples is None:
                                    amplitude = amplitude * math.sqrt(2)

                                length = pulse_def.get("length")
                                if length is None:
                                    length = len(samples) / signal_obj.sampling_rate

                                _logger.debug(
                                    "Sampling integration weights for %s with modulation_frequency %f",
                                    signal_obj.id,
                                    signal_obj.oscillator_frequency,
                                )
                                complex_modulation = awg.device_type != DeviceType.UHFQA

                                samples = sample_pulse(
                                    signal_type="iq",
                                    sampling_rate=signal_obj.sampling_rate,
                                    length=length,
                                    amplitude=amplitude,
                                    pulse_function=pulse_def["function"],
                                    modulation_frequency=signal_obj.oscillator_frequency,
                                    iq_phase=signal_obj.device_type.iq_phase,
                                    samples=samples,
                                    complex_modulation=complex_modulation,
                                )

                                samples["basename"] = (
                                    signal_obj.device_id
                                    + "_"
                                    + str(signal_obj.awg.awg_number)
                                    + "_"
                                    + str(min(signal_obj.channels))
                                    + "_"
                                    + play_wave_id
                                )
                                integration_weights[play_wave_id] = samples

                self._integration_weights[signal_obj.id] = integration_weights

            self._merge_dict_list(sampled_events, acquire_events)

        signals_to_process = awg.signals
        oscillator_frequencies_per_channel = []
        if awg.signal_type == AWGSignalType.MULTI:
            _dlogger.debug("Multi signal %s", awg)
            iq_signal_ids = []
            signals_to_process = []
            iq_signals = []
            for signal_obj in awg.signals:
                if signal_obj.signal_type != "integration":
                    _dlogger.debug(
                        "Non-integration signal in multi signal: %s", signal_obj
                    )
                    iq_signal_ids.append(signal_obj.id)
                    oscillator_frequencies_per_channel.append(
                        signal_obj.oscillator_frequency
                    )
                    iq_signals.append(signal_obj)
                else:
                    signals_to_process.append(signal_obj)

            virtual_signal_id = "_".join(iq_signal_ids)
            iq_phase = iq_signals[0].device_type.iq_phase

            interval_events = self.analyze_play_wave_times(
                events,
                signal_ids=iq_signal_ids,
                signal_obj=signal_obj,
                other_events=sampled_events,
                iq_phase=iq_phase,
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=signal_obj.sampling_rate,
                signal_type="iq",
                device_type=signal_obj.device_type,
                oscillator_frequency=None,
                oscillator_frequencies_per_channel=oscillator_frequencies_per_channel,
                multi_iq_signal=True,
            )

            self._sampled_signatures[virtual_signal_id] = sampled_signatures
            self._merge_dict_list(sampled_events, interval_events)
            if virtual_signal_id in self._sampled_signatures:
                for sig in self._sampled_signatures[virtual_signal_id].keys():
                    sig_string = self._signature_string(sig)
                    if sig_string in self._long_signatures:
                        declarations_generator.add_comment(
                            f"{sig_string} is {self._long_signatures[sig_string]}"
                        )
                    length = sig[0]
                    declarations_generator.add_wave_declaration(
                        signal_obj.device_type,
                        awg.signal_type.value,
                        sig_string,
                        length,
                    )

        if awg.signal_type != AWGSignalType.DOUBLE:
            for signal_obj in signals_to_process:

                iq_phase = signal_obj.device_type.iq_phase
                if signal_obj.device_type == DeviceType.SHFQA:
                    sub_channel = signal_obj.channels[0]
                else:
                    sub_channel = None
                interval_events = self.analyze_play_wave_times(
                    events,
                    signal_ids=[signal_obj.id],
                    signal_obj=signal_obj,
                    other_events=sampled_events,
                    iq_phase=iq_phase,
                    sub_channel=sub_channel,
                )

                sampled_signatures = self._sample_pulses(
                    signal_obj.id,
                    interval_events,
                    pulse_defs=pulse_defs,
                    sampling_rate=signal_obj.sampling_rate,
                    signal_type=signal_obj.signal_type,
                    device_type=signal_obj.device_type,
                    oscillator_frequency=signal_obj.oscillator_frequency,
                )

                self._sampled_signatures[signal_obj.id] = sampled_signatures
                self._merge_dict_list(sampled_events, interval_events)

                if signal_obj.id in self._sampled_signatures:
                    signature_infos = []
                    for sig in self._sampled_signatures[signal_obj.id].keys():
                        sig_string = self._signature_string(sig)
                        length = sig[0]
                        signature_infos.append((sig_string, length))

                    for siginfo in sorted(signature_infos):
                        declarations_generator.add_wave_declaration(
                            signal_obj.device_type,
                            signal_obj.signal_type,
                            siginfo[0],
                            siginfo[1],
                        )
        else:
            virtual_signal_id = (
                signals_to_process[0].id + "_" + signals_to_process[1].id
            )
            interval_events = self.analyze_play_wave_times(
                events,
                signal_ids=[signals_to_process[0].id, signals_to_process[1].id,],
                signal_obj=signals_to_process[0],
                other_events=sampled_events,
                iq_phase=0,
            )

            sampled_signatures = self._sample_pulses(
                virtual_signal_id,
                interval_events,
                pulse_defs=pulse_defs,
                sampling_rate=signals_to_process[0].sampling_rate,
                signal_type=signals_to_process[0].signal_type,
                device_type=signals_to_process[0].device_type,
                oscillator_frequency=None,
                oscillator_frequencies_per_channel=[
                    signals_to_process[0].oscillator_frequency,
                    signals_to_process[1].oscillator_frequency,
                ],
            )

            self._sampled_signatures[virtual_signal_id] = sampled_signatures
            self._merge_dict_list(sampled_events, interval_events)
            if virtual_signal_id in self._sampled_signatures:
                for sig in self._sampled_signatures[virtual_signal_id].keys():
                    sig_string = self._signature_string(sig)
                    length = sig[0]
                    declarations_generator.add_wave_declaration(
                        signals_to_process[0].device_type,
                        awg.signal_type.value,
                        sig_string,
                        length,
                    )
        self._sampled_events = sampled_events
        CodeGenerator.post_process_sampled_events(awg, sampled_events)

        current_time = 0
        deferred_function_calls = {"calls": []}

        init_generator = SeqCGenerator()
        if signal_obj.trigger_mode == TriggerMode.DIO_TRIGGER:
            if awg.awg_number == 0:
                init_generator.add_function_call_statement("setDIO", ["0"])
                init_generator.add_function_call_statement("wait", ["2000000"])
                init_generator.add_function_call_statement("playZero", ["512"])
                if signal_obj.reference_clock_source != "internal":
                    init_generator.add_function_call_statement("waitDigTrigger", ["1"])
                init_generator.add_function_call_statement("setDIO", ["0xffffffff"])
                init_generator.add_function_call_statement("waitDIOTrigger")
                delay_first_awg_samples = str(
                    round(signal_obj.sampling_rate * CodeGenerator.DELAY_FIRST_AWG / 16)
                    * 16
                )
                if int(delay_first_awg_samples) > 0:
                    deferred_function_calls["calls"].append(
                        {"name": "playZero", "args": [delay_first_awg_samples]}
                    )
                    deferred_function_calls["calls"].append(
                        {"name": "waitWave", "args": []}
                    )
            else:
                init_generator.add_function_call_statement("waitDIOTrigger")
                delay_other_awg_samples = str(
                    round(signal_obj.sampling_rate * CodeGenerator.DELAY_OTHER_AWG / 16)
                    * 16
                )
                if int(delay_other_awg_samples) > 0:
                    deferred_function_calls["calls"].append(
                        {"name": "playZero", "args": [delay_other_awg_samples]}
                    )
                    deferred_function_calls["calls"].append(
                        {"name": "waitWave", "args": []}
                    )

        elif signal_obj.trigger_mode == TriggerMode.DIO_WAIT:
            init_generator.add_variable_declaration("dio", "0xffffffff")
            body = SeqCGenerator()
            body.add_function_call_statement("getDIO", args=None, assign_to="dio")
            init_generator.add_do_while("dio & 0x0001", body)
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_uhfqa_samples = str(
                round(signal_obj.sampling_rate * CodeGenerator.DELAY_UHFQA / 8) * 8
            )
            if int(delay_uhfqa_samples) > 0:
                init_generator.add_function_call_statement(
                    "playZero", [delay_uhfqa_samples]
                )
                init_generator.add_function_call_statement("waitWave")

        elif signal_obj.trigger_mode == TriggerMode.INTERNAL_TRIGGER_WAIT:
            init_generator.add_function_call_statement("waitDigTrigger", ["1"])

        else:
            if (
                CodeGenerator.USE_ZSYNC_TRIGGER
                and signal_obj.device_type.supports_zsync
            ):
                init_generator.add_function_call_statement("waitZSyncTrigger")
            else:
                init_generator.add_function_call_statement("waitDIOTrigger")

        _dlogger.debug(
            "** Start processing events for awg %d of %d",
            awg.awg_number,
            awg.device_id,
        )
        loop_stack = []
        loop_stack_generators = [[init_generator, SeqCGenerator()]]

        last_event = None

        for sampled_event_list in sampled_events.values():
            _dlogger.debug("EventListBeforeSort: %s", sampled_event_list)
            sampled_event_list = CodeGenerator.sort_events(sampled_event_list)
            _dlogger.debug("-Processing list:")
            for sampled_event_for_log in sampled_event_list:
                _dlogger.debug("       %s", sampled_event_for_log)
            _dlogger.debug("-End event list")

            for sampled_event in sampled_event_list:
                _dlogger.debug("  Processing event %s", sampled_event)

                signature = sampled_event["signature"]
                start = sampled_event["start"]
                if "signal_id" in sampled_event:
                    signal_id = sampled_event["signal_id"]
                    if signature in self._sampled_signatures[signal_id]:
                        _dlogger.debug(
                            "  Found matching signature %s for event %s",
                            signature,
                            sampled_event,
                        )
                        current_time = self._advance_current_time(
                            current_time,
                            sampled_event,
                            deferred_function_calls,
                            loop_stack_generators,
                            signal_obj,
                        )

                        sig_string = self._signature_string(signature)
                        self._add_timing_comment(
                            loop_stack_generators[-1][-1],
                            current_time,
                            sampled_event["end"],
                            signal_obj.sampling_rate,
                            signal_obj.delay,
                        )

                        signal_type_for_wave_index = (
                            awg.signal_type.value
                            if signal_obj.device_type.supports_binary_waves
                            else "csv"  # Include CSV waves into the index to keep track of waves-AWG mapping
                        )
                        wave_index = wave_indices.lookup_index(
                            sig_string, signal_type_for_wave_index
                        )
                        play_wave_channel = None
                        if len(signal_obj.channels) > 0:
                            play_wave_channel = signal_obj.channels[0] % 2

                        loop_stack_generators[-1][-1].add_play_wave_statement(
                            signal_obj.device_type,
                            awg.signal_type.value,
                            sig_string,
                            wave_index,
                            play_wave_channel,
                        )
                        self._clear_deferred_function_calls(
                            deferred_function_calls,
                            sampled_event,
                            loop_stack_generators,
                        )

                        current_time = sampled_event["end"]
                if signature == "acquire":
                    _dlogger.debug("  Processing ACQUIRE EVENT %s", sampled_event)

                    args = [
                        "QA_INT_ALL",
                        "1" if "RAW" in sampled_event["acquisition_type"] else "0",
                    ]

                    if start > current_time:
                        current_time = self._advance_current_time(
                            current_time,
                            sampled_event,
                            deferred_function_calls,
                            loop_stack_generators,
                            signal_obj,
                        )
                        _dlogger.debug(
                            "  Deferring function call for %s", sampled_event
                        )
                        deferred_function_calls["calls"].append(
                            {"name": "startQA", "args": args}
                        )
                    else:
                        skip = False
                        if last_event is not None:
                            if (
                                last_event["signature"] == "acquire"
                                and last_event["start"] == start
                            ):
                                skip = True
                                _logger.debug(
                                    "Skipping acquire event %s because last event was also acquire: %s",
                                    sampled_event,
                                    last_event,
                                )
                        if not skip:
                            loop_stack_generators[-1][-1].add_function_call_statement(
                                "startQA", args
                            )

                elif signature == "QA_EVENT":

                    _dlogger.debug("  Processing QA_EVENT %s", sampled_event)
                    generator_channels = set()
                    for play_event in sampled_event["play_events"]:
                        _dlogger.debug("  play_event %s", play_event)
                        play_signature = play_event["signature"]
                        if "signal_id" in play_event:
                            signal_id = play_event["signal_id"]
                            if play_signature in self._sampled_signatures[signal_id]:
                                _dlogger.debug(
                                    "  Found matching signature %s for event %s",
                                    play_signature,
                                    play_event,
                                )
                                current_signal_obj = next(
                                    signal_obj
                                    for signal_obj in awg.signals
                                    if signal_obj.id == signal_id
                                )
                                generator_channels.update(current_signal_obj.channels)
                                sig_string = self._signature_string(play_signature)

                                wave_indices.add_numbered_wave(
                                    sig_string,
                                    "complex",
                                    current_signal_obj.channels[0],
                                )

                    integration_channels = [
                        event["channels"] for event in sampled_event["acquire_events"]
                    ]

                    integration_channels = [
                        item for sublist in integration_channels for item in sublist
                    ]

                    if len(integration_channels) > 0:

                        integrator_mask = "|".join(
                            map(lambda x: "QA_INT_" + str(x), integration_channels)
                        )
                    else:
                        integrator_mask = "QA_INT_NONE"

                    if len(generator_channels) > 0:
                        generator_mask = "|".join(
                            map(lambda x: "QA_GEN_" + str(x), generator_channels)
                        )
                    else:
                        generator_mask = "QA_GEN_NONE"

                    if "spectroscopy" in sampled_event["acquisition_type"]:
                        args = [0, 0, 0, 0, 1]
                    else:
                        args = [
                            generator_mask,
                            integrator_mask,
                            "1" if "RAW" in sampled_event["acquisition_type"] else "0",
                        ]

                    current_time = self._advance_current_time(
                        current_time,
                        sampled_event,
                        deferred_function_calls,
                        loop_stack_generators,
                        signal_obj,
                    )

                    if sampled_event["end"] > current_time:
                        play_zero_after_qa = sampled_event["end"] - current_time
                        self._add_timing_comment(
                            loop_stack_generators[-1][-1],
                            current_time,
                            current_time + play_zero_after_qa,
                            signal_obj.sampling_rate,
                            signal_obj.delay,
                        )
                        loop_stack_generators[-1][-1].add_play_zero_statement(
                            play_zero_after_qa, signal_obj.device_type
                        )
                    current_time = sampled_event["end"]

                    loop_stack_generators[-1][-1].add_function_call_statement(
                        "startQA", args
                    )
                    if "spectroscopy" in sampled_event["acquisition_type"]:
                        loop_stack_generators[-1][-1].add_function_call_statement(
                            "setTrigger", [0]
                        )

                elif signature in ("initial_reset_phase", "reset_phase"):
                    # If multiple phase reset events are scheduled at the same time,
                    # only process the *last* one. This way, `reset_phase` takes
                    # precedence.
                    # TODO (PW): Remove this check, once we no longer force oscillator
                    # resets at the start of the sequence.
                    last_reset = [
                        event
                        for event in sampled_event_list
                        if event["signature"] in ("reset_phase", "initial_reset_phase")
                    ][-1]
                    if last_reset is not sampled_event:
                        continue

                    _dlogger.debug("  Processing RESET PHASE event %s", sampled_event)
                    if signature == "initial_reset_phase":
                        if start > current_time:
                            current_time = self._advance_current_time(
                                current_time,
                                sampled_event,
                                deferred_function_calls,
                                loop_stack_generators,
                                signal_obj,
                            )
                            if awg.device_type.supports_reset_osc_phase:
                                deferred_function_calls["calls"].append("resetOscPhase")
                        else:
                            if awg.device_type.supports_reset_osc_phase:
                                loop_stack_generators[-1][
                                    -1
                                ].add_function_call_statement("resetOscPhase")
                    elif (
                        signature == "reset_phase"
                        and awg.device_type.supports_reset_osc_phase
                    ):
                        current_time = self._advance_current_time(
                            current_time,
                            sampled_event,
                            deferred_function_calls,
                            loop_stack_generators,
                            signal_obj,
                        )
                        deferred_function_calls["calls"].append("resetOscPhase")

                elif signature == "set_oscillator_frequency":
                    iteration = sampled_event["iteration"]
                    parameter_name = sampled_event["parameter_name"]
                    counter_variable_name = string_sanitize(f"index_{parameter_name}")
                    if iteration == 0:
                        if counter_variable_name != f"index_{parameter_name}":
                            _logger.warning(
                                "Parameter name '%s' has been sanitized in generated code.",
                                parameter_name,
                            )
                        declarations_generator.add_variable_declaration(
                            counter_variable_name, 0
                        )
                        declarations_generator.add_function_call_statement(
                            "configFreqSweep",
                            (
                                0,
                                sampled_event["start_frequency"],
                                sampled_event["step_frequency"],
                            ),
                        )
                        loop_stack_generators[-1][-1].add_variable_assignment(
                            counter_variable_name, 0
                        )
                    current_time = self._advance_current_time(
                        current_time,
                        sampled_event,
                        deferred_function_calls,
                        loop_stack_generators,
                        signal_obj,
                    )
                    deferred_function_calls["calls"].append(
                        {
                            "name": "setSweepStep",
                            "args": (0, f"{counter_variable_name}++"),
                        }
                    )

                elif signature == "LOOP_STEP_START":
                    _dlogger.debug(
                        "  Processing LOOP_STEP_START EVENT %s", sampled_event
                    )
                    current_time = self._advance_current_time(
                        current_time,
                        sampled_event,
                        deferred_function_calls,
                        loop_stack_generators,
                        signal_obj,
                    )
                    self._clear_deferred_function_calls(
                        deferred_function_calls, sampled_event, loop_stack_generators
                    )
                    if loop_stack_generators[-1][-1].num_statements() > 0:
                        loop_stack_generators[-1].append(SeqCGenerator())

                elif signature == "PUSH_LOOP":
                    _dlogger.debug(
                        "  Processing PUSH_LOOP EVENT %s, top of stack is %s",
                        sampled_event,
                        loop_stack_generators[-1][-1],
                    )
                    current_time = self._advance_current_time(
                        current_time,
                        sampled_event,
                        deferred_function_calls,
                        loop_stack_generators,
                        signal_obj,
                    )
                    self._clear_deferred_function_calls(
                        deferred_function_calls, sampled_event, loop_stack_generators
                    )

                    loop_stack_generators.append([SeqCGenerator()])
                    if CodeGenerator.EMIT_TIMING_COMMENTS:
                        loop_stack_generators[-1][-1].add_comment(
                            f"PUSH LOOP {sampled_event} current time = {current_time}"
                        )

                    loop_stack.append(sampled_event)

                elif signature == "ITERATE":
                    if loop_stack_generators[-1][-1].num_noncomment_statements() > 0:
                        _dlogger.debug(
                            "  Processing ITERATE EVENT %s, loop stack is %s",
                            sampled_event,
                            loop_stack,
                        )
                        if CodeGenerator.EMIT_TIMING_COMMENTS:
                            loop_stack_generators[-1][-1].add_comment(
                                f"ITERATE  {sampled_event}, current time = {current_time}"
                            )
                        current_time = self._advance_current_time(
                            current_time,
                            sampled_event,
                            deferred_function_calls,
                            loop_stack_generators,
                            signal_obj,
                        )
                        self._clear_deferred_function_calls(
                            deferred_function_calls,
                            sampled_event,
                            loop_stack_generators,
                        )
                        variable_name = string_sanitize(
                            "repeat_count_" + str(sampled_event["loop_id"])
                        )
                        if variable_name not in declared_variables:
                            declarations_generator.add_variable_declaration(
                                variable_name
                            )
                            declared_variables.add(variable_name)

                        # loop_stack_generators[-1][-1].add_comment(f"ITERATE")

                        loop_generator = SeqCGenerator()
                        open_generators = loop_stack_generators.pop()
                        _dlogger.debug(
                            "  Popped %s, stack is now %s",
                            open_generators,
                            loop_stack_generators,
                        )
                        loop_body = SeqCGenerator.compress_generators(
                            open_generators, declarations_generator
                        )
                        loop_generator.add_countdown_loop(
                            variable_name, sampled_event["num_repeats"], loop_body
                        )
                        if CodeGenerator.EMIT_TIMING_COMMENTS:
                            loop_generator.add_comment(f"Loop for {sampled_event}")
                        start_loop_event = loop_stack.pop()
                        delta = start - start_loop_event["start"]
                        current_time = (
                            start_loop_event["start"]
                            + sampled_event["num_repeats"] * delta
                        )
                        if CodeGenerator.EMIT_TIMING_COMMENTS:
                            loop_generator.add_comment(
                                f"Delta: {delta} current time after loop: {current_time}, corresponding start event: {start_loop_event}"
                            )
                        loop_stack_generators[-1].append(loop_generator)
                        loop_stack_generators[-1].append(SeqCGenerator())
                    else:
                        loop_stack_generators.pop()
                        loop_stack.pop()
                last_event = sampled_event

        self._clear_deferred_function_calls(
            deferred_function_calls, None, loop_stack_generators
        )
        seq_c_generators = []
        _dlogger.debug(
            "***  Finished event processing, loop_stack_generators: %s",
            loop_stack_generators,
        )
        for part in loop_stack_generators:
            for generator in part:
                seq_c_generators.append(generator)
        _dlogger.debug(
            "***  collected generators, seq_c_generators: %s", seq_c_generators
        )

        main_generator = SeqCGenerator.compress_generators_rle(
            seq_c_generators, declarations_generator
        )

        if main_generator.needs_play_zero_counter():
            declarations_generator.add_variable_declaration(
                main_generator.play_zero_counter_variable_name()
            )
        seq_c_generator = SeqCGenerator()
        seq_c_generator.append_statements_from(declarations_generator)
        seq_c_generator.append_statements_from(main_generator)

        if signal_obj.trigger_mode == TriggerMode.DIO_TRIGGER and awg.awg_number == 0:
            seq_c_generator.add_function_call_statement("setDIO", ["0"])

        seq_c_text = seq_c_generator.generate_seq_c()

        for line in seq_c_text.splitlines():
            _dlogger.debug(line)

        self._src.append({"filename": filename, "text": seq_c_text})
        self._wave_indices_all.append(
            {
                "filename": os.path.splitext(filename)[0] + "_waveindices.csv",
                "value": wave_indices.wave_indices(),
            }
        )

    def _analyze_loop_times(
        self, awg: AWGInfo, events: List[Any], sampling_rate: float, delay: float,
    ) -> SortedDict:
        retval = SortedDict()

        plays_anything = False
        signal_ids = [signal_obj.id for signal_obj in awg.signals]
        for e in events:
            if (
                e["event_type"] in ["PLAY_START", "ACQUIRE_START"]
                and e.get("signal") in signal_ids
            ):
                plays_anything = True
                break

        if plays_anything:
            _logger.debug(
                "Analyzing loop events for awg %d of %s", awg.awg_number, awg.device_id,
            )
        else:
            _logger.debug(
                "Skipping analysis of loop events for awg %d of %s because nothing is played",
                awg.awg_number,
                awg.device_id,
            )
            return retval

        loop_step_start_events = [
            event for event in events if event["event_type"] == "LOOP_STEP_START"
        ]
        loop_step_end_events = [
            event for event in events if event["event_type"] == "LOOP_STEP_END"
        ]
        loop_iteration_events = [
            event
            for event in events
            if event["event_type"] == "LOOP_ITERATION_END"
            and "compressed" in event
            and event["compressed"]
        ]

        compressed_sections = set(
            [event["section_name"] for event in loop_iteration_events]
        )
        section_repeats = {}
        for event in loop_iteration_events:
            if "num_repeats" in event:
                section_repeats[event["section_name"]] = event["num_repeats"]

        _dlogger.debug("Found %d loop step start events", len(loop_step_start_events))
        events_already_added = set()
        for event in loop_step_start_events:
            _dlogger.debug("  loop timing: processing  %s", event)

            event_time_in_samples = length_to_samples(
                event["time"] + delay, sampling_rate
            )
            loop_event = {
                "signature": "LOOP_STEP_START",
                "nesting_level": event["nesting_level"],
                "loop_id": event["section_name"],
                "start": event_time_in_samples,
                "end": event_time_in_samples,
            }

            if event["section_name"] not in compressed_sections:
                frozen = frozenset(loop_event.items())
                if frozen not in events_already_added:
                    self._add_to_dict_list(retval, event_time_in_samples, loop_event)
                    events_already_added.add(frozen)
                    _dlogger.debug("Added %s", loop_event)
                else:
                    _dlogger.debug("SKIP adding double %s", loop_event)
            elif event["iteration"] == 0:
                push_event = {
                    "signature": "PUSH_LOOP",
                    "nesting_level": event["nesting_level"],
                    "start": event_time_in_samples,
                    "end": event_time_in_samples,
                    "loop_id": event["section_name"],
                }
                if event["section_name"] in section_repeats:
                    push_event["num_repeats"] = section_repeats[event["section_name"]]
                self._add_to_dict_list(retval, event_time_in_samples, push_event)
                _dlogger.debug("Added %s", push_event)

        _dlogger.debug("Found %d loop step end events", len(loop_step_end_events))
        for event in loop_step_end_events:
            event_time_in_samples = length_to_samples(
                event["time"] + delay, sampling_rate
            )
            loop_event = {
                "signature": "LOOP_STEP_END",
                "nesting_level": event["nesting_level"],
                "loop_id": event["section_name"],
                "start": event_time_in_samples,
                "end": event_time_in_samples,
            }

            if event["section_name"] not in compressed_sections:
                frozen = frozenset(loop_event.items())
                if frozen not in events_already_added:
                    self._add_to_dict_list(retval, event_time_in_samples, loop_event)
                    events_already_added.add(frozen)
                    _dlogger.debug("Added %s", loop_event)
                else:
                    _dlogger.debug("SKIP adding double %s", loop_event)

        for event in loop_iteration_events:
            event_time_in_samples = length_to_samples(
                event["time"] + delay, sampling_rate
            )
            iteration_event = {
                "signature": "ITERATE",
                "nesting_level": event["nesting_level"],
                "start": event_time_in_samples,
                "end": event_time_in_samples,
                "loop_id": event["section_name"],
            }

            if event["section_name"] in section_repeats:
                iteration_event["num_repeats"] = section_repeats[event["section_name"]]

            self._add_to_dict_list(retval, event_time_in_samples, iteration_event)
            _dlogger.debug("Added %s", iteration_event)

            if event_time_in_samples % awg.device_type.sample_multiple != 0:
                _logger.warning(
                    "Event %s: event_time_in_samples %f at sampling rate %s is not divisible by %d",
                    event,
                    event_time_in_samples,
                    EngNumber(sampling_rate),
                    awg.device_type.sample_multiple,
                )

        return retval

    def _analyze_init_times(
        self, device_id: str, sampling_rate: float, delay: float
    ) -> SortedDict:
        _dlogger.debug("Analyzing init events for device %s", device_id)
        retval = SortedDict()
        delay_samples = length_to_samples(delay, sampling_rate)
        self._add_to_dict_list(
            retval,
            delay_samples,
            {
                "device_id": device_id,
                "start": delay_samples,
                "signature": "sequencer_start",
                "end": delay_samples,
            },
        )
        return retval

    def _analyze_phase_reset_times(
        self, events: List[Any], device_id: str, sampling_rate: float, delay: float
    ):
        retval = SortedDict()
        reset_phase_events = [
            event
            for event in events
            if event["event_type"]
            in (
                EventType.RESET_HW_OSCILLATOR_PHASE,
                EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
            )
            and "device_id" in event
            and event["device_id"] == device_id
        ]
        for event in reset_phase_events:
            event_time_in_samples = length_to_samples(
                event["time"] + delay, sampling_rate
            )
            signature = (
                "reset_phase"
                if event["event_type"] == EventType.RESET_HW_OSCILLATOR_PHASE
                else "initial_reset_phase"
            )
            init_event = {
                "start": event_time_in_samples,
                "signature": signature,
                "end": event_time_in_samples,
                "device_id": device_id,
            }

            self._add_to_dict_list(retval, event_time_in_samples, init_event)
        return retval

    def _analyze_set_oscillator_times(
        self,
        events: List,
        signal_id: str,
        device_id: str,
        device_type: DeviceType,
        sampling_rate: float,
        delay: float,
    ) -> SortedDict:

        set_oscillator_events = [
            event
            for event in events
            if event["event_type"] == "SET_OSCILLATOR_FREQUENCY_START"
            and event.get("device_id") == device_id
            and event.get("signal") == signal_id
        ]
        if len(set_oscillator_events) == 0:
            return SortedDict()

        if device_type not in (DeviceType.SHFQA, DeviceType.SHFSG):
            raise LabOneQException(
                "Real-time frequency sweep only supported on SHF devices"
            )

        iterations = {event["iteration"]: event for event in set_oscillator_events}
        assert list(iterations.keys()) == list(
            range(len(iterations))
        )  # "iteration" values are unique, ordered, and numbered 0 .. N-1
        start_frequency = iterations[0]["value"]
        step_frequency = iterations[1]["value"] - start_frequency

        retval = SortedDict()

        for iteration, event in iterations.items():
            if (
                abs(event["value"] - iteration * step_frequency - start_frequency)
                > 1e-3  # tolerance: 1 mHz
            ):
                raise LabOneQException("Realtime oscillator sweeps must be linear")

            event_time_in_samples = length_to_samples(
                event["time"] + delay, sampling_rate
            )
            set_oscillator_event = {
                "start": event_time_in_samples,
                "start_frequency": start_frequency,
                "step_frequency": step_frequency,
                "signature": "set_oscillator_frequency",
                "parameter_name": event["parameter"]["id"],
                "iteration": iteration,
            }

            self._add_to_dict_list(retval, event_time_in_samples, set_oscillator_event)

        return retval

    def _analyze_acquire_times(
        self,
        events: List[Any],
        signal_id: str,
        sampling_rate: float,
        delay: float,
        sample_multiple: int,
        channels,
    ) -> SortedDict:

        retval = SortedDict()

        signal_offset = 0

        for event in events:
            if (
                "signal_offset" in event
                and event["signal"] == signal_id
                and event["signal_offset"] is not None
            ):
                signal_offset = event["signal_offset"]
                _dlogger.debug(
                    "Found signal offset %f in event %s", signal_offset, event
                )

        _dlogger.debug(
            "Calculating acquire times for signal %s with delay %f and signal offset %f",
            signal_id,
            delay,
            signal_offset,
        )

        @dataclass
        class IntervalStartEvent:
            event_type: str
            time: float
            play_wave_id: str
            acquisition_type: list

        @dataclass
        class IntervalEndEvent:
            event_type: str
            time: float
            play_wave_id: str

        interval_zip = list(
            zip(
                [
                    IntervalStartEvent(
                        event["event_type"],
                        event["time"] + delay - signal_offset,
                        event["play_wave_id"],
                        event.get("acquisition_type", []),
                    )
                    for event in events
                    if event["event_type"] in ["ACQUIRE_START"]
                    and event["signal"] == signal_id
                ],
                [
                    IntervalEndEvent(
                        event["event_type"],
                        event["time"] + delay - signal_offset,
                        event["play_wave_id"],
                    )
                    for event in events
                    if event["event_type"] in ["ACQUIRE_END"]
                    and event["signal"] == signal_id
                ],
            )
        )

        for interval_start, interval_end in interval_zip:
            start_samples, end_samples = interval_to_samples(
                interval_start.time, interval_end.time, sampling_rate
            )
            if start_samples % sample_multiple != 0:
                start_samples = round(start_samples / sample_multiple) * sample_multiple

            if end_samples % sample_multiple != 0:
                end_samples = round(end_samples / sample_multiple) * sample_multiple

            acquire_event = {
                "start": start_samples,
                "signature": "acquire",
                "end": end_samples,
                "signal_id": signal_id,
                "play_wave_id": interval_start.play_wave_id,
                "acquisition_type": interval_start.acquisition_type,
                "channels": channels,
            }

            self._add_to_dict_list(retval, acquire_event["start"], acquire_event)

        return retval

    def waveform_size_hints(self, device: DeviceType):
        settings = self._compiler_settings
        if device == DeviceType.HDAWG:
            return settings.HDAWG_MIN_PLAYWAVE_HINT, settings.HDAWG_MIN_PLAYZERO_HINT
        if device == DeviceType.UHFQA:
            return settings.UHFQA_MIN_PLAYWAVE_HINT, settings.UHFQA_MIN_PLAYZERO_HINT
        if device == DeviceType.SHFQA:
            return settings.SHFQA_MIN_PLAYWAVE_HINT, settings.SHFQA_MIN_PLAYZERO_HINT
        if device == DeviceType.SHFSG:
            return settings.SHFSG_MIN_PLAYWAVE_HINT, settings.SHFSG_MIN_PLAYZERO_HINT

    def analyze_play_wave_times(
        self,
        events: List[Dict],
        signal_ids: List[str],
        signal_obj: SignalObj,
        other_events: Dict,
        iq_phase: float,
        sub_channel: Optional[int] = None,
    ):
        if len(events) == 0:
            return SortedDict()
        sampling_rate = signal_obj.sampling_rate
        delay = signal_obj.delay
        sample_multiple = signal_obj.device_type.sample_multiple
        min_play_wave = signal_obj.device_type.min_play_wave
        play_wave_size_hint, play_zero_size_hint = self.waveform_size_hints(
            signal_obj.device_type
        )
        signal_id = "_".join(signal_ids)
        for k, v in other_events.items():
            _dlogger.debug("Signal %s other event %s %s", signal_id, k, v)

        if sub_channel is not None:
            _dlogger.debug("Signal %s: using sub_channel = %s", signal_id, sub_channel)

        @dataclass
        class IntervalStartEvent:
            event_type: str
            time: float
            play_wave_id: str
            amplitude: float
            index: int
            oscillator_phase: Optional[float]
            phase: Optional[float]
            sub_channel: Optional[int]
            baseband_phase: Optional[float]

        @dataclass
        class IntervalEndEvent:
            event_type: str
            time: float
            play_wave_id: str
            index: int

        @dataclass
        class IntervalData:
            pulse: str
            index: int
            amplitude: float
            channel: int
            oscillator_phase: float
            baseband_phase: float
            phase: float
            sub_channel: int

        interval_zip = []
        for index, cur_signal_id in enumerate(signal_ids):
            interval_zip.extend(
                zip(
                    [
                        IntervalStartEvent(
                            event["event_type"],
                            event["time"] + delay,
                            event["play_wave_id"],
                            event["amplitude"],
                            index,
                            event.get("oscillator_phase"),
                            event.get("phase"),
                            sub_channel,
                            event.get("baseband_phase"),
                        )
                        for event in events
                        if event["event_type"] in ["PLAY_START"]
                        and event["signal"] == cur_signal_id
                    ],
                    [
                        IntervalEndEvent(
                            event["event_type"],
                            event["time"] + delay,
                            event["play_wave_id"],
                            index,
                        )
                        for event in events
                        if event["event_type"] in ["PLAY_END"]
                        and event["signal"] == cur_signal_id
                    ],
                )
            )
        if len(interval_zip) > 0:
            _logger.debug(
                "Analyzing play wave timings for %d play wave events on signals %s",
                len(interval_zip),
                signal_ids,
            )

        for ivzip in interval_zip:
            _dlogger.debug("Signal %s interval zip: %s", signal_id, ivzip)

        t = IntervalTree()
        PHASE_RESOLUTION_RANGE = CodeGenerator.phase_resolution_range()
        for index, (interval_start, interval_end) in enumerate(interval_zip):
            oscillator_phase = None
            if interval_start.oscillator_phase is not None:
                oscillator_phase = math.floor(
                    CodeGenerator.normalize_phase(interval_start.oscillator_phase)
                    / 2
                    / math.pi
                    * PHASE_RESOLUTION_RANGE
                )

            baseband_phase = None
            if interval_start.baseband_phase is not None:
                baseband_phase = math.floor(
                    CodeGenerator.normalize_phase(interval_start.baseband_phase)
                    / 2
                    / math.pi
                    * PHASE_RESOLUTION_RANGE
                )
            start_samples, end_samples = interval_to_samples(
                interval_start.time, interval_end.time, sampling_rate
            )
            if start_samples != end_samples:
                t.addi(
                    start_samples,
                    end_samples,
                    IntervalData(
                        pulse=interval_start.play_wave_id,
                        index=index,
                        amplitude=interval_start.amplitude,
                        channel=interval_start.index,
                        oscillator_phase=oscillator_phase,
                        baseband_phase=baseband_phase,
                        phase=interval_start.phase,
                        sub_channel=interval_start.sub_channel,
                    ),
                )
            else:
                _dlogger.debug(
                    "Skipping interval %s because it is zero length (from %s samples to %s samples) ",
                    interval_start.play_wave_id,
                    start_samples,
                    end_samples,
                )

        for ivs in sorted(t.items()):
            _dlogger.debug("Signal %s intervaltree:%s", signal_id, ivs)

        cut_points = set()
        for event_time, other_event in other_events.items():
            intervals = [
                interval for interval in t.at(event_time) if interval.begin < event_time
            ]

            if len(intervals) > 0:
                raise RuntimeError(
                    f"Event {other_event} intersects playWave intervals {intervals}",
                )
            else:
                cut_points.add(event_time)

        sequence_end = length_to_samples(events[-1]["time"] + delay, sampling_rate)
        sequence_end += play_wave_size_hint + play_zero_size_hint  # slack
        sequence_end += (-sequence_end) % sample_multiple  # align to sequencer grid
        cut_points.add(sequence_end)
        cut_points = sorted(list(cut_points))

        _logger.debug(
            "Collecting pulses to ensure waveform lengths are above the minimum of %d "
            "samples and are a multiple of %d samples for signal %s",
            min_play_wave,
            sample_multiple,
            signal_id,
        )

        try:
            compacted_intervals = calculate_intervals(
                t,
                min_play_wave,
                play_wave_size_hint,
                play_zero_size_hint,
                cut_points,
                granularity=sample_multiple,
            )
        except MinimumWaveformLengthViolation as e:
            raise LabOneQException(
                f"Failed to map the scheduled pulses to seqC without violating the "
                f"minimum waveform size {min_play_wave} of device "
                f"'{signal_obj.device_type.value}'.\n"
                f"Suggested workaround: manually add delays to overly short loops, etc."
            ) from e

        interval_events = SortedDict()

        signatures = set()

        _logger.debug("Calculating waveform signatures for signal %s", signal_id)

        for k in sorted(compacted_intervals.items()):
            _dlogger.debug("Calculating signature for %s and its children", k)

            overlap = t.overlap(k.begin, k.end)
            _dlogger.debug("Overlap is %s", overlap)

            v = sorted(t.overlap(k.begin, k.end))

            signature = [k.length()]
            has_child = False
            for iv in sorted(v, key=lambda x: (x.begin, x.data.channel)):
                data: IntervalData = iv.data
                _dlogger.debug("Calculating looking at child %s", iv)
                has_child = True
                start = iv.begin - k.begin
                end = iv.end - k.begin
                float_phase = data.phase
                if float_phase is not None:
                    float_phase = CodeGenerator.normalize_phase(float_phase)
                    int_phase = int(
                        CodeGenerator.PHASE_FIXED_SCALE * float_phase / 2 / math.pi
                    )
                else:
                    int_phase = None

                signature.append(
                    frozenset(
                        {
                            "start": start,
                            "end": end,
                            "pulse": data.pulse,
                            "pulse_samples": iv.length(),
                            "amplitude": data.amplitude,
                            "phase": int_phase,
                            "oscillator_phase": data.oscillator_phase,
                            "baseband_phase": data.baseband_phase,
                            "iq_phase": iq_phase,
                            "channel": data.channel if len(signal_ids) > 1 else None,
                            "sub_channel": data.sub_channel,
                        }.items()
                    )
                )
            if has_child:
                signatures.add(tuple(signature))
                start = k.begin
                interval_event = {
                    "start": start,
                    "signature": tuple(signature),
                    "end": k.end,
                    "signal_id": signal_id,
                }
                self._add_to_dict_list(interval_events, start, interval_event)

        if len(signatures) > 0:
            _logger.debug(
                "Signatures calculated: %d signatures for signal %s",
                len(signatures),
                signal_id,
            )
        for sig in signatures:
            _dlogger.debug(sig)
        _dlogger.debug("Interval events: %s", interval_events)

        return interval_events

    def _sample_pulses(
        self,
        signal_id,
        interval_events,
        pulse_defs,
        sampling_rate,
        signal_type,
        device_type,
        oscillator_frequency,
        oscillator_frequencies_per_channel=None,
        multi_iq_signal=False,
    ):
        PHASE_RESOLUTION_RANGE = CodeGenerator.phase_resolution_range()
        signatures = set()
        for interval_event_list in interval_events.values():
            for interval_event in interval_event_list:
                signature = interval_event["signature"]
                _dlogger.debug("Signature found %s in %s", signature, interval_event)
                signatures.add(signature)
        _dlogger.debug("Signatures: %s", signatures)
        sampled_signatures = {}
        max_amplitude = 0.0
        needs_conjugate = device_type == DeviceType.SHFSG
        for signature in signatures:
            length = signature[0]
            _dlogger.debug(
                "Sampling pulses for signature %s for signal %s, length %d device type %s",
                self._signature_string(signature),
                signal_id,
                length,
                device_type.value,
            )

            if length % device_type.sample_multiple != 0:
                raise Exception(
                    f"Length of signature {self._signature_string(signature)} is not divisible by {device_type.sample_multiple}, which it needs to be for {device_type.value}"
                )

            signature_pulse_map: Dict[str, PulseWaveformMap] = {}
            samples_i = np.zeros(length)
            samples_q = np.zeros(length)
            has_q = False
            for pulse_part in signature[1:]:
                signature_dict = dict(pulse_part)
                _dlogger.debug(" Sampling pulse part %s", signature_dict)
                pulse_def = pulse_defs[signature_dict["pulse"]]
                _dlogger.debug(" Pulse def: %s", pulse_def)

                sampling_signal_type = signal_type
                if signature_dict["channel"] is not None:
                    sampling_signal_type = "single"
                if multi_iq_signal:
                    sampling_signal_type = "iq"
                if signature_dict["sub_channel"] is not None:
                    sampling_signal_type = "iq"

                amplitude = 1.0
                if "amplitude" in pulse_def:
                    amplitude = pulse_def["amplitude"]

                amplitude_multiplier = 1.0
                if signature_dict["amplitude"] is not None:
                    amplitude_multiplier = signature_dict["amplitude"]

                if device_type == DeviceType.UHFQA and signal_type == "iq":
                    amplitude_multiplier *= math.sqrt(2)

                amplitude *= amplitude_multiplier

                if abs(amplitude) > max_amplitude:
                    max_amplitude = abs(amplitude)

                oscillator_phase = None
                if (
                    "oscillator_phase" in signature_dict
                    and signature_dict["oscillator_phase"] is not None
                ):
                    oscillator_phase = (
                        2
                        * math.pi
                        * signature_dict["oscillator_phase"]
                        / PHASE_RESOLUTION_RANGE
                    )

                baseband_phase = None
                if (
                    "baseband_phase" in signature_dict
                    and signature_dict["baseband_phase"] is not None
                ):
                    baseband_phase = (
                        2
                        * math.pi
                        * signature_dict["baseband_phase"]
                        / PHASE_RESOLUTION_RANGE
                    )

                used_oscillator_frequency = oscillator_frequency
                if (
                    oscillator_frequencies_per_channel is not None
                    and signature_dict["channel"] is not None
                ):
                    used_oscillator_frequency = oscillator_frequencies_per_channel[
                        signature_dict["channel"]
                    ]
                _dlogger.debug(
                    " Sampling pulse %s using oscillator frequency %s",
                    signature_dict,
                    used_oscillator_frequency,
                )

                if used_oscillator_frequency and device_type == DeviceType.SHFSG:
                    amplitude /= math.sqrt(2)
                    amplitude_multiplier /= math.sqrt(2)

                iq_phase = signature_dict.get("iq_phase", 0.0)
                if iq_phase is None:
                    iq_phase = 0.0

                # In case oscillator phase can't be set at runtime (e.g. HW oscillator without
                # phase control from a sequencer), apply oscillator phase on a baseband (iq) signal
                if baseband_phase is not None:
                    # negation is necessary to get the right behavior, same as with "phase" below
                    iq_phase -= baseband_phase

                if signature_dict.get("phase") is not None:
                    float_phase = (
                        float(signature_dict.get("phase") * math.pi * 2)
                        / CodeGenerator.PHASE_FIXED_SCALE
                    ) % (2 * math.pi)
                    # According to "QCCS Software: Signal, channel and oscillator concept" REQ 1.3
                    iq_phase -= float_phase % (2 * math.pi)

                iq_phase = CodeGenerator.normalize_phase(iq_phase)

                samples = pulse_def.get("samples")

                complex_modulation = True
                if device_type == DeviceType.UHFQA:
                    complex_modulation = False

                sampled_pulse = sample_pulse(
                    signal_type=sampling_signal_type,
                    sampling_rate=sampling_rate,
                    length=signature_dict["pulse_samples"] / sampling_rate,
                    amplitude=amplitude,
                    pulse_function=pulse_def["function"],
                    modulation_frequency=used_oscillator_frequency,
                    modulation_phase=oscillator_phase,
                    iq_phase=iq_phase,
                    samples=samples,
                    complex_modulation=complex_modulation,
                )

                if "samples_q" in sampled_pulse and len(
                    sampled_pulse["samples_i"]
                ) != len(sampled_pulse["samples_q"]):
                    _logger.warning(
                        "Expected samples_q and samples_i to be of equal length"
                    )
                len_i = len(sampled_pulse["samples_i"])
                if not len_i == signature_dict["pulse_samples"] and samples is None:
                    num_samples = length_to_samples(pulse_def["length"], sampling_rate)
                    _logger.warning(
                        "Pulse part %s: Expected %d samples but got %d; length = %f num samples=%d length in samples=%d",
                        signature_dict,
                        signature_dict["pulse_samples"],
                        len_i,
                        pulse_def["length"],
                        num_samples,
                        pulse_def["length"] * sampling_rate,
                    )
                    raise Exception("Len mismatch")

                if (
                    signature_dict["channel"] == 0
                    and not multi_iq_signal
                    and not device_type == DeviceType.SHFQA
                ):
                    CodeGenerator.stencil_samples(
                        signature_dict["start"], sampled_pulse["samples_i"], samples_i
                    )
                    has_q = True
                elif (
                    signature_dict["channel"] == 1
                    and not multi_iq_signal
                    and not device_type == DeviceType.SHFQA
                ):
                    CodeGenerator.stencil_samples(
                        signature_dict["start"], sampled_pulse["samples_i"], samples_q
                    )
                    has_q = True
                else:
                    CodeGenerator.stencil_samples(
                        signature_dict["start"], sampled_pulse["samples_i"], samples_i
                    )
                    if "samples_q" in sampled_pulse:
                        CodeGenerator.stencil_samples(
                            signature_dict["start"],
                            sampled_pulse["samples_q"],
                            samples_q,
                        )
                        has_q = True

                pm = signature_pulse_map.get(pulse_def["id"])
                if pm is None:
                    pm = PulseWaveformMap(
                        sampling_rate=sampling_rate,
                        length_samples=signature_dict["pulse_samples"],
                        signal_type=sampling_signal_type,
                        complex_modulation=complex_modulation,
                    )
                    signature_pulse_map[pulse_def["id"]] = pm
                pm.instances.append(
                    PulseInstance(
                        offset_samples=signature_dict["start"],
                        amplitude=amplitude_multiplier,
                        modulation_frequency=used_oscillator_frequency,
                        modulation_phase=oscillator_phase,
                        iq_phase=iq_phase,
                        channel=signature_dict["channel"],
                        needs_conjugate=needs_conjugate,
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

            if max_amplitude > 1e-9:
                sampled_signatures[signature] = sampled_pulse_obj

        return sampled_signatures

    def _signature_string(self, signature):
        retval = "p_" + str(signature[0]).zfill(4)
        for pulse_entry in signature[1:]:
            retval += "_"
            pulse_entry_dict = dict(pulse_entry)
            retval += pulse_entry_dict["pulse"]
            for key, separator, scale, fill in (
                ("start", "_", 1, 2),
                ("amplitude", "a_", 1e9, 10),
                ("oscillator_phase", "_ph_", 1, 7),
                ("baseband_phase", "_bb_", 1, 7),
                ("channel", "_c_", 1, 0),
                ("sub_channel", "_sc_", 1, 0),
                ("phase", "_ap_", 1, 0),
            ):
                if pulse_entry_dict.get(key) is not None:
                    value = pulse_entry_dict.get(key)
                    sign = ""
                    if value < 0:
                        sign = "m"
                    retval += (
                        separator + sign + str(abs(round(scale * value))).zfill(fill)
                    )

        if len(retval) > 64:
            hashed_signature = hashlib.md5(retval.encode()).hexdigest()
            self._long_signatures[hashed_signature] = retval
            retval = hashed_signature

        return string_sanitize(retval)

    def waves(self):
        return self._waves

    def src(self):
        return self._src

    def wave_indices(self):
        return self._wave_indices_all

    def pulse_map(self) -> Dict[str, PulseMapEntry]:
        return self._pulse_map

    @staticmethod
    def sort_events(events):
        # For events that happen at the same sample, emit the play wave first, because it is asynchronous
        later = {
            "sequencer_start": -100,
            "initial_reset_phase": -4,
            "LOOP_STEP_START": -3,
            "PUSH_LOOP": -2,
            "reset_phase": -1,
            "acquire": 1,
            "ITERATE": 2,
        }
        sampled_event_list = sorted(
            events,
            key=lambda x: later[x["signature"]] if x["signature"] in later else 0,
        )

        return sampled_event_list

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
            _dlogger.debug(
                "Not inserting: %d:%d , %d:%d",
                source_start,
                source_end,
                target_start,
                target_end,
            )

    @staticmethod
    def normalize_phase(phase):
        if phase < 0:
            retval = phase + (int(-phase / 2 / math.pi) + 1) * 2 * math.pi
        else:
            retval = phase
        retval = retval % (2 * math.pi)
        return retval

    @staticmethod
    def phase_resolution_range():
        return round(math.pow(2, CodeGenerator.PHASE_RESOLUTION_BITS))

    @staticmethod
    def post_process_sampled_events(awg: AWGInfo, sampled_events):
        if awg.device_type == DeviceType.SHFQA:
            has_acquire = False
            for sampled_event_list in sampled_events.values():
                for x in sampled_event_list:
                    if x["signature"] == "acquire":
                        has_acquire = True
                        break

            for sampled_event_list in sampled_events.values():
                start = None
                end = None
                acquisition_types = set()
                play = []
                acquire = []
                for x in sampled_event_list:
                    if "signal_id" in x and x["signature"] != "acquire":
                        play.append(x)
                        start = x["start"]
                        end = x["end"] if end is None else max(end, x["end"])
                    elif x["signature"] == "acquire":
                        start = x["start"]
                        if "acquisition_type" in x:
                            acquisition_types.update(x["acquisition_type"])
                        acquire.append(x)
                        end = x["end"] if end is None else max(end, x["end"])
                if len(play) > 0 and len(acquire) == 0 and has_acquire:
                    _logger.warning("Problem:")
                    for log_event in sampled_event_list:
                        _logger.warning("  %s", log_event)
                    raise Exception(f"Play and acquire must happen at the same time")
                if len(play) > 0 or len(acquire) > 0:
                    end = (
                        round(end / DeviceType.SHFQA.sample_multiple)
                        * DeviceType.SHFQA.sample_multiple
                    )
                    qa_event = {
                        "signature": "QA_EVENT",
                        "acquire_events": acquire,
                        "play_events": play,
                        "start": start,
                        "end": end,
                        "acquisition_type": list(acquisition_types),
                    }

                    for to_delete in play + acquire:
                        sampled_event_list.remove(to_delete)
                    sampled_event_list.append(qa_event)
