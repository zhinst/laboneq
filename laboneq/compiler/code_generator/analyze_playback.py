# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import itertools
import logging
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from intervaltree import Interval, IntervalTree

from laboneq.compiler.code_generator.interval_calculator import (
    MinimumWaveformLengthViolation,
    MutableInterval,
    calculate_intervals,
)
from laboneq.compiler.code_generator.signatures import (
    PlaybackSignature,
    PulseSignature,
    WaveformSignature,
    reduce_signature_amplitude,
    reduce_signature_phase,
)
from laboneq.compiler.code_generator.utils import normalize_phase
from laboneq.compiler.common.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.play_wave_type import PlayWaveType
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    combine_pulse_parameters,
    interval_to_samples_with_errors,
    length_to_samples,
)

_logger = logging.getLogger(__name__)


@dataclass
class _IntervalStartEvent:
    event_type: str
    signal_id: str
    time: float
    play_wave_id: str | None
    amplitude: float
    index: int
    oscillator_phase: Optional[float]
    oscillator_frequency: Optional[float]
    phase: Optional[float]
    sub_channel: Optional[int]
    baseband_phase: Optional[float]
    play_pulse_parameters: Optional[Dict[str, Any]]
    pulse_pulse_parameters: Optional[Dict[str, Any]]
    state: Optional[int]
    markers: Optional[Any]


@dataclass
class _IntervalEndEvent:
    event_type: str
    time: float
    play_wave_id: str
    index: int


@dataclass
class _PlayIntervalData:
    pulse: str | None
    index: int
    signal_id: str
    amplitude: float
    channel: int
    oscillator_phase: float
    oscillator_frequency: float
    baseband_phase: float
    phase: float
    sub_channel: int
    play_pulse_parameters: Optional[Dict[str, Any]]
    pulse_pulse_parameters: Optional[Dict[str, Any]]
    state: int
    start_rounding_error: float
    markers: Any


def _analyze_branches(events, delay, sampling_rate, playwave_max_hint):
    """For feedback, the pulses in the branches create a single waveform which spans
    the whole duration of the section; also keep the state to be able to split
    later.
    """
    cut_points = set()
    branching_intervals: Dict[str, List[MutableInterval]] = {}
    states: Dict[str, int] = {}
    for ev in events:
        if ev["event_type"] == "SECTION_START":
            handle = ev.get("handle", None)
            user_register = ev.get("user_register", None)
            if handle is not None or user_register is not None:
                begin = length_to_samples(ev["time"] + delay, sampling_rate)
                # Add the command table interval boundaries as cut points
                cut_points.add(begin)
                branching_intervals.setdefault(ev["section_name"], []).append(
                    MutableInterval(
                        # Add min_play_wave samples for executing executeTableEntry at the
                        # right time; todo(JL): Use actual min_play_wave
                        begin=begin,
                        end=None,
                        data=(handle, ev["local"], user_register),
                    )
                )
            else:
                if "state" in ev:
                    states[ev["section_name"]] = ev["state"]
        if ev["event_type"] == "SECTION_END":
            if ev["section_name"] in branching_intervals:
                iv = branching_intervals[ev["section_name"]][-1]
                end = length_to_samples(ev["time"] + delay, sampling_rate)
                cut_points.add(end)
                iv.end = end
                if playwave_max_hint != 0 and iv.end - iv.begin > playwave_max_hint:
                    raise LabOneQException(
                        f"In section {ev['section_name']}, waveform length ls larger "
                        f"than the allowed {playwave_max_hint} samples."
                    )

    return branching_intervals, states, cut_points


def _interval_list(events, states, signal_ids, delay, sub_channel):
    """Compute a flat list of (start, stop) intervals for tall the playback events"""

    interval_zip: List[Tuple[_IntervalStartEvent, _IntervalEndEvent]] = []
    for state in itertools.chain(set(states.values()), (None,)):
        for index, cur_signal_id in enumerate(signal_ids):
            interval_zip.extend(
                zip(
                    [
                        _IntervalStartEvent(
                            event_type=event["event_type"],
                            signal_id=cur_signal_id,
                            time=event["time"] + delay,
                            play_wave_id=event["play_wave_id"],
                            amplitude=event["amplitude"],
                            index=index,
                            oscillator_phase=event.get("oscillator_phase"),
                            oscillator_frequency=event.get("oscillator_frequency"),
                            phase=event.get("phase"),
                            sub_channel=sub_channel,
                            baseband_phase=event.get("baseband_phase"),
                            play_pulse_parameters=event.get("play_pulse_parameters"),
                            pulse_pulse_parameters=event.get("pulse_pulse_parameters"),
                            state=states.get(event["section_name"], None),
                            markers=event.get("markers"),
                        )
                        for event in events
                        if event["event_type"] in ["PLAY_START"]
                        and event["signal"] == cur_signal_id
                        and states.get(event["section_name"], None) == state
                    ],
                    [
                        _IntervalEndEvent(
                            event_type=event["event_type"],
                            time=event["time"] + delay,
                            play_wave_id=event["play_wave_id"],
                            index=index,
                        )
                        for event in events
                        if event["event_type"] in ["PLAY_END"]
                        and event["signal"] == cur_signal_id
                        and states.get(event["section_name"], None) == state
                    ],
                )
            )
            interval_zip.extend(
                zip(
                    [
                        _IntervalStartEvent(
                            event_type="DELAY_START",
                            signal_id=cur_signal_id,
                            time=event["time"] + delay,
                            play_wave_id=None,
                            amplitude=None,
                            index=index,
                            oscillator_phase=None,
                            oscillator_frequency=None,
                            phase=None,
                            sub_channel=sub_channel,
                            baseband_phase=None,
                            play_pulse_parameters=None,
                            pulse_pulse_parameters=None,
                            state=states.get(event["section_name"], None),
                            markers=None,
                        )
                        for event in events
                        if (
                            event["event_type"] == "DELAY_START"
                            and event.get("play_wave_type")
                            == PlayWaveType.EMPTY_CASE.name
                        )
                        and event["signal"] == cur_signal_id
                        and states.get(event["section_name"], None) == state
                    ],
                    [
                        _IntervalEndEvent(
                            event_type="DELAY_END",
                            time=event["time"] + delay,
                            play_wave_id=None,
                            index=index,
                        )
                        for event in events
                        if (
                            event["event_type"] == "DELAY_END"
                            and event.get("play_wave_type")
                            == PlayWaveType.EMPTY_CASE.name
                        )
                        and event["signal"] == cur_signal_id
                        and states.get(event["section_name"], None) == state
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
        _logger.debug("Signals %s interval zip: %s", signal_ids, ivzip)

    return interval_zip


def _make_interval_tree(
    events, states, signal_ids, delay, sub_channel, sampling_rate
) -> IntervalTree:
    interval_zip = _interval_list(events, states, signal_ids, delay, sub_channel)

    interval_tree = IntervalTree()

    for index, (interval_start, interval_end) in enumerate(interval_zip):
        oscillator_phase = interval_start.oscillator_phase

        baseband_phase: Optional[float] = None
        if interval_start.baseband_phase is not None:
            baseband_phase = interval_start.baseband_phase
        (start_samples, end_samples), (
            start_rounding_error,
            _,
        ) = interval_to_samples_with_errors(
            interval_start.time, interval_end.time, sampling_rate
        )

        if start_samples != end_samples:
            interval_tree.addi(
                start_samples,
                end_samples,
                _PlayIntervalData(
                    pulse=interval_start.play_wave_id,
                    signal_id=interval_start.signal_id,
                    index=index,
                    amplitude=interval_start.amplitude,
                    channel=interval_start.index,
                    oscillator_phase=oscillator_phase,
                    oscillator_frequency=interval_start.oscillator_frequency,
                    baseband_phase=baseband_phase,
                    phase=interval_start.phase,
                    sub_channel=interval_start.sub_channel,
                    play_pulse_parameters=interval_start.play_pulse_parameters,
                    pulse_pulse_parameters=interval_start.pulse_pulse_parameters,
                    state=interval_start.state,
                    start_rounding_error=start_rounding_error,
                    markers=interval_start.markers,
                ),
            )

        else:
            _logger.debug(
                "Skipping interval %s because it is zero length (from %s samples to %s samples) ",
                interval_start.play_wave_id,
                start_samples,
                end_samples,
            )

    for ivs in sorted(interval_tree.items()):
        _logger.debug("Signal(s) %s intervaltree:%s", signal_ids, ivs)

    return interval_tree


def _oscillator_switch_cut_points(
    interval_tree: IntervalTree,
    signals: Dict[str, SignalObj],
    sample_multiple,
) -> Tuple[AWGSampledEventSequence, Set]:
    cut_points = set()

    osc_switch_events = AWGSampledEventSequence()
    hw_oscillators = {
        signal_id: signals[signal_id].hw_oscillator for signal_id in signals
    }
    hw_oscs_values = set(hw_oscillators.values())
    awg = next(iter(signals.values())).awg
    device_type = awg.device_type
    if device_type == DeviceType.HDAWG:
        if awg.signal_type == AWGSignalType.DOUBLE:
            # Skip for now. In double mode, 2 oscillators may (?) be active.
            # todo (PW): Do we support dual HW modulated RF signals?
            return osc_switch_events, cut_points
    if not device_type.supports_oscillator_switching:
        if len(hw_oscs_values) > 1:
            raise LabOneQException(
                f"Attempting to multiplex several HW-modulated signals "
                f"({', '.join(signals)}) on {device_type.value}, which does not "
                f"support oscillator switching."
            )
    if len(hw_oscs_values) <= 1:
        return osc_switch_events, cut_points

    if None in hw_oscs_values:
        missing_oscillator_signal = next(
            signal_id for signal_id, osc in hw_oscillators.items() if osc is None
        )
        del hw_oscillators[missing_oscillator_signal]
        other_signals = set(hw_oscillators.keys())
        raise LabOneQException(
            f"Attempting to multiplex HW-modulated signal(s) "
            f"({', '.join(other_signals)}) "
            f"with signal that is not HW modulated ({missing_oscillator_signal})."
        )

    osc_intervals = IntervalTree()
    for pulse_iv in interval_tree:
        oscillator = signals[pulse_iv.data.signal_id].hw_oscillator
        # if there were any pulses w/o HW modulator, we should have returned already
        assert oscillator is not None

        osc_intervals.addi(
            # Round down to sequencer grid
            pulse_iv.begin // sample_multiple * sample_multiple,
            pulse_iv.end,
            {
                "oscillator": oscillator,
                "signal": pulse_iv.data.signal_id,
            },
        )

    def reducer(a, b):
        if a["oscillator"] != b["oscillator"]:
            raise LabOneQException(
                f"Overlapping HW oscillators: "
                f"'{a['oscillator']}' on signal '{a['signal']}' and "
                f"'{b['oscillator']}' on signal '{b['signal']}'"
            )
        return a

    osc_intervals.merge_overlaps(reducer)

    oscillator_switch_events = AWGSampledEventSequence()
    for iv in osc_intervals:
        osc_switch_event = AWGEvent(
            type=AWGEventType.SWITCH_OSCILLATOR,
            start=iv.begin,
            params={"oscillator": iv.data["oscillator"], "signal": iv.data["signal"]},
        )
        oscillator_switch_events.add(iv.begin, osc_switch_event)
        cut_points.add(iv.begin)

    return oscillator_switch_events, cut_points


def _sequence_end(events, sampling_rate, sample_multiple, delay, waveform_size_hint):
    sequence_end = length_to_samples(
        max(event["time"] for event in events) + delay, sampling_rate
    )
    play_wave_size_hint, play_zero_size_hint = waveform_size_hint
    sequence_end += play_wave_size_hint + play_zero_size_hint  # slack
    sequence_end += (-sequence_end) % sample_multiple  # align to sequencer grid

    return sequence_end


def _oscillator_intervals(
    signals: Iterable[SignalObj],
    oscillator_switch_events: AWGSampledEventSequence,
    sequence_end,
):
    retval = IntervalTree()
    start = 0
    active_oscillator = next(iter(signals)).hw_oscillator
    for osc_event_time, osc_event in oscillator_switch_events.sequence.items():
        if start != osc_event_time:
            retval.addi(start, osc_event_time, {"oscillator": active_oscillator})
        active_oscillator = osc_event[0].params["oscillator"]
        start = osc_event_time
    if start != sequence_end:
        retval.addi(start, sequence_end, {"oscillator": active_oscillator})
    return retval


def _make_pulse_signature(pulse_iv: Interval, wave_iv: Interval, signal_ids: List[str]):
    data: _PlayIntervalData = pulse_iv.data
    _logger.debug("Looking at child %s", pulse_iv)
    start = pulse_iv.begin - wave_iv.begin
    phase = data.phase
    if phase is not None:
        phase = normalize_phase(phase)

    oscillator_frequency = data.oscillator_frequency
    oscillator_phase = data.oscillator_phase
    if oscillator_phase is not None:
        if oscillator_frequency is not None:
            correction = -data.start_rounding_error * oscillator_frequency * 2 * math.pi
            oscillator_phase += correction
        oscillator_phase = normalize_phase(oscillator_phase)

    baseband_phase = data.baseband_phase
    if baseband_phase is not None:
        baseband_phase = normalize_phase(baseband_phase)

    combined_pulse_parameters = combine_pulse_parameters(
        data.pulse_pulse_parameters, None, data.play_pulse_parameters
    )
    markers = data.markers
    signature = PulseSignature(
        start=start,
        pulse=data.pulse,
        length=pulse_iv.length(),
        amplitude=data.amplitude,
        phase=phase,
        oscillator_phase=oscillator_phase,
        oscillator_frequency=oscillator_frequency,
        baseband_phase=baseband_phase,
        channel=data.channel if len(signal_ids) > 1 else None,
        sub_channel=data.sub_channel,
        pulse_parameters=None
        if combined_pulse_parameters is None
        else frozenset(combined_pulse_parameters.items()),
        markers=None if not markers else tuple(frozenset(m.items()) for m in markers),
    )
    pulse_parameters = (
        frozenset((data.play_pulse_parameters or {}).items()),
        frozenset((data.pulse_pulse_parameters or {}).items()),
    )
    return signature, pulse_parameters


def _interval_start_after_oscillator_reset(
    events, signals, compacted_intervals: IntervalTree, delay, sampling_rate
):
    device_id = next(iter(signals.values())).awg.device_id

    osc_reset_event_time = [
        length_to_samples(event["time"] + delay, sampling_rate)
        for event in events
        if event["event_type"] == "RESET_HW_OSCILLATOR_PHASE"
        and event["device_id"] == device_id
    ]

    interval_start_times = sorted(iv.begin for iv in compacted_intervals)
    if len(interval_start_times) == 0:
        return set()
    return set(
        interval_start_times[
            bisect_left(
                interval_start_times,
                event,
            )
        ]
        for event in osc_reset_event_time
    )


def analyze_play_wave_times(
    events: List[Dict],
    signals: Dict[str, SignalObj],
    device_type: DeviceType,
    sampling_rate,
    delay: float,
    other_events: AWGSampledEventSequence,
    waveform_size_hints: Tuple[int, int],
    phase_resolution_range: int,
    sub_channel: Optional[int] = None,
    use_command_table: bool = False,
) -> AWGSampledEventSequence:
    signal_ids = list(signals.keys())
    sample_multiple = device_type.sample_multiple
    min_play_wave = device_type.min_play_wave
    if device_type == DeviceType.SHFQA:
        playwave_max_hint = 4096  # in integration mode, 4096 samples is the limit
    else:
        playwave_max_hint = 0  # 0 means no limit
    play_wave_size_hint, play_zero_size_hint = waveform_size_hints
    signal_id = "_".join(signal_ids)
    for k, v in other_events.sequence.items():
        _logger.debug("Signal %s other event %s %s", signal_id, k, v)

    if sub_channel is not None:
        _logger.debug("Signal %s: using sub_channel = %s", signal_id, sub_channel)

    if len(events) == 0:
        return AWGSampledEventSequence()

    branching_intervals, states, cut_points = _analyze_branches(
        events, delay, sampling_rate, playwave_max_hint
    )

    interval_tree = _make_interval_tree(
        events, states, signal_ids, delay, sub_channel, sampling_rate
    )

    use_ct_phase = use_command_table and all(s.hw_oscillator for s in signals.values())

    sequence_end = _sequence_end(
        events, sampling_rate, sample_multiple, delay, waveform_size_hints
    )
    cut_points.add(sequence_end)
    cut_points.update(other_events.sequence)

    (
        oscillator_switch_events,
        oscillator_switch_cut_points,
    ) = _oscillator_switch_cut_points(interval_tree, signals, sample_multiple)

    cut_points.update(oscillator_switch_cut_points)
    oscillator_intervals = _oscillator_intervals(
        signals.values(), oscillator_switch_events, sequence_end
    )
    used_oscillators = {osc_iv.data["oscillator"] for osc_iv in oscillator_intervals}
    if len(used_oscillators) > 1 and not use_command_table:
        raise LabOneQException(
            "HW oscillator switching only possible in command-table mode"
        )

    # Check whether any cut point is within a command table interval
    for cp in cut_points:
        for ivs in branching_intervals.values():
            for iv in ivs:
                assert not (iv.begin < cp < iv.end)

    cut_points = sorted(list(cut_points))

    _logger.debug(
        "Collecting pulses to ensure waveform lengths are above the minimum of %d "
        "samples and are a multiple of %d samples for signal %s",
        min_play_wave,
        sample_multiple,
        signal_id,
    )

    # Adapt intervals to obey min/max playwave constraints and hints, keeping the used
    # pulses of each interval
    try:
        compacted_intervals = calculate_intervals(
            interval_tree,
            min_play_wave,
            play_wave_size_hint,
            play_zero_size_hint,
            playwave_max_hint,
            cut_points,
            granularity=sample_multiple,
            force_command_table_intervals=[
                iv for ivs in branching_intervals.values() for iv in ivs
            ],
        )
    except MinimumWaveformLengthViolation as e:
        raise LabOneQException(
            f"Failed to map the scheduled pulses to SeqC without violating the "
            f"minimum waveform size {min_play_wave} of device "
            f"'{device_type.value}'.\n"
            f"Suggested workaround: manually add delays to overly short loops, etc."
        ) from e

    interval_events = AWGSampledEventSequence()

    signatures = set()

    # Add branching points as events
    for section_name, ivs in branching_intervals.items():
        for interval in ivs:
            interval_event = AWGEvent(
                type=AWGEventType.MATCH,
                start=interval.begin,
                end=interval.end,
                params={
                    "handle": interval.data[0],
                    "local": interval.data[1],
                    "user_register": interval.data[2],
                    "signal_id": signal_id,
                    "section_name": section_name,
                },
            )
            interval_events.add(interval.begin, interval_event)

    _logger.debug("Calculating waveform signatures for signal %s", signal_id)

    interval_start_after_oscillator_reset = _interval_start_after_oscillator_reset(
        events, signals, compacted_intervals, delay, sampling_rate
    )

    # When the sequencer starts, the phase of all sine generators is initialized to 0
    # (On HDAWG, we additionally use `setSinePhase()` to set the phase of the I channel to 90°)
    hw_oscillator_phases = {
        signal.hw_oscillator: 0.0
        for signal in signals.values()
        if signal.hw_oscillator is not None
    }

    k: Interval
    for k in sorted(compacted_intervals.items()):
        _logger.debug("Calculating signature for %s and its children", k)

        if k.begin in interval_start_after_oscillator_reset:
            hw_oscillator_phases = {}

        overlap: Set[Interval] = interval_tree.overlap(k.begin, k.end)
        _logger.debug("Overlap is %s", overlap)

        v = sorted(overlap)

        hw_oscillator_intervals: Set[Interval] = oscillator_intervals.overlap(k)
        assert len(hw_oscillator_intervals) == 1
        hw_oscillator = next(iter(hw_oscillator_intervals)).data["oscillator"]

        # Group by states for match/state sections
        v_state: Dict[int, List[Interval]] = {}
        for i in v:
            data: _PlayIntervalData = i.data
            v_state.setdefault(data.state, []).append(i)

        for state, intervals in v_state.items():
            signature_pulses = []
            pulse_parameters = []
            has_child = False
            for iv in sorted(intervals, key=lambda x: (x.begin, x.data.channel)):
                pulse_signature, these_pulse_parameters = _make_pulse_signature(
                    iv, k, signal_ids
                )
                signature_pulses.append(pulse_signature)
                pulse_parameters.append(these_pulse_parameters)
                has_child = True
            waveform_signature = WaveformSignature(k.length(), tuple(signature_pulses))

            if hw_oscillator is not None:
                current_hw_oscillator_phase = hw_oscillator_phases.get(hw_oscillator)
            else:
                current_hw_oscillator_phase = None

            if use_command_table and device_type == DeviceType.SHFSG:
                ct_hw_oscillator = hw_oscillator
            else:
                ct_hw_oscillator = None

            precomp_reset = any(
                pulse.pulse == "dummy_precomp_reset" for pulse in signature_pulses
            )

            signature = reduce_signature_phase(
                PlaybackSignature(
                    waveform=waveform_signature,
                    hw_oscillator=ct_hw_oscillator,
                    pulse_parameters=tuple(pulse_parameters),
                    state=state,
                    clear_precompensation=precomp_reset,
                ),
                use_ct_phase,
                current_hw_oscillator_phase,
            )
            if use_command_table:
                signature = reduce_signature_amplitude(signature)

            if hw_oscillator is not None:
                if signature.set_phase is not None:
                    hw_oscillator_phases[hw_oscillator] = signature.set_phase
                elif signature.increment_phase is not None:
                    hw_oscillator_phases[hw_oscillator] += signature.increment_phase

            if phase_resolution_range >= 1:
                signature.quantize_phase(phase_resolution_range)

            if has_child:
                signatures.add(signature)
                start = k.begin
                interval_event = AWGEvent(
                    type=AWGEventType.PLAY_WAVE,
                    start=start,
                    end=k.end,
                    params={
                        "playback_signature": signature,
                        "signal_id": signal_id,
                    },
                )
                interval_events.add(start, interval_event)

    if len(signatures) > 0:
        _logger.debug(
            "Signatures calculated: %d signatures for signal %s",
            len(signatures),
            signal_id,
        )
    for sig in signatures:
        _logger.debug(sig)
    _logger.debug("Interval events: %s", interval_events.sequence)

    return interval_events
