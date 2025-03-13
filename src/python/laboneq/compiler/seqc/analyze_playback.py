# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Iterable, List, Set, Tuple, TYPE_CHECKING
from laboneq._rust.intervals import Interval, IntervalTree
from laboneq.compiler.common.compiler_settings import TINYSAMPLE
from laboneq.compiler.seqc.analyze_amplitude_registers import (
    analyze_amplitude_register_set_events,
)
from laboneq.compiler.seqc.interval_calculator import (
    MinimumWaveformLengthViolation,
    MutableInterval,
    calculate_intervals,
)
from laboneq.compiler.seqc.signatures import (
    PlaybackSignature,
    PulseSignature,
    WaveformSignature,
    reduce_signature_amplitude,
    reduce_signature_amplitude_register,
    reduce_signature_phase,
)
from laboneq.compiler.seqc.utils import normalize_phase
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.event_list.event_type import EventType
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    combine_pulse_parameters,
    interval_to_samples_with_errors,
    length_to_samples,
)

if TYPE_CHECKING:
    from laboneq.compiler.seqc.passes.intervals import EmptyInterval

_logger = logging.getLogger(__name__)


@dataclass
class _IntervalStartEvent:
    event_type: str
    signal_id: str
    time: float
    play_wave_id: str | None
    amplitude: float
    index: int
    oscillator_phase: float | None
    oscillator_frequency: float | None
    phase: float | None
    sub_channel: int | None
    increment_oscillator_phase: float | None
    play_pulse_parameters: dict[str, Any] | None
    pulse_pulse_parameters: dict[str, Any] | None
    state: int | None
    markers: Any
    amp_param: str | None
    incr_phase_param: str | None


@dataclass
class _IntervalEndEvent:
    event_type: str
    time: float
    play_wave_id: str
    index: int


@dataclass
class _PlayIntervalData:
    pulse: str | None
    signal_id: str
    amplitude: float
    channel: int
    oscillator_phase: float
    oscillator_frequency: float
    increment_oscillator_phase: float
    phase: float
    sub_channel: int
    play_pulse_parameters: dict[str, Any] | None
    pulse_pulse_parameters: dict[str, Any] | None
    state: int
    start_rounding_error: float
    markers: Any
    amp_param: str | None
    incr_phase_param: str | None


@dataclass
class FeedbackIntervalData:
    handle: str | None
    local: bool | None
    user_register: int | None
    prng_sample: str | None


def _analyze_branches(
    events, delay, sampling_rate, playwave_max_hint
) -> tuple[dict[str, list[MutableInterval]], dict[str, int], set[int]]:
    """For feedback, the pulses in the branches create a single waveform which spans
    the whole duration of the section; also keep the state to be able to split
    later.
    """
    cut_points = set()
    branching_intervals = {}
    states = {}
    for ev in events:
        if ev["event_type"] == "SECTION_START":
            handle = ev.get("handle", None)
            user_register = ev.get("user_register", None)
            prng_sample = ev.get("prng_sample")
            if (
                handle is not None
                or user_register is not None
                or prng_sample is not None
            ):
                begin = length_to_samples(ev["time"] + delay, sampling_rate)
                # Add the command table interval boundaries as cut points
                cut_points.add(begin)
                branching_intervals.setdefault(ev["section_name"], []).append(
                    MutableInterval(
                        # Add min_play_wave samples for executing executeTableEntry at the
                        # right time; todo(JL): Use actual min_play_wave
                        begin=begin,
                        end=None,
                        data=FeedbackIntervalData(
                            handle, ev.get("local"), user_register, prng_sample
                        ),
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


def find_event_pairs(events, start_types, end_types):
    start_events = {}
    end_events = {}
    for index, event in enumerate(events):
        if event["event_type"] in start_types:
            start_events[event["chain_element_id"]] = index, event
        elif event["event_type"] in end_types:
            end_events[event["chain_element_id"]] = event

    return [
        (index, (start, end_events[id]))
        for (id, (index, start)) in start_events.items()
    ]


def _interval_list(events, states, signal_ids, delay, sub_channel):
    """Compute a flat list of (start, stop) intervals for all the playback events"""

    interval_zip: List[Tuple[_IntervalStartEvent, _IntervalEndEvent]] = []
    play_event_pairs = find_event_pairs(
        events, [EventType.PLAY_START], [EventType.PLAY_END]
    )
    interval_zip.extend(
        (
            (
                _IntervalStartEvent(
                    event_type=start["event_type"],
                    signal_id=cur_signal_id,
                    time=start["time"] + delay,
                    play_wave_id=start["play_wave_id"],
                    amplitude=start.get("amplitude", 1.0),
                    index=index,
                    oscillator_phase=start.get("oscillator_phase"),
                    oscillator_frequency=start.get("oscillator_frequency"),
                    phase=start.get("phase"),
                    sub_channel=sub_channel,
                    increment_oscillator_phase=start.get("increment_oscillator_phase"),
                    play_pulse_parameters=start.get("play_pulse_parameters"),
                    pulse_pulse_parameters=start.get("pulse_pulse_parameters"),
                    state=states.get(start["section_name"], None),
                    markers=start.get("markers"),
                    amp_param=start.get("amplitude_parameter"),
                    incr_phase_param=start.get("phase_increment_parameter"),
                ),
                _IntervalEndEvent(
                    event_type=end["event_type"],
                    time=end["time"] + delay,
                    play_wave_id=end["play_wave_id"],
                    index=index,
                ),
            )
            for state in itertools.chain(set(states.values()), (None,))
            for index, cur_signal_id in enumerate(signal_ids)
            for _, (start, end) in play_event_pairs
            if start["signal"] == cur_signal_id
            and states.get(start["section_name"], None) == state
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
    events: list[dict],
    states: dict,
    signal_ids: list[str],
    delay: float,
    sub_channel: int,
    sampling_rate: float,
    empty_intervals: list[EmptyInterval] | None = None,
) -> IntervalTree:
    interval_zip = _interval_list(events, states, signal_ids, delay, sub_channel)
    intervals = []
    if empty_intervals:
        for index, sig in enumerate(signal_ids):
            for branch in [iv for iv in empty_intervals if iv.signal == sig]:
                assert branch.start != branch.end, "Interval cannot be zero length"
                start = branch.start * TINYSAMPLE + delay
                end = branch.end * TINYSAMPLE + delay
                (
                    (start_samples, end_samples),
                    (start_rounding_error, _),
                ) = interval_to_samples_with_errors(start, end, sampling_rate)
                intervals.extend(
                    (
                        Interval(
                            start_samples,
                            end_samples,
                            _PlayIntervalData(
                                pulse=None,
                                signal_id=sig,
                                amplitude=None,
                                channel=index,
                                oscillator_phase=None,
                                oscillator_frequency=None,
                                increment_oscillator_phase=None,
                                phase=None,
                                sub_channel=sub_channel,
                                play_pulse_parameters=None,
                                pulse_pulse_parameters=None,
                                state=branch.state,
                                start_rounding_error=start_rounding_error,
                                markers=None,
                                amp_param=None,
                                incr_phase_param=None,
                            ),
                        ),
                    ),
                )
    for interval_start, interval_end in interval_zip:
        oscillator_phase = interval_start.oscillator_phase
        # Note: the scheduler will already align all the pulses with the sample grid,
        # and the rounding error should be 0 except for some floating point epsilon.
        (
            (start_samples, end_samples),
            (start_rounding_error, _),
        ) = interval_to_samples_with_errors(
            interval_start.time, interval_end.time, sampling_rate
        )

        if start_samples != end_samples:
            intervals.append(
                Interval(
                    start_samples,
                    end_samples,
                    _PlayIntervalData(
                        pulse=interval_start.play_wave_id,
                        signal_id=interval_start.signal_id,
                        amplitude=interval_start.amplitude,
                        channel=interval_start.index,
                        oscillator_phase=oscillator_phase,
                        oscillator_frequency=interval_start.oscillator_frequency,
                        increment_oscillator_phase=interval_start.increment_oscillator_phase,
                        phase=interval_start.phase,
                        sub_channel=interval_start.sub_channel,
                        play_pulse_parameters=interval_start.play_pulse_parameters,
                        pulse_pulse_parameters=interval_start.pulse_pulse_parameters,
                        state=interval_start.state,
                        start_rounding_error=start_rounding_error,
                        markers=interval_start.markers,
                        amp_param=interval_start.amp_param,
                        incr_phase_param=interval_start.incr_phase_param,
                    ),
                )
            )

        else:
            _logger.debug(
                "Skipping interval %s because it is zero length (from %s samples to %s samples) ",
                interval_start.play_wave_id,
                start_samples,
                end_samples,
            )
    return IntervalTree(intervals)


def _oscillator_phase_increment_points(
    events, signals, delay, sampling_rate
) -> set[int]:
    virtual_z_gates = _find_frame_changes(events, delay, sampling_rate)
    virtual_z_gates = [v for v in virtual_z_gates if v.signal in signals]

    frame_change_times = {frame_change.time for frame_change in virtual_z_gates}

    return frame_change_times


@dataclass
class _FrameChange:
    time: int
    phase: float | None
    parameter: str | None
    signal: str
    priority: int
    state: int | None = None


def _find_frame_changes(events, delay, sampling_rate):
    delays = find_event_pairs(events, [EventType.DELAY_START], [EventType.DELAY_END])
    virtual_z_gates = []
    for index, (delay_start, _) in delays:
        incr_phase = delay_start.get("increment_oscillator_phase")
        set_phase = delay_start.get("set_oscillator_phase")
        assert set_phase is None, "should have been dissolved in scheduler"
        if incr_phase is not None:
            time = delay_start["time"] + delay
            time_samples = length_to_samples(time, sampling_rate)
            virtual_z_gates.append(
                _FrameChange(
                    time=time_samples,
                    phase=incr_phase,
                    parameter=delay_start.get("phase_increment_parameter"),
                    signal=delay_start["signal"],
                    state=delay_start.get("state"),
                    priority=index,
                )
            )
    return virtual_z_gates


def _make_virtual_z_gate_event(frame_change: _FrameChange, signal: SignalObj):
    return AWGEvent(
        type=AWGEventType.CHANGE_OSCILLATOR_PHASE,
        start=frame_change.time,
        end=frame_change.time,
        priority=frame_change.priority,
        params={
            "signal": frame_change.signal,
            "phase": frame_change.phase,
            "oscillator": signal.hw_oscillator
            if signal.awg.device_type.supports_oscillator_switching
            else None,
            "parameter": frame_change.parameter,
        },
    )


def _insert_frame_changes(
    interval_events: AWGSampledEventSequence,
    events,
    delay,
    sampling_rate,
    signals: dict[str, SignalObj],
):
    frame_change_events = AWGSampledEventSequence()
    intervals = IntervalTree(
        [
            Interval(ev.start, ev.end, ev)
            for t, evl in interval_events.sequence.items()
            for ev in evl
            if ev.type == AWGEventType.PLAY_WAVE
        ]
    )

    frame_changes = _find_frame_changes(events, delay, sampling_rate)
    frame_changes = [fc for fc in frame_changes if fc.signal in signals]

    for frame_change in frame_changes:
        if frame_change.signal not in signals.keys():
            continue

        signal = signals[frame_change.signal]
        # check if the frame change overlaps a (compacted) interval
        play_ivs = [
            iv
            for iv in intervals.at(frame_change.time)
            if iv.data.params["playback_signature"].state == frame_change.state
        ]

        if len(play_ivs) == 0:
            if frame_change.state is not None:
                raise LabOneQException(
                    "Cannot handle free-standing oscillator phase change in conditional"
                    " branch.\n"
                    "A zero-length `increment_oscillator_phase` must not occur at the"
                    " very end of a `case` block. Consider stretching the branch by"
                    " adding a small delay after the phase increment."
                )
            frame_change_events.add(
                frame_change.time, _make_virtual_z_gate_event(frame_change, signal)
            )
            continue

        # mutate the signature of the play interval, inserting the phase increment
        [play_iv] = play_ivs

        signature = play_iv.data.params["playback_signature"]
        if (
            signal.hw_oscillator != signature.hw_oscillator
            and signature.hw_oscillator is not None
        ):
            # Oscillator conflict!
            # We may still get out of this, if the frame change happens at the beginning
            # of the interval, by emitting a 0-length command table entry.
            if frame_change.time == play_iv.data.start:
                frame_change_events.add(
                    frame_change.time, _make_virtual_z_gate_event(frame_change, signal)
                )
                continue

            raise LabOneQException(
                f"Cannot increment oscillator '{signal.hw_oscillator}' of signal"
                f" '{signal.id}': the line is occupied by '{signature.hw_oscillator}'."
            )
        assert signature.waveform is not None
        assert len(signature.waveform.pulses)

        # offset of the increment in the signature
        offset = frame_change.time - play_iv.data.start

        for pulse in signature.waveform.pulses:
            if pulse.start < offset:
                continue
            pulse.increment_oscillator_phase = (
                pulse.increment_oscillator_phase or 0.0
            ) + frame_change.phase
            pulse.incr_phase_params = tuple(
                list(pulse.incr_phase_params) + [frame_change.parameter]
            )
            break
        else:
            # Insert the phase increment into the last pulse anyway, and rewind its
            # own phase by the same amount.
            pulse = signature.waveform.pulses[-1]
            pulse.increment_oscillator_phase = pulse.increment_oscillator_phase or 0.0
            pulse.oscillator_phase = pulse.oscillator_phase or 0.0
            pulse.increment_oscillator_phase += frame_change.phase
            pulse.oscillator_phase -= frame_change.phase
            pulse.incr_phase_params = tuple(
                (list(pulse.incr_phase_params) or [None]) + [frame_change.parameter]
            )
    return frame_change_events


def _oscillator_switch_cut_points(
    interval_tree: IntervalTree,
    signals: Dict[str, SignalObj],
    sample_multiple,
    oscillator_phase_increment_times: set[int],
    branching_intervals: dict[str, list[MutableInterval]],
) -> Tuple[AWGSampledEventSequence, Set]:
    cut_points = set()

    osc_switch_events = AWGSampledEventSequence()
    hw_oscillators = {
        signal_id: signals[signal_id].hw_oscillator for signal_id in signals
    }
    hw_oscs_values = set(hw_oscillators.values())
    awg = next(iter(signals.values())).awg
    device_type = awg.device_type
    if awg.signal_type == AWGSignalType.DOUBLE:
        # Skip for now. In double mode, 2 oscillators may (?) be active.
        # Still, oscillator switching is not supported on these instruments.
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

    if interval_tree.is_empty():
        # Nothing to do if there are no pulses in an experiment.
        return osc_switch_events, cut_points

    osc_intervals_ = []
    for pulse_iv in interval_tree.intervals:
        oscillator = signals[pulse_iv.data.signal_id].hw_oscillator
        # if there were any pulses w/o HW modulator, we should have returned already
        assert oscillator is not None

        osc_intervals_.append(
            Interval(
                # Round down to sequencer grid
                pulse_iv.begin // sample_multiple * sample_multiple,
                pulse_iv.end,
                {
                    "oscillator": oscillator,
                    "signal": pulse_iv.data.signal_id,
                },
            )
        )

    osc_intervals = IntervalTree(osc_intervals_)

    for branching_interval_list in branching_intervals.values():
        for branching_interval in branching_interval_list:
            begin, end = branching_interval.begin, branching_interval.end
            if osc_intervals.overlaps_range(begin, end):
                osc_intervals.addi(begin, end, {})

    def reducer(a: dict[str, Any], b: dict[str, Any]):
        osc_a, osc_b = a.get("oscillator"), b.get("oscillator")
        if osc_a != osc_b and None not in (osc_a, osc_b):
            raise LabOneQException(
                f"Overlapping HW oscillators: "
                f"'{osc_a}' on signal '{a['signal']}' and "
                f"'{osc_b}' on signal '{b['signal']}'"
            )
        return a | b

    osc_intervals.merge_overlaps(reducer)

    for time in oscillator_phase_increment_times:
        # Check if the frame change happens after the last oscillator switch - the
        # compiler will otherwise assume that the last oscillator switch stays valid
        # to the end of the sequence.
        if time >= osc_intervals.end():
            cut_points.add(ceil(time / sample_multiple) * sample_multiple)

    oscillator_switch_events = AWGSampledEventSequence()
    for iv in osc_intervals.intervals:
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
) -> IntervalTree:
    intervals_ = []
    start = 0
    active_oscillator = next(iter(signals)).hw_oscillator
    for osc_event_time, osc_event in oscillator_switch_events.sequence.items():
        if start != osc_event_time:
            intervals_.append(
                Interval(start, osc_event_time, {"oscillator": active_oscillator})
            )
        active_oscillator = osc_event[0].params["oscillator"]
        start = osc_event_time
    if start != sequence_end:
        intervals_.append(
            Interval(start, sequence_end, {"oscillator": active_oscillator})
        )
    return IntervalTree(intervals_)


def _make_pulse_signature(
    pulse_iv: Interval,
    wave_iv: Interval,
    signal_ids: list[str],
    amplitude_registers: dict[str, int],
):
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

    increment_oscillator_phase = data.increment_oscillator_phase
    if increment_oscillator_phase:
        incr_phase_params = (data.incr_phase_param,)
    else:
        incr_phase_params = ()

    combined_pulse_parameters = combine_pulse_parameters(
        data.pulse_pulse_parameters, None, data.play_pulse_parameters
    )

    amplitude_register = amplitude_registers.get(data.amp_param, 0)

    markers = data.markers

    signature = PulseSignature(
        start=start,
        pulse=data.pulse,
        length=pulse_iv.length(),
        amplitude=data.amplitude,
        phase=phase,
        oscillator_phase=oscillator_phase,
        oscillator_frequency=oscillator_frequency,
        increment_oscillator_phase=increment_oscillator_phase,
        channel=data.channel if len(signal_ids) > 1 else None,
        sub_channel=data.sub_channel,
        pulse_parameters=None
        if combined_pulse_parameters is None
        else frozenset(combined_pulse_parameters.items()),
        markers=tuple(frozenset(m.items()) for m in markers) if markers else None,
        preferred_amplitude_register=amplitude_register,
        incr_phase_params=incr_phase_params,
    )
    pulse_parameters = (
        frozenset((data.play_pulse_parameters or {}).items()),
        frozenset((data.pulse_pulse_parameters or {}).items()),
    )
    return signature, pulse_parameters


def analyze_play_wave_times(
    events: List[Dict],
    signals: Dict[str, SignalObj],
    device_type: DeviceType,
    sampling_rate,
    delay: float,
    other_events: AWGSampledEventSequence,
    waveform_size_hints: Tuple[int, int],
    phase_resolution_range: int,
    amplitude_resolution_range: int,
    sub_channel: int | None = None,
    use_command_table: bool = False,
    use_amplitude_increment: bool = False,
    empty_intervals: list[EmptyInterval] | None = None,
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
        events,
        states,
        signal_ids,
        delay,
        sub_channel,
        sampling_rate,
        empty_intervals=empty_intervals,
    )

    sequence_end = _sequence_end(
        events, sampling_rate, sample_multiple, delay, waveform_size_hints
    )
    cut_points.add(sequence_end)
    cut_points.update(other_events.sequence)

    # Oscillator phase increments are determined in a two-step process.
    #  1. A preliminary list of all phase increments is collected. These are used to
    #     determine the cut points for waveforms, as some phase increments require
    #     oscillator switching.
    #  2. Once the waveform intervals have been established (aka compacted), the actual
    #     phase increments are baked into the intervals.
    prelim_oscillator_phase_increments = _oscillator_phase_increment_points(
        events, signals, delay, sampling_rate
    )

    (
        oscillator_switch_events,
        oscillator_switch_cut_points,
    ) = _oscillator_switch_cut_points(
        interval_tree,
        signals,
        sample_multiple,
        prelim_oscillator_phase_increments,
        branching_intervals,
    )

    cut_points.update(oscillator_switch_cut_points)
    oscillator_intervals = _oscillator_intervals(
        signals.values(), oscillator_switch_events, sequence_end
    )
    used_oscillators = {
        osc_iv.data["oscillator"] for osc_iv in oscillator_intervals.intervals
    }
    if len(used_oscillators) > 1:
        assert use_command_table, (
            "HW oscillator switching only possible in command-table mode"
        )

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

    # Add branching points as events
    for section_name, ivs in branching_intervals.items():
        for interval in ivs:
            interval_event = AWGEvent(
                type=AWGEventType.MATCH,
                start=interval.begin,
                end=interval.end,
                params={
                    "handle": interval.data.handle,
                    "local": interval.data.local,
                    "user_register": interval.data.user_register,
                    "prng_sample": interval.data.prng_sample,
                    "signal_id": signal_id,
                    "section_name": section_name,
                },
            )
            interval_events.add(interval.begin, interval_event)

    _logger.debug("Calculating waveform signatures for signal %s", signal_id)

    amplitude_register_count = (
        device_type.amplitude_register_count if use_command_table else 1
    )
    (
        amplitude_register_set_events,
        amplitude_register_by_param,
    ) = analyze_amplitude_register_set_events(
        events, device_type, sampling_rate, delay, use_command_table
    )

    use_ct_hw_oscillator = use_command_table and device_type == DeviceType.SHFSG
    for k in compacted_intervals:
        _logger.debug("Calculating signature for %s and its children", k)

        overlap: list[Interval] = interval_tree.overlap(k.begin, k.end)
        _logger.debug("Overlap is %s", overlap)
        [hw_oscillator_interval] = oscillator_intervals.overlap(k.begin, k.end)
        hw_oscillator = hw_oscillator_interval.data["oscillator"]

        # Group by states for match/state sections
        v_state: Dict[int, list[Interval]] = {}
        for i in overlap:
            data: _PlayIntervalData = i.data
            v_state.setdefault(data.state, []).append(i)

        for state, intervals in v_state.items():
            signature_pulses = []
            pulse_parameters = []
            for iv in sorted(intervals, key=lambda x: (x.begin, x.data.channel)):
                pulse_signature, these_pulse_parameters = _make_pulse_signature(
                    iv, k, signal_ids, amplitude_register_by_param
                )
                signature_pulses.append(pulse_signature)
                pulse_parameters.append(these_pulse_parameters)
            if signature_pulses:
                waveform_signature = WaveformSignature(
                    k.length(), tuple(signature_pulses)
                )
                signature = PlaybackSignature(
                    waveform=waveform_signature,
                    hw_oscillator=hw_oscillator if use_ct_hw_oscillator else None,
                    pulse_parameters=tuple(pulse_parameters),
                    state=state,
                    clear_precompensation=False,
                )
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
                if len(signal_ids) > 1:
                    interval_event.params["multiplexed_signal_ids"] = signal_ids
                interval_events.add(start, interval_event)

    # A value of 'None' indicates that we do not know the current value (e.g. after a
    # match block, where we may conditionally have written to the register).
    amplitude_register_values: list[float | None] = [None] * amplitude_register_count

    use_ct_phase = use_command_table and all(s.hw_oscillator for s in signals.values())
    signatures = set()

    frame_change_events = _insert_frame_changes(
        interval_events, events, delay, sampling_rate, signals
    )

    interval_events.merge(amplitude_register_set_events)

    priorities = {
        AWGEventType.INIT_AMPLITUDE_REGISTER: 0,
        AWGEventType.PLAY_WAVE: 1,
        AWGEventType.MATCH: 1,
    }
    interval_events.sort()
    for event_list in interval_events.sequence.values():
        for event in sorted(event_list, key=lambda x: priorities[x.type]):
            if event.type == AWGEventType.PLAY_WAVE:
                signature = event.params["playback_signature"]
                signature = reduce_signature_amplitude_register(signature)
                signature = reduce_signature_amplitude(
                    signature,
                    use_command_table,
                    amplitude_register_values if use_amplitude_increment else None,
                )

                signature = reduce_signature_phase(signature, use_ct_phase)
                if phase_resolution_range >= 1:
                    signature.quantize_phase(phase_resolution_range)

                if amplitude_resolution_range >= 1:
                    signature.quantize_amplitude(amplitude_resolution_range)
                if signature.set_amplitude is not None:
                    amplitude_register_values[signature.amplitude_register] = (
                        signature.set_amplitude
                    )
                if signature.increment_amplitude is not None:
                    amplitude_register_values[signature.amplitude_register] += (
                        signature.increment_amplitude
                    )

                if signature.state is not None:
                    # After a conditional playback, we do not know for sure what value
                    # was written to the amplitude register. Mark it as such, to avoid
                    # emitting an increment for the next signature.
                    # Todo: There are ways to improve on this:
                    #   - Do not write the amplitude register in branches
                    #   - Only clear those registers that were indeed written in a branch
                    #   - ...
                    amplitude_register_values = [None] * amplitude_register_count

                if signature not in signatures:
                    signatures.add(signature)
            elif event.type == AWGEventType.INIT_AMPLITUDE_REGISTER:
                signature = event.params["playback_signature"]
                signature = reduce_signature_amplitude(
                    signature, use_command_table, amplitude_register_values
                )
                if amplitude_resolution_range >= 1:
                    signature.quantize_amplitude(amplitude_resolution_range)
                if signature.set_amplitude is not None:
                    amplitude_register_values[signature.amplitude_register] = (
                        signature.set_amplitude
                    )
                if signature.increment_amplitude is not None:
                    amplitude_register_values[signature.amplitude_register] += (
                        signature.increment_amplitude
                    )

    if len(signatures) > 0:
        _logger.debug(
            "Signatures calculated: %d signatures for signal %s",
            len(signatures),
            signal_id,
        )
    for sig in signatures:
        _logger.debug(sig)
    _logger.debug("Interval events: %s", interval_events.sequence)

    interval_events.merge(frame_change_events)

    return interval_events
