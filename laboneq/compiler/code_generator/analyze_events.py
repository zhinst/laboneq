# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from engineering_notation import EngNumber
from sortedcontainers import SortedDict

from laboneq.compiler.code_generator.feedback_register_allocator import (
    FeedbackRegisterAllocator,
)
from laboneq.compiler.code_generator.utils import resample_state
from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.common.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import interval_to_samples, length_to_samples

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj

_logger = logging.getLogger(__name__)


def analyze_loop_times(
    awg: AWGInfo,
    events: List[Any],
    sampling_rate: float,
    delay: float,
) -> AWGSampledEventSequence:
    loop_events = AWGSampledEventSequence()

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
            "Analyzing loop events for awg %d of %s", awg.awg_number, awg.device_id
        )
    else:
        _logger.debug(
            "Skipping analysis of loop events for awg %d of %s because nothing is played",
            awg.awg_number,
            awg.device_id,
        )
        return loop_events

    loop_step_start_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == "LOOP_STEP_START"
    ]
    loop_step_end_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == "LOOP_STEP_END"
    ]
    loop_iteration_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == "LOOP_ITERATION_END"
        and "compressed" in event
        and event["compressed"]
    ]

    compressed_sections = {event["section_name"] for _, event in loop_iteration_events}
    section_repeats = {}
    for _, event in loop_iteration_events:
        if "num_repeats" in event:
            section_repeats[event["section_name"]] = event["num_repeats"]

    _logger.debug("Found %d loop step start events", len(loop_step_start_events))
    events_already_added = set()
    for index, event in loop_step_start_events:
        _logger.debug("  loop timing: processing  %s", event)

        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        loop_event = AWGEvent(
            type=AWGEventType.LOOP_STEP_START,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=index,
            params={
                "nesting_level": event["nesting_level"],
                "loop_id": event["section_name"],
            },
        )

        if event["section_name"] not in compressed_sections:
            frozen = loop_event.frozen()
            if frozen not in events_already_added:
                loop_events.add(event_time_in_samples, loop_event)
                events_already_added.add(frozen)
                _logger.debug("Added %s", loop_event)
            else:
                _logger.debug("SKIP adding double %s", loop_event)
        elif event["iteration"] == 0:
            push_event = AWGEvent(
                type=AWGEventType.PUSH_LOOP,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=index,
                params={
                    "nesting_level": event["nesting_level"],
                    "loop_id": event["section_name"],
                    "num_repeats": section_repeats.get(event["section_name"]),
                },
            )
            loop_events.add(event_time_in_samples, push_event)
            _logger.debug("Added %s", push_event)

    _logger.debug("Found %d loop step end events", len(loop_step_end_events))
    for index, event in loop_step_end_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        loop_event = AWGEvent(
            type=AWGEventType.LOOP_STEP_END,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=index,
            params={
                "nesting_level": event["nesting_level"],
                "loop_id": event["section_name"],
            },
        )

        if event["section_name"] not in compressed_sections:
            frozen = loop_event.frozen()
            if frozen not in events_already_added:
                loop_events.add(event_time_in_samples, loop_event)
                events_already_added.add(frozen)
                _logger.debug("Added %s", loop_event)
            else:
                _logger.debug("SKIP adding double %s", loop_event)

    for index, event in loop_iteration_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        iteration_event = AWGEvent(
            type=AWGEventType.ITERATE,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=index,
            params={
                "nesting_level": event["nesting_level"],
                "loop_id": event["section_name"],
                "num_repeats": section_repeats.get(event["section_name"]),
            },
        )

        loop_events.add(event_time_in_samples, iteration_event)
        _logger.debug("Added %s", iteration_event)

        if event_time_in_samples % awg.device_type.sample_multiple != 0:
            _logger.warning(
                "Event %s: event_time_in_samples %f at sampling rate %s is not divisible by %d",
                event,
                event_time_in_samples,
                EngNumber(sampling_rate),
                awg.device_type.sample_multiple,
            )

    return loop_events


def analyze_init_times(
    device_id: str, sampling_rate: float, delay: float
) -> AWGSampledEventSequence:
    _logger.debug("Analyzing init events for device %s", device_id)
    init_events = AWGSampledEventSequence()
    delay_samples = length_to_samples(delay, sampling_rate)
    init_events.add(
        delay_samples,
        AWGEvent(
            type=AWGEventType.SEQUENCER_START,
            start=delay_samples,
            end=delay_samples,
            priority=-100,
            params={
                "device_id": device_id,
            },
        ),
    )
    return init_events


def analyze_precomp_reset_times(
    events: List[Any],
    signals: List[str],
    sampling_rate: float,
    delay: float,
) -> AWGSampledEventSequence:
    precomp_events = [
        (index, event)
        for index, event in enumerate(events)
        if (
            event["event_type"] == EventType.RESET_PRECOMPENSATION_FILTERS
            and event.get("signal_id", None) in signals
        )
    ]
    precomp_reset_events = AWGSampledEventSequence()

    # We clear the precompensation filters in a dedicated command table entry.
    # Currently, a bug (HULK-1246) prevents us from doing so in a zero-length
    # command, so instead we allocate 32 samples (minimum waveform length) for this, and
    # register the end of this interval as a cut point.

    PRECOMP_RESET_LENGTH = 32

    for index, event in precomp_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        signal_id = event.get("signal_id")
        sampled_event = AWGEvent(
            type=AWGEventType.RESET_PRECOMPENSATION_FILTERS,
            start=event_time_in_samples,
            end=event_time_in_samples + PRECOMP_RESET_LENGTH,
            priority=index,
            params={"signal_id": signal_id},
        )
        precomp_reset_events.add(event_time_in_samples, sampled_event)

        # Add end event to force a cut point
        sampled_event = AWGEvent(
            type=AWGEventType.RESET_PRECOMPENSATION_FILTERS_END,
            start=event_time_in_samples + PRECOMP_RESET_LENGTH,
            priority=-100,  # does not actually emit code, so irrelevant
            params={"signal_id": signal_id},
        )
        precomp_reset_events.add(event_time_in_samples, sampled_event)

    return precomp_reset_events


def analyze_phase_reset_times(
    events: List[Any],
    device_id: str,
    sampling_rate: float,
    delay: float,
) -> AWGSampledEventSequence:
    reset_phase_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"]
        in (
            EventType.RESET_HW_OSCILLATOR_PHASE,
            EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
        )
        and "device_id" in event
        and event["device_id"] == device_id
    ]
    phase_reset_events = AWGSampledEventSequence()
    for index, event in reset_phase_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        event_type = (
            AWGEventType.RESET_PHASE
            if event["event_type"] == EventType.RESET_HW_OSCILLATOR_PHASE
            else AWGEventType.INITIAL_RESET_PHASE
        )
        init_event = AWGEvent(
            type=event_type,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=index,
            params={
                "device_id": device_id,
            },
        )
        phase_reset_events.add(event_time_in_samples, init_event)
    return phase_reset_events


def analyze_set_oscillator_times(
    events: List, signal_obj: SignalObj, global_delay: float
) -> AWGSampledEventSequence:
    signal_id = signal_obj.id
    device_id = signal_obj.awg.device_id
    device_type = signal_obj.awg.device_type
    sampling_rate = signal_obj.awg.sampling_rate
    set_oscillator_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == "SET_OSCILLATOR_FREQUENCY_START"
        and event.get("device_id") == device_id
        and event.get("signal") == signal_id
    ]
    if len(set_oscillator_events) == 0:
        return AWGSampledEventSequence()

    if device_type not in (DeviceType.HDAWG, DeviceType.SHFQA, DeviceType.SHFSG):
        raise LabOneQException(
            "Real-time frequency sweep only supported on SHF and HDAWG devices"
        )

    iterations = [event["iteration"] for _, event in set_oscillator_events]

    start_frequency = set_oscillator_events[0][1]["value"]
    if len(iterations) > 1:
        step_frequency = set_oscillator_events[1][1]["value"] - start_frequency
    else:
        step_frequency = 0

    retval = AWGSampledEventSequence()

    for index, event in set_oscillator_events:
        iteration = event["iteration"]
        if (
            abs(event["value"] - iteration * step_frequency - start_frequency)
            > 1e-3  # tolerance: 1 mHz
        ):
            raise LabOneQException("Realtime oscillator sweeps must be linear")

        event_time_in_samples = length_to_samples(
            event["time"] + global_delay, sampling_rate
        )
        set_oscillator_event = AWGEvent(
            type=AWGEventType.SET_OSCILLATOR_FREQUENCY,
            start=event_time_in_samples,
            priority=index,
            params={
                "start_frequency": start_frequency,
                "step_frequency": step_frequency,
                "parameter_name": event["parameter"]["id"],
                "iteration": iteration,
                "iterations": len(iterations),
                "oscillator_id": event["oscillator_id"],
            },
        )

        retval.add(event_time_in_samples, set_oscillator_event)

    return retval


def analyze_acquire_times(
    events: List[Any],
    signal_obj: SignalObj,
    feedback_register_allocator: FeedbackRegisterAllocator,
) -> AWGSampledEventSequence:
    signal_id = signal_obj.id
    sampling_rate = signal_obj.awg.sampling_rate
    delay = signal_obj.total_delay
    sample_multiple = signal_obj.awg.device_type.sample_multiple
    channels = signal_obj.channels

    _logger.debug(
        "Calculating acquire times for signal %s with delay %s ( %s samples)",
        signal_id,
        str(delay),
        str(round(delay * sampling_rate)),
    )

    @dataclass
    class IntervalStartEvent:
        event_type: str
        time: float
        play_wave_id: str
        acquisition_type: list
        acquire_handle: str
        play_pulse_parameters: Optional[Dict[str, Any]]
        pulse_pulse_parameters: Optional[Dict[str, Any]]
        channels: list[int | list[int]]
        priority: int

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
                    event["time"] + delay,
                    event["play_wave_id"],
                    event.get("acquisition_type", []),
                    event["acquire_handle"],
                    event.get("play_pulse_parameters"),
                    event.get("pulse_pulse_parameters"),
                    event.get("channel") or channels,
                    priority=index,
                )
                for index, event in enumerate(events)
                if event["event_type"] in ["ACQUIRE_START"]
                and event["signal"] == signal_id
            ],
            [
                IntervalEndEvent(
                    event["event_type"],
                    event["time"] + delay,
                    event["play_wave_id"],
                )
                for event in events
                if event["event_type"] in ["ACQUIRE_END"]
                and event["signal"] == signal_id
            ],
        )
    )

    acquire_events = AWGSampledEventSequence()
    for interval_start, interval_end in interval_zip:
        start_samples, end_samples = interval_to_samples(
            interval_start.time, interval_end.time, sampling_rate
        )
        if start_samples % sample_multiple != 0:
            start_samples = (
                math.floor(start_samples / sample_multiple) * sample_multiple
            )

        if end_samples % sample_multiple != 0:
            end_samples = math.floor(end_samples / sample_multiple) * sample_multiple

        feedback_register = (
            None
            if feedback_register_allocator is None
            else feedback_register_allocator.allocate(
                signal_id, interval_start.acquire_handle
            )
        )

        acquire_event = AWGEvent(
            type=AWGEventType.ACQUIRE,
            start=start_samples,
            end=end_samples,
            priority=interval_start.priority,
            params={
                "signal_id": signal_id,
                "play_wave_id": interval_start.play_wave_id,
                "acquisition_type": interval_start.acquisition_type,
                "acquire_handles": [interval_start.acquire_handle],
                "feedback_register": feedback_register,
                "channels": interval_start.channels,
                "play_pulse_parameters": interval_start.play_pulse_parameters,
                "pulse_pulse_parameters": interval_start.pulse_pulse_parameters,
            },
        )

        acquire_events.add(start_samples, acquire_event)

    return acquire_events


def analyze_trigger_events(
    events: List[Dict], signal: SignalObj, loop_events: AWGSampledEventSequence
) -> AWGSampledEventSequence:
    digital_signal_change_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == EventType.DIGITAL_SIGNAL_STATE_CHANGE
        and signal.id == event["signal"]
    ]
    delay = signal.total_delay
    sampling_rate = signal.awg.sampling_rate
    device_type = signal.awg.device_type

    sampled_digital_signal_change_events = AWGSampledEventSequence()

    event: Dict[Any, Any]
    for index, event in digital_signal_change_events:
        time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        sampled_digital_signal_change_events.add(
            time_in_samples, AWGEvent(type=None, params=event, priority=index)
        )

    current_state = 0
    retval = AWGSampledEventSequence()
    state_progression = SortedDict()

    event: AWGEvent
    event_list: List[AWGEvent]
    for (
        time_in_samples,
        event_list,
    ) in sampled_digital_signal_change_events.sequence.items():
        if device_type in (DeviceType.SHFQA, DeviceType.SHFSG):
            for event in event_list:
                if event.params["bit"] > 0:
                    raise LabOneQException(
                        f"On device {device_type.value}, only a single trigger channel is "
                        f"available (section {event.params['section_name']})."
                    )
        for event in [e for e in event_list if e.params["change"] == "CLEAR"]:
            mask = ~(2 ** event.params["bit"])
            current_state = current_state & mask
        for event in [e for e in event_list if e.params["change"] == "SET"]:
            mask = 2 ** event.params["bit"]
            current_state = current_state | mask
        state_progression[time_in_samples] = current_state
        retval.add(
            time_in_samples,
            AWGEvent(
                type=AWGEventType.TRIGGER_OUTPUT,
                start=time_in_samples,
                priority=event_list[0].priority,
                params={
                    "state": current_state,
                },
            ),
        )

    if state_progression and 0 not in state_progression:
        # assume that at time 0, the state is 0
        state_progression[0] = 0

    # When the trigger is raised at the end of the averaging loop (which is not
    # unrolled), things get a bit dicey: the command to reset the trigger signal must
    # be deferred to the next iteration. Which means that the very first iteration must
    # already include this trigger reset, and that we must issue it again after the loop.

    resampled_states = resample_state(loop_events.sequence.keys(), state_progression)
    if resampled_states:
        for time_in_samples, event_list in loop_events.sequence.items():
            if any(event.type == AWGEventType.PUSH_LOOP for event in event_list):
                retval.add(
                    time_in_samples,
                    AWGEvent(
                        type=AWGEventType.TRIGGER_OUTPUT,
                        start=time_in_samples,
                        priority=event_list[0].priority,
                        params={"state": resampled_states[time_in_samples][1]},
                    ),
                )
    return retval


def analyze_prng_setup_times(events, sampling_rate, delay):
    retval = AWGSampledEventSequence()
    filtered_events = (
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == EventType.SETUP_PRNG
    )
    for index, event in filtered_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        retval.add(
            event_time_in_samples,
            AWGEvent(
                type=AWGEventType.SEED_PRNG,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=index,
                params={"range": event["range"], "seed": event["seed"]},
            ),
        )

    return retval
