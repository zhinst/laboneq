# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from engineering_notation import EngNumber
from sortedcontainers import SortedDict

from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.event_list.event_type import EventType
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.seqc.utils import resample_state
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import length_to_samples

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj

_logger = logging.getLogger(__name__)


def analyze_loop_times(
    awg: AWGInfo,
    events: list[dict[str, Any]],
    sampling_rate: float,
    delay: float,
) -> AWGSampledEventSequence:
    loop_events = AWGSampledEventSequence()
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

    compressed_sections = {event["section_name"] for event in loop_iteration_events}
    section_repeats = {}
    for event in loop_iteration_events:
        if "num_repeats" in event:
            section_repeats[event["section_name"]] = event["num_repeats"]

    _logger.debug("Found %d loop step start events", len(loop_step_start_events))
    events_already_added = set()
    for event in loop_step_start_events:
        _logger.debug("  loop timing: processing  %s", event)

        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        loop_event = AWGEvent(
            type=AWGEventType.LOOP_STEP_START,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=event["position"],
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
                priority=event["position"],
                params={
                    "nesting_level": event["nesting_level"],
                    "loop_id": event["section_name"],
                    "num_repeats": section_repeats.get(event["section_name"]),
                },
            )
            loop_events.add(event_time_in_samples, push_event)
            _logger.debug("Added %s", push_event)

    _logger.debug("Found %d loop step end events", len(loop_step_end_events))
    for event in loop_step_end_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        loop_event = AWGEvent(
            type=AWGEventType.LOOP_STEP_END,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=event["position"],
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

    for event in loop_iteration_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        iteration_event = AWGEvent(
            type=AWGEventType.ITERATE,
            start=event_time_in_samples,
            end=event_time_in_samples,
            priority=event["position"],
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


def analyze_phase_reset_times(
    events: list[dict[str, Any]],
    device_id: str,
    sampling_rate: float,
    delay: float,
) -> AWGSampledEventSequence:
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
    phase_reset_events = AWGSampledEventSequence()
    for event in reset_phase_events:
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
            priority=event["position"]
            if event["event_type"] == EventType.RESET_HW_OSCILLATOR_PHASE
            else -100,
            params={
                "device_id": device_id,
            },
        )
        phase_reset_events.add(event_time_in_samples, init_event)
    return phase_reset_events


def analyze_set_oscillator_times(
    events: list[dict[str, Any]], signal_obj: SignalObj, global_delay: float
) -> AWGSampledEventSequence:
    signal_id = signal_obj.id
    device_id = signal_obj.awg.device_id
    device_type = signal_obj.awg.device_type
    sampling_rate = signal_obj.awg.sampling_rate
    set_oscillator_events = [
        (index, event)
        for index, event in enumerate(events)
        if event["event_type"] == EventType.SET_OSCILLATOR_FREQUENCY_START
        and event.get("device_id") == device_id
        and signal_id in event.get("signal")
    ]
    if len(set_oscillator_events) == 0:
        return AWGSampledEventSequence()

    if device_type not in (DeviceType.HDAWG, DeviceType.SHFQA, DeviceType.SHFSG):
        raise LabOneQException(
            "Real-time frequency sweep only supported on SHF and HDAWG devices"
        )

    n_iterations = len([event["iteration"] for _, event in set_oscillator_events])
    if device_type == DeviceType.HDAWG and n_iterations > 512:
        raise LabOneQException(
            "HDAWG can only handle RT frequency sweeps up to 512 steps."
        )
    start_frequency = set_oscillator_events[0][1]["value"]
    if n_iterations > 1:
        step_frequency = set_oscillator_events[1][1]["value"] - start_frequency
    else:
        step_frequency = 0

    retval = AWGSampledEventSequence()

    for _, event in set_oscillator_events:
        iteration = event["iteration"]
        if (
            abs(event["value"] - iteration * step_frequency - start_frequency)
            > 1e-3  # tolerance: 1 mHz
        ):
            raise LabOneQException("Realtime oscillator sweeps must be linear")

        osc_index = signal_obj.awg.oscs[signal_obj.hw_oscillator]
        event_time_in_samples = length_to_samples(
            event["time"] + global_delay, sampling_rate
        )
        set_oscillator_event = AWGEvent(
            type=AWGEventType.SET_OSCILLATOR_FREQUENCY,
            start=event_time_in_samples,
            priority=event["position"],
            params={
                "start_frequency": start_frequency,
                "step_frequency": step_frequency,
                "iteration": iteration,
                "iterations": n_iterations,
                "osc_index": osc_index,
            },
        )

        retval.add(event_time_in_samples, set_oscillator_event)

    # remove what we handled
    for i, _ in reversed(set_oscillator_events):
        events.pop(i)

    return retval


def analyze_trigger_events(
    events: list[dict[str, Any]],
    signals: list[SignalObj],
    loop_events: AWGSampledEventSequence,
) -> AWGSampledEventSequence:
    signal_by_id = {signal.id: signal for signal in signals}
    digital_signal_change_events = [
        ev
        for ev in events
        if ev["event_type"] == EventType.DIGITAL_SIGNAL_STATE_CHANGE
        for signal in signals
        if signal.id == ev["signal"]
    ]

    [sampling_rate] = {signal.awg.sampling_rate for signal in signals}
    [device_type] = {signal.awg.device_type for signal in signals}
    [hdawg_rf_mode] = {signal.signal_type == "single" for signal in signals}

    sampled_digital_signal_change_events = AWGSampledEventSequence()

    for ev in digital_signal_change_events:
        delay = signal_by_id[ev["signal"]].total_delay
        time_in_samples = length_to_samples(ev["time"] + delay, sampling_rate)
        sampled_digital_signal_change_events.add(
            time_in_samples,
            AWGEvent(type=None, params=ev, priority=ev["position"]),
        )

    # The trigger output is stateful, so we must process each event in the correct order.
    sampled_digital_signal_change_events.sort()

    current_state = 0
    retval = AWGSampledEventSequence()
    state_progression = SortedDict()

    for (
        time_in_samples,
        event_list,
    ) in sampled_digital_signal_change_events.sequence.items():
        if device_type in (DeviceType.SHFQA, DeviceType.SHFSG) or hdawg_rf_mode:
            for event in event_list:
                if event.params["bit"] > 0:
                    raise LabOneQException(
                        f"On device {device_type.value}, only a single trigger channel is "
                        f"available (section {event.params['section_name']})."
                    )
        for event in event_list:
            op = event.params["change"]
            mask = 2 ** event.params["bit"]
            if hdawg_rf_mode:
                signal = signal_by_id[event.params["signal"]]
                mask <<= signal.channels[0] % 2
            if op == "CLEAR":
                current_state = current_state & ~mask
            else:  # op == "SET"
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


def analyze_prng_times(events, sampling_rate, delay):
    retval = AWGSampledEventSequence()
    filtered_events = (
        event
        for event in events
        if event["event_type"]
        in (
            EventType.PRNG_SETUP,
            EventType.DROP_PRNG_SETUP,
            EventType.DRAW_PRNG_SAMPLE,
            EventType.DROP_PRNG_SAMPLE,
        )
    )
    for event in filtered_events:
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)
        if event["event_type"] == EventType.PRNG_SETUP:
            awg_event = AWGEvent(
                type=AWGEventType.SETUP_PRNG,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=event["position"],
                params={
                    "range": event["range"],
                    "seed": event["seed"],
                    "section": event["section_name"],
                },
            )
        elif event["event_type"] == EventType.DROP_PRNG_SETUP:
            awg_event = AWGEvent(
                type=AWGEventType.DROP_PRNG_SETUP,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=event["position"],
                params={"section": event["section_name"]},
            )
        elif event["event_type"] == EventType.DRAW_PRNG_SAMPLE:
            awg_event = AWGEvent(
                type=AWGEventType.PRNG_SAMPLE,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=event["position"],
                params={
                    "sample_name": event["sample_name"],
                    "section_name": event["section_name"],
                },
            )
        else:  # EventType.DROP_PRNG_SAMPLE
            awg_event = AWGEvent(
                type=AWGEventType.DROP_PRNG_SAMPLE,
                start=event_time_in_samples,
                end=event_time_in_samples,
                priority=event["position"],
                params={"sample_name": event["sample_name"]},
            )
        retval.add(event_time_in_samples, awg_event)

    return retval
