# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List

from laboneq.compiler.code_generator.signatures import PlaybackSignature
from laboneq.compiler.common.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
    AWGSampledEventSequence,
)
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.core.utilities.pulse_sampler import length_to_samples


def _ancestors(section, parents):
    retval = []
    while section in parents:
        retval.append(section := parents[section])
    return retval


def find_subsections(events):
    """Find all (transitive) child relations between sections in the event list"""
    subsections: dict[str, set[str]] = {}  # all descendants (also indirect)
    parent_sections: dict[str, str] = {}  # direct parent
    for event in events:
        if event["event_type"] != EventType.SUBSECTION_START:
            continue
        parent, child = event["section_name"], event["subsection_name"]
        assert parent_sections.get(child, parent) == parent
        parent_sections[child] = parent
        ancestors = _ancestors(child, parent_sections)
        for ancestor in ancestors:
            subsections.setdefault(ancestor, set()).add(child)
    return subsections


def find_all_loop_sections(events):
    return {
        event["section_name"]: event.get("compressed", False)
        for event in events
        if event["event_type"] == EventType.LOOP_ITERATION_END
    }


def analyze_amplitude_register_set_events(
    events: List[Dict],
    device_type: DeviceType,
    sampling_rate,
    delay: float,
    use_command_table: bool,
) -> tuple[AWGSampledEventSequence, dict[str, int]]:
    event_sequence = AWGSampledEventSequence()
    if not use_command_table:
        return event_sequence, {}
    amplitude_parameters: set[str] = set()

    for event in events:
        if event["event_type"] != EventType.PLAY_START:
            continue
        if (amp_param := event.get("amplitude_parameter")) is not None:
            amplitude_parameters.add(amp_param)

    amp_register_counter = 1
    amplitude_register_by_parameter: dict[str, int] = {}
    available_registers = device_type.amplitude_register_count
    for param_name in sorted(amplitude_parameters):  # sort for deterministic order
        amplitude_register_by_parameter[param_name] = (
            amp_register_counter if amp_register_counter < available_registers else 0
        )
        amp_register_counter += 1
        # Note: lifetime of registers is not tracked. We are not reusing the register
        # in a later sweep!

    all_subsections = find_subsections(events)
    loop_sections = find_all_loop_sections(events)
    compressed_loops = [
        loop for loop, is_compressed in loop_sections.items() if is_compressed
    ]

    # Any of these sweep loops contain a compressed loop (the averaging loop).
    # We must emit a dedicated amplitude set command before entering that inner loop.
    critical_loops = [
        loop
        for loop in loop_sections
        if any(
            compressed_loop in all_subsections.get(loop, [])
            for compressed_loop in compressed_loops
        )
    ]

    for event in events:
        if event["event_type"] != EventType.PARAMETER_SET:
            continue
        if event["section_name"] not in critical_loops:
            continue
        param_name = event["parameter"]["id"]
        if param_name not in amplitude_register_by_parameter:
            continue
        value = event["value"]
        register = amplitude_register_by_parameter[param_name]
        event_time_in_samples = length_to_samples(event["time"] + delay, sampling_rate)

        # create a zero-length command table signature
        signature = PlaybackSignature(
            waveform=None,
            hw_oscillator=None,
            pulse_parameters=(),
            amplitude_register=register,
            set_amplitude=value,
        )
        set_amp_register_event = AWGEvent(
            type=AWGEventType.INIT_AMPLITUDE_REGISTER,
            start=event_time_in_samples,
            end=event_time_in_samples,
            params={"playback_signature": signature},
        )
        event_sequence.add(event_time_in_samples, set_amp_register_event)
    return event_sequence, amplitude_register_by_parameter
