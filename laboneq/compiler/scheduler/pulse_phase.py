# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from laboneq.compiler import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access import ExperimentDAO

_logger = logging.getLogger(__name__)


def calculate_osc_phase(event_list, experiment_dao: ExperimentDAO):
    """Traverse the event list, and elaborate the phase of each played pulse.

    For SW oscillators, calculate the time since the last set/reset of that oscillator,
    and store it in the event as `oscillator_phase`. Illegal phase sets/resets in
    conditional branches have previously been ruled out (see scheduler).
    The `[increment|set]_oscillator_phase` fields are removed if present, and their
    effect is aggregated into `oscillator_phase`.

    For HW oscillators, do nothing. Absolute phase sets are illegal (and were caught in
    the scheduler), and phase increments will be handled in the code generator.

    After this function returns, all play events will contain the following phase-related
    fields:
     - "phase": the baseband phase of the pulse
     - "oscillator_phase": the oscillator phase for SW modulators, `None` for HW
     - "increment_oscillator_phase": if present, the event should increment the HW modulator
    """
    oscillator_phase_cumulative = {}
    oscillator_phase_sets = {}

    phase_reset_time = 0.0
    priority_map = {
        EventType.PLAY_START: 0,
        EventType.DELAY_START: 0,
        EventType.ACQUIRE_START: 0,
        EventType.RESET_SW_OSCILLATOR_PHASE: -15,
    }
    sorted_events = sorted(
        (e for e in event_list if e["event_type"] in priority_map),
        key=lambda e: (e["time"], priority_map[e["event_type"]]),
    )

    for event in sorted_events:
        if event["event_type"] == EventType.RESET_SW_OSCILLATOR_PHASE:
            phase_reset_time = event["time"]
            for signal_id in oscillator_phase_cumulative.keys():
                oscillator_phase_cumulative[signal_id] = 0.0

        else:
            signal_id = event["signal"]
            oscillator_info = experiment_dao.signal_oscillator(signal_id)
            is_hw_osc = oscillator_info.is_hardware if oscillator_info else False
            if (phase_incr := event.get("increment_oscillator_phase")) is not None:
                if not is_hw_osc:
                    if signal_id not in oscillator_phase_cumulative:
                        oscillator_phase_cumulative[signal_id] = 0.0
                    oscillator_phase_cumulative[signal_id] += phase_incr
                    del event["increment_oscillator_phase"]

            # if both "increment_oscillator_phase" and "set_oscillator_phase" are specified,
            # the absolute phase overwrites the increment.
            if (osc_phase := event.get("set_oscillator_phase")) is not None:
                assert (
                    not oscillator_info.is_hardware
                ), "cannot set phase of HW oscillators (should have been caught earlier)"

                oscillator_phase_cumulative[signal_id] = osc_phase
                oscillator_phase_sets[signal_id] = event["time"]
                del event["set_oscillator_phase"]

            if is_hw_osc:
                event["oscillator_phase"] = None
            else:  # SW oscillator
                device = experiment_dao.device_from_signal(signal_id)
                device_type = DeviceType.from_device_info_type(device.device_type)
                if not device_type.is_qa_device:
                    incremented_phase = oscillator_phase_cumulative.get(signal_id, 0.0)
                    phase_reference_time = max(
                        phase_reset_time, oscillator_phase_sets.get(signal_id, 0.0)
                    )
                    oscillator_frequency = event.get("oscillator_frequency", 0.0)
                    t = event["time"] - phase_reference_time
                    event["oscillator_phase"] = (
                        t * 2.0 * math.pi * oscillator_frequency + incremented_phase
                    )
                else:
                    event["oscillator_phase"] = 0.0
