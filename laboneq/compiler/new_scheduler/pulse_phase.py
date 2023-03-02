# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access import ExperimentDAO

_logger = logging.getLogger(__name__)


def calculate_osc_phase(event_list, experiment_dao: ExperimentDAO):
    """Traverse the event list, and elaborate the phase of each played pulse."""
    oscillator_phase_cumulative = {}
    oscillator_phase_sets = {}

    phase_reset_time = 0.0
    priority_map = {
        EventType.SET_OSCILLATOR_PHASE: -5,
        EventType.PLAY_START: 0,
        EventType.ACQUIRE_START: 0,
        EventType.INCREMENT_OSCILLATOR_PHASE: -9,
        EventType.RESET_SW_OSCILLATOR_PHASE: -15,
    }
    sorted_events = sorted(
        (e for e in event_list if e["event_type"] in priority_map),
        key=lambda e: (e["time"], priority_map[e["event_type"]]),
    )

    for event in sorted_events:
        if event["event_type"] == EventType.RESET_SW_OSCILLATOR_PHASE:
            phase_reset_time = event["time"]
        elif event["event_type"] == EventType.INCREMENT_OSCILLATOR_PHASE:
            signal_id = event["signal"]
            if signal_id not in oscillator_phase_cumulative:
                oscillator_phase_cumulative[signal_id] = 0.0
            phase_incr = event["increment_oscillator_phase"]
            oscillator_phase_cumulative[signal_id] += phase_incr
        elif event["event_type"] == EventType.SET_OSCILLATOR_PHASE:
            signal_id = event["signal"]
            osc_phase = event["set_oscillator_phase"]
            oscillator_phase_cumulative[signal_id] = osc_phase
            oscillator_phase_sets[signal_id] = event["time"]

        elif event["event_type"] in [EventType.PLAY_START, EventType.ACQUIRE_START]:
            oscillator_phase = None
            baseband_phase = None
            signal_id = event["signal"]
            signal_info = experiment_dao.signal_info(signal_id)
            oscillator_info = experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is not None:
                if signal_info.modulation and signal_info.device_type in [
                    "hdawg",
                    "shfsg",
                ]:
                    incremented_phase = oscillator_phase_cumulative.get(signal_id, 0.0)

                    if oscillator_info.hardware:
                        if len(oscillator_phase_sets) > 0:
                            raise Exception(
                                f"There are set_oscillator_phase entries for signal "
                                f"'{signal_id}', but oscillator '{oscillator_info.id}' "
                                f"is a hardware oscillator. Setting absolute phase is "
                                f"not supported for hardware oscillators."
                            )
                        baseband_phase = incremented_phase
                    else:
                        phase_reference_time = phase_reset_time
                        if signal_id in oscillator_phase_sets:
                            phase_reference_time = max(
                                phase_reset_time, oscillator_phase_sets[signal_id]
                            )
                        oscillator_phase = (
                            event["time"] - phase_reference_time
                        ) * 2.0 * math.pi * event.get(
                            "oscillator_frequency", 0.0
                        ) + incremented_phase
            event["oscillator_phase"] = oscillator_phase
            event["baseband_phase"] = baseband_phase
