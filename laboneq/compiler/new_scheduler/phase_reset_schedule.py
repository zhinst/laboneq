# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule


@dataclass(frozen=True)
class PhaseResetSchedule(IntervalSchedule):
    section: str
    hw_osc_devices: List[Tuple[str, float]]
    reset_sw_oscillators: bool

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
    ) -> List[Dict]:
        events = [
            {
                "event_type": EventType.RESET_HW_OSCILLATOR_PHASE,
                "time": start,
                "section_name": self.section,
                "id": next(id_tracker),
                "duration": duration,
                "device_id": device,
            }
            for device, duration in self.hw_osc_devices
        ]

        if self.reset_sw_oscillators:
            events.append(
                {
                    "event_type": EventType.RESET_SW_OSCILLATOR_PHASE,
                    "time": start,
                    "section_name": self.section,
                    "id": next(id_tracker),
                }
            )

        return events

    def __hash__(self):
        super().__hash__()
