# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List

from attrs import define

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define
class SweptHardwareOscillator:
    id: str
    signal: str
    device: str


@define(kw_only=True, slots=True)
class OscillatorFrequencyStepSchedule(IntervalSchedule):
    section: str
    oscillators: List[SweptHardwareOscillator]
    params: List[str]
    values: List[float]
    iteration: int

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        assert self.absolute_start is not None
        retval = []
        for param, osc, value in zip(self.params, self.oscillators, self.values):
            start_id = next(id_tracker)
            retval.extend(
                [
                    {
                        "event_type": EventType.SET_OSCILLATOR_FREQUENCY_START,
                        "time": start,
                        "parameter": {"id": param},
                        "iteration": self.iteration,
                        "value": value,
                        "section_name": self.section,
                        "device_id": osc.device,
                        "signal": osc.signal,
                        "oscillator_id": osc.id,
                        "id": start_id,
                        "chain_element_id": start_id,
                    },
                    {
                        "event_type": EventType.SET_OSCILLATOR_FREQUENCY_END,
                        "time": start + self.length,
                        "id": next(id_tracker),
                        "chain_element_id": start_id,
                    },
                ]
            )
        return retval

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None
        return start

    def __hash__(self):
        super().__hash__()
