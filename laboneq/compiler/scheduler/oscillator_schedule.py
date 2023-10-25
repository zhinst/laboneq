# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from attrs import define

from laboneq.compiler.common.swept_hardware_oscillator import SweptHardwareOscillator
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class OscillatorFrequencyStepSchedule(IntervalSchedule):
    section: str
    oscillators: List[SweptHardwareOscillator]
    params: List[str]
    values: List[float]
    iteration: int

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None
        return start

    def __hash__(self):
        return super().__hash__()
