# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class PhaseResetSchedule(IntervalSchedule):
    section: str
    hw_osc_devices: List[Tuple[str, float]]
    reset_sw_oscillators: bool

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None
        return start

    def __hash__(self):
        return super().__hash__()
