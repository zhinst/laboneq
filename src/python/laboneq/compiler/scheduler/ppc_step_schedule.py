# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class PPCStepSchedule(IntervalSchedule):
    section: str
    qa_device: str
    qa_channel: int
    ppc_device: str
    ppc_channel: int
    trigger_duration: float

    pump_power: float | None = None
    pump_frequency: float | None = None
    probe_power: float | None = None
    probe_frequency: float | None = None
    cancellation_phase: float | None = None
    cancellation_attenuation: float | None = None

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None
        return start
