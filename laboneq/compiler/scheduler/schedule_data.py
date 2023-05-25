# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from laboneq.compiler.common.compiler_settings import CompilerSettings
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
    from laboneq.compiler.scheduler.pulse_schedule import PulseSchedule
    from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker


@dataclass
class ScheduleData:
    experiment_dao: ExperimentDAO
    sampling_rate_tracker: SamplingRateTracker
    settings: CompilerSettings
    acquire_pulses: Dict[str, List[PulseSchedule]] = field(default_factory=dict)
    signal_objects: Dict[str, SignalObj] = field(default_factory=dict)
    combined_warnings: Dict[str, Tuple[Callable, List]] = field(default_factory=dict)
    TINYSAMPLE: float = field(init=False)

    def __post_init__(self):
        self.TINYSAMPLE = self.settings.TINYSAMPLE
