# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from laboneq.compiler.common.compiler_settings import CompilerSettings
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO


@dataclass
class ScheduleData:
    experiment_dao: ExperimentDAO
    settings: Optional[CompilerSettings]
    TINYSAMPLE: float = field(init=False)

    def __post_init__(self):
        self.TINYSAMPLE = getattr(self.settings, "TINYSAMPLE", 1 / 3600000e6)
