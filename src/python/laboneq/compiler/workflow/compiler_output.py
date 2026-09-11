# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_compiler_output import (
        CombinedOutput,
    )


@dataclass
class CombinedRTCompilerOutputContainer:
    """Container for the compiler artifacts, after linking."""

    backend: str
    combined_output: CombinedOutput
    schedule: compiler_rs.PulseSheetSchedule | None = None
