# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class SectionInfo:
    section_id: str
    has_repeat: bool
    execution_type: Optional[str]
    acquisition_types: Optional[List[str]]
    averaging_type: Optional[str]
    count: int
    align: Optional[str]
    on_system_grid: bool
    length: Optional[float]
    averaging_mode: Optional[str]
    repetition_mode: Optional[str]
    repetition_time: Optional[float]
    play_after: Optional[Union[str, List[str]]]
    reset_oscillator_phase: bool
    handle: Optional[str]
    state: Optional[int]
    local: Optional[bool]
    section_display_name: Optional[str] = None
    trigger_output: List[Dict] = field(default_factory=list)
