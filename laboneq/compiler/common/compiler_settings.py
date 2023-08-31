# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Optional, TypeVar

_logger = logging.getLogger(__name__)


_USER_ENABLED_SETTINGS = [
    "MAX_EVENTS_TO_PUBLISH",
    "PHASE_RESOLUTION_BITS",
    "HDAWG_MIN_PLAYWAVE_HINT",
    "HDAWG_MIN_PLAYZERO_HINT",
    "UHFQA_MIN_PLAYWAVE_HINT",
    "UHFQA_MIN_PLAYZERO_HINT",
    "SHFQA_MIN_PLAYWAVE_HINT",
    "SHFQA_MIN_PLAYZERO_HINT",
    "SHFSG_MIN_PLAYWAVE_HINT",
    "SHFSG_MIN_PLAYZERO_HINT",
    "EMIT_TIMING_COMMENTS",
    "HDAWG_FORCE_COMMAND_TABLE",
    "SHFSG_FORCE_COMMAND_TABLE",
    "PREPARE_PSV_DATA",
    "LOG_REPORT",
]

DEFAULT_HDAWG_LEAD_PQSC: float = 80e-9
DEFAULT_HDAWG_LEAD_PQSC_2GHz: float = 80e-9
DEFAULT_HDAWG_LEAD_DESKTOP_SETUP: float = (
    20e-9  # PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
)
DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = 24e-9
DEFAULT_UHFQA_LEAD_PQSC: float = 80e-9
DEFAULT_SHFQA_LEAD_PQSC: float = 80e-9
DEFAULT_SHFSG_LEAD_PQSC: float = 80e-9


def round_min_playwave_hint(n: int, multiple: int) -> int:
    return math.ceil(n / multiple) * multiple


@dataclass(frozen=True)
class CompilerSettings:
    # IMPORTANT: All fields must be type annotated for dataclass to pick them up
    # properly.
    # Alternatively, use `dataclasses.field()`.

    HDAWG_LEAD_PQSC: float = DEFAULT_HDAWG_LEAD_PQSC
    HDAWG_LEAD_PQSC_2GHz: float = DEFAULT_HDAWG_LEAD_PQSC_2GHz
    HDAWG_LEAD_DESKTOP_SETUP: float = DEFAULT_HDAWG_LEAD_DESKTOP_SETUP
    HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHz
    UHFQA_LEAD_PQSC: float = DEFAULT_UHFQA_LEAD_PQSC
    SHFQA_LEAD_PQSC: float = DEFAULT_SHFQA_LEAD_PQSC
    SHFSG_LEAD_PQSC: float = DEFAULT_SHFSG_LEAD_PQSC

    AMPLITUDE_RESOLUTION_BITS: int = 24
    PHASE_RESOLUTION_BITS: int = 16
    MAX_EVENTS_TO_PUBLISH: int = 1000
    EXPAND_LOOPS_FOR_SCHEDULE: bool = True
    PREPARE_PSV_DATA: bool = False
    CONSTRAINT_TOLERANCE: float = 1e-15
    TINYSAMPLE: float = 1 / 3600000e6

    FIXED_SLOW_SECTION_GRID: bool = False

    HDAWG_MIN_PLAYWAVE_HINT: int = 128
    HDAWG_MIN_PLAYZERO_HINT: int = 128
    UHFQA_MIN_PLAYWAVE_HINT: int = 64
    UHFQA_MIN_PLAYZERO_HINT: int = 64
    SHFQA_MIN_PLAYWAVE_HINT: int = 64
    SHFQA_MIN_PLAYZERO_HINT: int = 64
    SHFSG_MIN_PLAYWAVE_HINT: int = 64
    SHFSG_MIN_PLAYZERO_HINT: int = 64

    HDAWG_FORCE_COMMAND_TABLE: bool = True
    SHFSG_FORCE_COMMAND_TABLE: bool = True

    EMIT_TIMING_COMMENTS: bool = False
    IGNORE_GRAPH_VERIFY_RESULTS: bool = False

    LOG_REPORT: bool = True

    def __post_init__(self):
        if (
            self.MAX_EVENTS_TO_PUBLISH != CompilerSettings.MAX_EVENTS_TO_PUBLISH
            and not self.PREPARE_PSV_DATA
        ):
            warnings.warn(
                "Setting `PREPARE_PSV_DATA` is required in addition to"
                " `MAX_EVENTS_TO_PUBLISH` if a schedule should be produced.",
                FutureWarning,
            )

        if self.EXPAND_LOOPS_FOR_SCHEDULE != CompilerSettings.EXPAND_LOOPS_FOR_SCHEDULE:
            warnings.warn(
                """Setting `EXPAND_LOOPS_FOR_SCHEDULE` is deprecated.
                          Use the expand_loops_for_schedule argument of laboneq.pulse_sheet_viewer.pulse_sheet_viewer.view_pulse_sheet
                          to set loop expansion for the pulse sheet viewer""",
                FutureWarning,
            )


UserSettings = TypeVar("UserSettings", Dict, None)


def filter_user_settings(settings: UserSettings = None) -> UserSettings:
    if settings is not None:
        settings = {k: v for k, v in settings.items() if k in _USER_ENABLED_SETTINGS}
    return settings


def from_dict(settings: Optional[Dict] = None) -> CompilerSettings:
    compiler_settings_dict = asdict(CompilerSettings())

    if settings is not None:
        for k, v in settings.items():
            if k not in compiler_settings_dict:
                raise KeyError(f"Not a valid setting: {k}")
            compiler_settings_dict[k] = v

    compiler_settings = CompilerSettings(**compiler_settings_dict)

    for k, v in asdict(compiler_settings).items():
        _logger.debug("Setting %s=%s", k, v)

    return compiler_settings


EXECUTETABLEENTRY_LATENCY = 3
