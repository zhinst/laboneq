# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, fields
from typing import Dict, TypeVar

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
    "USE_AMPLITUDE_INCREMENT",
    "OUTPUT_EXTRAS",
    "FORCE_IR_ROUNDTRIP",
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
DEFAULT_TESTDEVICE_LEAD: float = 80e-9


def round_min_playwave_hint(n: int, multiple: int) -> int:
    return math.ceil(n / multiple) * multiple


@dataclass(frozen=True)
class CompilerSettings:
    HDAWG_LEAD_PQSC: float = DEFAULT_HDAWG_LEAD_PQSC
    HDAWG_LEAD_PQSC_2GHz: float = DEFAULT_HDAWG_LEAD_PQSC_2GHz
    HDAWG_LEAD_DESKTOP_SETUP: float = DEFAULT_HDAWG_LEAD_DESKTOP_SETUP
    HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHz
    UHFQA_LEAD_PQSC: float = DEFAULT_UHFQA_LEAD_PQSC
    SHFQA_LEAD_PQSC: float = DEFAULT_SHFQA_LEAD_PQSC
    SHFSG_LEAD_PQSC: float = DEFAULT_SHFSG_LEAD_PQSC
    PRETTYPRINTERDEVICE_LEAD: float = DEFAULT_TESTDEVICE_LEAD

    AMPLITUDE_RESOLUTION_BITS: int = 24
    PHASE_RESOLUTION_BITS: int = 24
    MAX_EVENTS_TO_PUBLISH: int = 1000
    EXPAND_LOOPS_FOR_SCHEDULE: bool = True
    OUTPUT_EXTRAS: bool = False
    FORCE_IR_ROUNDTRIP: bool = False
    TINYSAMPLE: float = 1 / 3600000e6

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
    USE_AMPLITUDE_INCREMENT: bool = True

    EMIT_TIMING_COMMENTS: bool = False

    LOG_REPORT: bool = True

    @classmethod
    def from_dict(cls, settings: dict | None = None):
        if settings is None:
            return cls()

        if "EXPAND_LOOPS_FOR_SCHEDULE" in settings:
            warnings.warn(
                "Setting `EXPAND_LOOPS_FOR_SCHEDULE` is deprecated.\n"
                "Use the expand_loops_for_schedule argument of laboneq.pulse_sheet_viewer.pulse_sheet_viewer.view_pulse_sheet"
                " to set loop expansion for the pulse sheet viewer",
                FutureWarning,
                stacklevel=2,
            )

        if "SHFSG_FORCE_COMMAND_TABLE" in settings:
            warnings.warn(
                "The setting `SHFSG_FORCE_COMMAND_TABLE` is ignored and will be removed in a future version",
                FutureWarning,
                stacklevel=2,
            )
        if "HDAWG_FORCE_COMMAND_TABLE" in settings:
            warnings.warn(
                "The setting `HDAWG_FORCE_COMMAND_TABLE` is ignored and will be removed in a future version",
                FutureWarning,
                stacklevel=2,
            )

        if ("MAX_EVENTS_TO_PUBLISH" in settings) and ("OUTPUT_EXTRAS" not in settings):
            warnings.warn(
                "Setting `MAX_EVENTS_TO_PUBLISH` has no effect unless used together with `OUTPUT_EXTRAS=True`.",
                FutureWarning,
                stacklevel=2,
            )

        valid_field_names = [field.name for field in fields(cls)]

        for k in settings:
            if k not in valid_field_names:
                raise KeyError(f"Not a valid setting: {k}")

        return cls(**settings)


UserSettings = TypeVar("UserSettings", Dict, None)


def filter_user_settings(settings: UserSettings | None = None) -> UserSettings:
    if settings is not None:
        settings = {k: v for k, v in settings.items() if k in _USER_ENABLED_SETTINGS}
    return settings


from_dict = CompilerSettings.from_dict


EXECUTETABLEENTRY_LATENCY = 3
