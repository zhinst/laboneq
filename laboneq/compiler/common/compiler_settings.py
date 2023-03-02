# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
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
    "USE_EXPERIMENTAL_SCHEDULER",
]


def round_min_playwave_hint(n: int, multiple: int) -> int:
    return math.ceil(n / multiple) * multiple


@dataclass(frozen=True)
class CompilerSettings:
    # IMPORTANT: All fields must be type annotated for dataclass to pick them up
    # properly.
    # Alternatively, use `dataclasses.field()`.

    HDAWG_LEAD_PQSC: float = 80e-9
    HDAWG_LEAD_PQSC_2GHz: float = 80e-9
    HDAWG_LEAD_DESKTOP_SETUP: float = (
        20e-9  # PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
    )
    HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = 24e-9
    UHFQA_LEAD_PQSC: float = 80e-9
    SHFQA_LEAD_PQSC: float = 80e-9
    SHFSG_LEAD_PQSC: float = 80e-9

    AMPLITUDE_RESOLUTION_BITS: int = 24
    PHASE_RESOLUTION_BITS: int = 12
    MAX_EVENTS_TO_PUBLISH: int = 1000
    EXPAND_LOOPS_FOR_SCHEDULE: bool = True
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

    HDAWG_FORCE_COMMAND_TABLE: bool = False
    SHFSG_FORCE_COMMAND_TABLE: bool = True

    EMIT_TIMING_COMMENTS: bool = False
    IGNORE_GRAPH_VERIFY_RESULTS: bool = False

    USE_EXPERIMENTAL_SCHEDULER: bool = False


UserSettings = TypeVar("UserSettings", Dict, None)


def filter_user_settings(settings: UserSettings = None) -> UserSettings:
    if settings is not None:
        settings = {k: v for k, v in settings.items() if k in _USER_ENABLED_SETTINGS}
    return settings


def from_dict(settings: Optional[Dict] = None) -> CompilerSettings:
    def to_value(input_string):
        try:
            return int(input_string)
        except ValueError:
            pass
        try:
            return float(input_string)
        except ValueError:
            pass
        if input_string.lower() in ["true", "false"]:
            return input_string.lower() == "true"

    PREFIX = "QCCS_COMPILER_"
    compiler_settings_dict = asdict(CompilerSettings())

    for settings_key in compiler_settings_dict.keys():
        key = PREFIX + settings_key
        if key in os.environ:
            value = to_value(os.environ[key])
            if value is not None:
                compiler_settings_dict[settings_key] = value
                _logger.warning(
                    "Environment variable %s is set. %s overridden to be %s instead of default value %s",
                    key,
                    settings_key,
                    value,
                    getattr(CompilerSettings, settings_key),
                )
        else:
            _logger.debug("Key %s not found in environment variables", key)

    if settings is not None:
        for k, v in settings.items():
            if not k in compiler_settings_dict:
                raise KeyError(f"Not a valid setting: {k}")
            compiler_settings_dict[k] = v

    compiler_settings = CompilerSettings(**compiler_settings_dict)

    for k, v in asdict(compiler_settings).items():
        _logger.debug("Setting %s=%s", k, v)

    return compiler_settings
