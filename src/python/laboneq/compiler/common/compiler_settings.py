# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields

DEFAULT_HDAWG_LEAD_PQSC: float = 80e-9
DEFAULT_HDAWG_LEAD_PQSC_2GHz: float = 80e-9
DEFAULT_HDAWG_LEAD_DESKTOP_SETUP: float = (
    20e-9  # PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
)
DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = 24e-9
DEFAULT_UHFQA_LEAD_PQSC: float = 80e-9
DEFAULT_SHFQA_LEAD_PQSC: float = 80e-9
DEFAULT_SHFSG_LEAD_PQSC: float = 80e-9
DEFAULT_TESTDEVICE_LEAD: float = 1200e-9

# Mirrors TINYSAMPLE_DURATION from laboneq-units. See that constant for the rationale.
TINYSAMPLE: float = 1 / 3600000e6

SHF_OUTPUT_MUTE_MIN_DURATION: float = 280e-9


@dataclass(frozen=True)
class CompilerSettings:
    # NOTE: The rest of the settings are processed in the Rust compiler.
    LOG_REPORT: bool = True
    IGNORE_RESOURCE_LIMITATION_ERRORS: bool = False

    @classmethod
    def from_dict(cls, settings: dict | None = None):
        if settings is None:
            return cls()

        include_fields = [field.name for field in fields(cls)]
        return cls(
            **{key: value for key, value in settings.items() if key in include_fields}
        )


from_dict = CompilerSettings.from_dict


EXECUTETABLEENTRY_LATENCY = 3
