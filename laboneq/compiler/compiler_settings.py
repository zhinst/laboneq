# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.compiler.device_type import DeviceType


@dataclass(frozen=True)
class CompilerSettings:
    # IMPORTANT: All fields must be type annotated for dataclass to pick them up
    # properly.
    # Alternatively, use `dataclasses.field()`.

    HDAWG_LEAD_PQSC: float = 80e-9
    HDAWG_LEAD_PQSC_2GHz: float = 80e-9
    HDAWG_LEAD_DESKTOP_SETUP: float = 20e-9  # PW 2022-09-21, dev2806, FPGA 68366, dev8047, FPGA 68666 & 68603
    HDAWG_LEAD_DESKTOP_SETUP_2GHz: float = 24e-9
    UHFQA_LEAD_PQSC: float = 80e-9
    SHFQA_LEAD_PQSC: float = 80e-9
    SHFSG_LEAD_PQSC: float = 80e-9

    AMPLITUDE_RESOLUTION_BITS: int = 24
    MAX_EVENTS_TO_PUBLISH: int = 1000
    EXPAND_LOOPS_FOR_SCHEDULE: bool = True
    CONSTRAINT_TOLERANCE: float = 1e-15
    TINYSAMPLE: float = 1 / 3600000e6

    FIXED_SLOW_SECTION_GRID: bool = False

    HDAWG_MIN_PLAYWAVE_HINT: int = 128
    HDAWG_MIN_PLAYZERO_HINT: int = 128
    UHFQA_MIN_PLAYWAVE_HINT: int = 768
    UHFQA_MIN_PLAYZERO_HINT: int = 768
    SHFQA_MIN_PLAYWAVE_HINT: int = 1024
    SHFQA_MIN_PLAYZERO_HINT: int = 1024
    SHFSG_MIN_PLAYWAVE_HINT: int = 1024
    SHFSG_MIN_PLAYZERO_HINT: int = 1024
