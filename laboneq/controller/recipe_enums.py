# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    IQ = "iq"
    SINGLE = "single"
    INTEGRATION = "integration"
    MARKER = "marker"


class RefClkType(Enum):
    _10MHZ = 10
    _100MHZ = 100


class TriggeringMode(Enum):
    ZSYNC_FOLLOWER = 1
    DIO_FOLLOWER = 2
    DESKTOP_LEADER = 3
    DESKTOP_DIO_FOLLOWER = 4
    INTERNAL_FOLLOWER = 5


@dataclass(frozen=True)
class NtStepKey:
    indices: tuple[int]


class AcquisitionType(Enum):
    INTEGRATION_TRIGGER = "integration_trigger"
    SPECTROSCOPY_IQ = "spectroscopy"
    SPECTROSCOPY_PSD = "spectroscopy_psd"
    SPECTROSCOPY = SPECTROSCOPY_IQ
    DISCRIMINATION = "discrimination"
    RAW = "raw"
