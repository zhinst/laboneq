# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Optional

import attr
from attr import field

from laboneq.compiler.ir.root_ir import RootScheduleIR
from laboneq.data.compilation_job import (
    DeviceInfo,
    DeviceInfoType,
    OscillatorInfo,
    PulseDef,
    SignalInfo,
)


@attr.define(slots=True)
class DeviceIR:
    uid: str = ""
    device_type: Optional[DeviceInfoType] = None

    @classmethod
    def from_device_info(cls, device_info: DeviceInfo):
        return cls(uid=device_info.uid, device_type=device_info.device_type)


@attr.define(slots=True)
class SignalIR:
    uid: str = ""
    device: Optional[DeviceIR] = None
    oscillator: Optional[OscillatorInfo] = None

    @classmethod
    def from_signal_info(cls, signal_info: SignalInfo):
        return cls(
            uid=signal_info.uid,
            device=DeviceIR.from_device_info(signal_info.device),
            oscillator=signal_info.oscillator,
        )


@attr.define(slots=True)
class IRTree:
    devices: list[DeviceIR] = field(factory=list)
    signals: list[SignalIR] = field(factory=list)
    root: Optional[RootScheduleIR] = None
    pulse_defs: list[PulseDef] = field(factory=list)

    def round_trip(self):
        from laboneq.dsl.serialization import Serializer

        json = Serializer.to_json(self)
        rt = Serializer.from_json(json, IRTree)
        assert rt == self
