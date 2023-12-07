# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import ArrayLike

from laboneq.compiler.ir.root_ir import RootIR
from laboneq.data.compilation_job import (
    DeviceInfo,
    DeviceInfoType,
    OscillatorInfo,
    PulseDef,
    SignalInfo,
    SectionInfo,
)


@dataclass
class DeviceIR:
    uid: str = ""
    device_type: Optional[DeviceInfoType] = None

    @classmethod
    def from_device_info(cls, device_info: DeviceInfo):
        return cls(uid=device_info.uid, device_type=device_info.device_type)


@dataclass
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


@dataclass
class PulseDefIR:
    uid: str = ""
    amplitude: Optional[float] = None
    can_compress: bool = False
    function: Optional[str] = None
    samples: Optional[ArrayLike] = None
    length: float = 0.0

    @classmethod
    def from_pulse_def(cls, pulse_def: PulseDef):
        return cls(
            uid=pulse_def.uid,
            amplitude=pulse_def.amplitude,
            can_compress=pulse_def.can_compress,
            function=pulse_def.function,
            length=pulse_def.length,
            samples=pulse_def.samples,
        )


@dataclass
class SectionInfoIR:
    uid: str = ""

    @classmethod
    def from_section_info(cls, section_info: SectionInfo):
        return cls(uid=section_info.uid)


@dataclass
class IR:
    uid: str = None
    devices: list[DeviceIR] = field(default_factory=list)
    signals: list[SignalIR] = field(default_factory=list)
    root: Optional[RootIR] = None
    pulse_defs: list[PulseDefIR] = field(default_factory=list)
    root_section: Optional[SectionInfoIR] = None
    root_section_children: Optional[list[SectionInfoIR]] = None
    section_signals_with_chidlren_ids: Optional[dict[SectionInfoIR, set(str)]] = None

    def round_trip(self):
        from laboneq.dsl.serialization import Serializer

        json = Serializer.to_json(self)
        rt = Serializer.from_json(json, IR)
        assert rt == self
