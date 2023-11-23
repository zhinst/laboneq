# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from laboneq.compiler.ir.root_ir import RootIR
from laboneq.data.compilation_job import DeviceInfo, PulseDef, SignalInfo


@dataclass
class IR:
    uid: str = None
    devices: list[DeviceInfo] = field(default_factory=list)
    signals: list[SignalInfo] = field(default_factory=list)
    root: Optional[RootIR] = None
    global_leader_device: Optional[DeviceInfo] = None  # todo: remove
    pulse_defs: list[PulseDef] = field(default_factory=list)

    def round_trip(self):
        from laboneq.dsl.serialization import Serializer

        json = Serializer.to_json(self)
        rt = Serializer.from_json(json, IR)
        assert rt == self
