# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""QCCS SystemProfile serialization models (Gen2 devices)."""

from __future__ import annotations

import datetime
import sys
from enum import Enum
from typing import Any, ClassVar, Type, cast

import attrs

from laboneq.compiler.common.device_type import DeviceType
from laboneq.dsl.device.system_profile_qccs import (
    DeviceCapabilitiesQCCS,
    SystemProfileQCCS,
)
from laboneq.serializers.implementations._models._common import (
    collect_models,
    register_models,
)
from laboneq.serializers.implementations._models._device_setup import (
    _converter,
    register_system_profile_plugin,
)


class DeviceTypeModel(Enum):
    HDAWG = "hdawg"
    UHFQA = "uhfqa"
    SHFQA = "shfqa"
    SHFSG = "shfsg"
    PRETTYPRINTERDEVICE = "prettyprinterdevice"
    _target_class: ClassVar[Type] = DeviceType


@attrs.define
class DeviceCapabilitiesQCCSModel:
    device_model: str | None
    device_options: list[str]
    _target_class: ClassVar[Type] = DeviceCapabilitiesQCCS


@attrs.define
class DateTimeModel:
    _target_class: ClassVar[Type] = datetime.datetime

    @classmethod
    def _unstructure(cls, obj: datetime.datetime) -> str:
        return obj.isoformat()

    @classmethod
    def _structure(cls, obj: str, _) -> datetime.datetime:
        return datetime.datetime.fromisoformat(obj)


@attrs.define
class SystemProfileQCCSModel:
    version: str
    generated_at: DateTimeModel
    laboneq_version: str
    setup_uid: str
    devices: dict[str, DeviceCapabilitiesQCCSModel]
    _target_class: ClassVar[Type] = SystemProfileQCCS

    @classmethod
    def _unstructure(cls, obj: SystemProfileQCCS) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["version"] = obj.version
        result["generated_at"] = DateTimeModel._unstructure(obj.generated_at)
        result["laboneq_version"] = obj.laboneq_version
        result["setup_uid"] = obj.setup_uid
        result["server_address"] = obj.server_address
        result["server_port"] = obj.server_port
        result["server_version"] = obj.server_version
        result["devices"] = {}
        for uid, dev in obj.devices.items():
            result["devices"][uid] = _converter.unstructure(
                dev, DeviceCapabilitiesQCCSModel
            )
        return result

    @classmethod
    def _structure(cls, obj: dict[str, Any], _) -> SystemProfileQCCS:
        devices = {
            cast(str, uid): _converter.structure(dev, DeviceCapabilitiesQCCSModel)
            for uid, dev in obj["devices"].items()
        }
        result = SystemProfileQCCS(
            version=obj["version"],
            generated_at=DateTimeModel._structure(obj["generated_at"], None),
            laboneq_version=obj["laboneq_version"],
            setup_uid=obj["setup_uid"],
            server_address=obj["server_address"],
            server_port=obj["server_port"],
            server_version=obj["server_version"],
            devices=devices,
        )
        return result


register_system_profile_plugin("QCCS", SystemProfileQCCS, SystemProfileQCCSModel)
register_models(_converter, collect_models(sys.modules[__name__]))
