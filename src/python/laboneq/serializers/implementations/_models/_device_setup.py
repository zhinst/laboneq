# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from enum import Enum
from typing import ClassVar, Type, Union

import attrs

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
from laboneq.dsl.device.connection import Connection
from laboneq.dsl.device.instruments.hdawg import HDAWG
from laboneq.dsl.device.instruments.pqsc import PQSC
from laboneq.dsl.device.instruments.pretty_printer_device import PRETTYPRINTERDEVICE
from laboneq.dsl.device.instruments.shfppc import SHFPPC
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.device.instruments.shfqc import SHFQC
from laboneq.dsl.device.instruments.shfsg import SHFSG
from laboneq.dsl.device.instruments.uhfqa import UHFQA
from laboneq.dsl.device.instruments.qhub import QHUB
from laboneq.dsl.device.instruments.nonqc import NonQC
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
    PhysicalChannelType,
)
from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
from laboneq.dsl.device.physical_channel_group import PhysicalChannelGroup
from laboneq.dsl.device.servers.data_server import DataServer
from laboneq.serializers.implementations._models._calibration import (
    SignalCalibrationModel,
)
from laboneq.serializers.implementations._models._calibration import (
    make_converter as make_calibration_converter,
)

from ._common import (
    collect_models,
    register_models,
)


class IODirectionModel(Enum):
    IN = "IN"
    OUT = "OUT"
    _target_class = IODirection


class IOSignalTypeModel(Enum):
    I = "I"
    Q = "Q"
    IQ = "IQ"
    RF = "RF"
    SINGLE = "SINGLE"
    LO = "LO"
    DIO = "DIO"
    ZSYNC = "ZSYNC"
    PPC = "PPC"
    _target_class = IOSignalType


class ReferenceClockSourceModel(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    _target_class = ReferenceClockSource


class PhysicalChannelTypeModel(Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"
    _target_class = PhysicalChannelType


@attrs.define
class DataServerModel:
    uid: str
    host: str
    port: str | int
    api_level: int
    _target_class: ClassVar[Type] = DataServer


@attrs.define
class ConnectionModel:
    direction: IODirectionModel
    local_path: str | None
    local_port: str | None
    remote_path: str | None
    remote_port: str | None
    signal_type: IOSignalTypeModel | None
    _target_class: ClassVar[Type] = Connection


@attrs.define
class ZIStandardInstrumentModel:
    uid: str
    interface: str
    connections: list[ConnectionModel]
    server_uid: str | None
    address: str | None
    device_options: str | None
    reference_clock_source: ReferenceClockSourceModel | str | None
    _target_class: ClassVar[Type] = ZIStandardInstrument
    _instrument_map: ClassVar[dict] = {
        "HDAWG": HDAWG,
        "UHFQA": UHFQA,
        "SHFQC": SHFQC,
        "SHFSG": SHFSG,
        "SHFPPC": SHFPPC,
        "SHFQA": SHFQA,
        "PQSC": PQSC,
        "QHUB": QHUB,
        "NonQC": NonQC,
        "PRETTYPRINTERDEVICE": PRETTYPRINTERDEVICE,
    }

    @classmethod
    def _unstructure(cls, obj):
        if type(obj).__name__ not in cls._instrument_map:
            raise ValueError(
                f"Unsupported instrument type: {type(obj).__name__} when unstructuring ZIStandardInstrumentModel"
            )
        return {
            "uid": obj.uid,
            "interface": obj.interface,
            "connections": [
                _converter.unstructure(i, ConnectionModel) for i in obj.connections
            ],
            "server_uid": obj.server_uid,
            "address": obj.address,
            "device_options": obj.device_options,
            "reference_clock_source": _converter.unstructure(
                obj.reference_clock_source, Union[ReferenceClockSourceModel, str, None]
            ),
            "_instrument_type": type(obj).__name__,
        }

    @classmethod
    def _structure(cls, obj, _):
        if obj["_instrument_type"] not in cls._instrument_map:
            raise ValueError(
                f"Unsupported instrument type: {obj['_instrument_type']} when structuring InstrumentModel"
            )
        instrument = cls._instrument_map[obj["_instrument_type"]]
        if obj["reference_clock_source"] is None or isinstance(
            obj["reference_clock_source"], str
        ):
            ref_clk_source = obj["reference_clock_source"]
        else:
            ref_clk_source = ReferenceClockSourceModel._target_class.value(
                obj["reference_clock_source"]
            )
        return instrument(
            uid=obj["uid"],
            interface=obj["interface"],
            connections=[
                _converter.structure(i, ConnectionModel) for i in obj["connections"]
            ],
            server_uid=obj["server_uid"],
            address=obj["address"],
            device_options=obj["device_options"],
            reference_clock_source=ref_clk_source,
        )


@attrs.define
class PhysicalChannelModel:
    uid: str
    name: str | None
    type: PhysicalChannelTypeModel | None
    path: str | None
    calibration: SignalCalibrationModel | None
    _target_class: ClassVar[Type] = PhysicalChannel


@attrs.define
class PhysicalChannelGroupModel:
    uid: str
    channels: dict[str, PhysicalChannelModel]
    _target_class: ClassVar[Type] = PhysicalChannelGroup


@attrs.define
class LogicalSignalModel:
    uid: str
    name: str | None
    calibration: SignalCalibrationModel | None
    _physical_channel: PhysicalChannelModel | None
    path: str | None
    direction: IODirectionModel | None
    _target_class: ClassVar[Type] = LogicalSignal

    @classmethod
    def _structure(cls, obj, _):
        if obj["calibration"]:
            calibration = _converter.structure(
                obj["calibration"], SignalCalibrationModel
            )
        else:
            calibration = None
        # LogicalSignal was updated in 2.51.0 to
        # have _physical_channel as a private attribute.
        # This is a workaround to keep backward compatibility
        physical_channel = obj.get("_physical_channel") or obj.get("physical_channel")
        if physical_channel is not None:
            physical_channel = _converter.structure(
                physical_channel, PhysicalChannelModel
            )
        return cls._target_class(
            uid=obj["uid"],
            name=obj["name"],
            calibration=calibration,
            physical_channel=physical_channel,
            path=obj["path"],
            direction=IODirectionModel._target_class.value(obj["direction"])
            if obj["direction"]
            else None,
        )


@attrs.define
class LogicalSignalGroupModel:
    uid: str | None
    logical_signals: dict[str, LogicalSignalModel]
    _target_class: ClassVar[Type] = LogicalSignalGroup


def make_converter():
    _converter = make_calibration_converter()
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
