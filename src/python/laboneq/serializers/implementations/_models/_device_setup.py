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
from laboneq.dsl.calibration.signal_calibration import SignalCalibration
from laboneq.dsl.device.connection import Connection
from laboneq.dsl.device.instruments.hdawg import HDAWG
from laboneq.dsl.device.instruments.pqsc import PQSC
from laboneq.dsl.device.instruments.shfppc import SHFPPC
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.device.instruments.shfqc import SHFQC
from laboneq.dsl.device.instruments.shfsg import SHFSG
from laboneq.dsl.device.instruments.uhfqa import UHFQA
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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "host": obj.host,
            "port": obj.port,
            "api_level": obj.api_level,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            host=obj["host"],
            port=obj["port"],
            api_level=obj["api_level"],
        )


@attrs.define
class ConnectionModel:
    direction: IODirectionModel
    local_path: str | None
    local_port: str | None
    remote_path: str | None
    remote_port: str | None
    signal_type: IOSignalTypeModel | None
    _target_class: ClassVar[Type] = Connection

    @classmethod
    def _unstructure(cls, obj):
        return {
            "direction": _converter.unstructure(
                obj.direction, Union[IODirectionModel, None]
            ),
            "local_path": obj.local_path,
            "local_port": obj.local_port,
            "remote_path": obj.remote_path,
            "remote_port": obj.remote_port,
            "signal_type": _converter.unstructure(
                obj.signal_type, Union[IOSignalTypeModel, None]
            ),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            direction=None
            if obj["direction"] is None
            else IODirectionModel._target_class.value(obj["direction"]),
            local_path=obj["local_path"],
            local_port=obj["local_port"],
            remote_path=obj["remote_path"],
            remote_port=obj["remote_port"],
            signal_type=None
            if obj["signal_type"] is None
            else IOSignalTypeModel._target_class.value(obj["signal_type"]),
        )


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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "name": obj.name,
            "type": _converter.unstructure(obj.type, Union[PhysicalChannelType, None]),
            "path": obj.path,
            "calibration": _converter.unstructure(
                obj._calibration, Union[SignalCalibrationModel, None]
            ),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            name=obj["name"],
            type=None
            if obj["type"] is None
            else PhysicalChannelTypeModel._target_class.value(obj["type"]),
            path=obj["path"],
            calibration=None
            if obj["calibration"] is None
            else _converter.structure(obj["calibration"], SignalCalibrationModel),
        )


@attrs.define
class PhysicalChannelGroupModel:
    uid: str
    channels: dict[str, PhysicalChannelModel]
    _target_class: ClassVar[Type] = PhysicalChannelGroup

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "channels": {
                k: _converter.unstructure(v, PhysicalChannelModel)
                for k, v in obj.channels.items()
            },
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            channels={
                k: _converter.structure(v, PhysicalChannelModel)
                for k, v in obj["channels"].items()
            },
        )


@attrs.define
class LogicalSignalModel:
    uid: str
    name: str | None
    calibration: SignalCalibration | None
    physical_channel: PhysicalChannelModel | None
    path: str | None
    direction: IODirectionModel | None
    _target_class: ClassVar[Type] = LogicalSignal

    @classmethod
    def _unstructure(cls, obj):
        if obj.calibration:
            calibration = _converter.unstructure(
                obj.calibration, SignalCalibrationModel
            )
        else:
            calibration = None
        if obj.physical_channel:
            physical_channel = _converter.unstructure(
                obj.physical_channel, PhysicalChannelModel
            )
        else:
            physical_channel = None
        return {
            "uid": obj.uid,
            "name": obj.name,
            "calibration": calibration,
            "physical_channel": physical_channel,
            "path": obj.path,
            "direction": _converter.unstructure(obj.direction, IODirectionModel)
            if obj.direction
            else None,
        }

    @classmethod
    def _structure(cls, obj, _):
        if obj["calibration"]:
            calibration = _converter.structure(
                obj["calibration"], SignalCalibrationModel
            )
        else:
            calibration = None
        if obj["physical_channel"]:
            physical_channel = _converter.structure(
                obj["physical_channel"], PhysicalChannelModel
            )
        else:
            physical_channel = None
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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "logical_signals": {
                k: _converter.unstructure(v, LogicalSignalModel)
                for k, v in obj.logical_signals.items()
            },
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            logical_signals={
                k: _converter.structure(v, LogicalSignalModel)
                for k, v in obj["logical_signals"].items()
            },
        )


def make_converter():
    _converter = make_calibration_converter()
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
