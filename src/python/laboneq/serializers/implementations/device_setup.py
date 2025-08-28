# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.implementations._models._calibration import (
    remove_high_pass_clearing,
)
from laboneq.serializers.implementations._models._device_setup import (
    DataServerModel,
    LogicalSignalGroupModel,
    PhysicalChannelGroupModel,
    ZIStandardInstrumentModel,
    make_converter,
)
from laboneq.serializers.implementations.quantum_element import QuantumElementSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

_logger = logging.getLogger(__name__)
_converter = make_converter()


@serializer(types=DeviceSetup, public=True)
class DeviceSetupSerializer(VersionedClassSerializer[DeviceSetup]):
    SERIALIZER_ID = "laboneq.serializers.implementations.DeviceSetupSerializer"
    VERSION = 3

    @classmethod
    def to_dict(
        cls, obj: DeviceSetup, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        uid = obj.uid
        servers = {
            k: _converter.unstructure(v, DataServerModel)
            for k, v in obj.servers.items()
        }
        instruments = [
            _converter.unstructure(instrument, ZIStandardInstrumentModel)
            for instrument in obj.instruments
        ]
        physical_channels_groups = {
            k: _converter.unstructure(v, PhysicalChannelGroupModel)
            for k, v in obj.physical_channel_groups.items()
        }
        logical_signal_groups = {
            k: _converter.unstructure(v, LogicalSignalGroupModel)
            for k, v in obj.logical_signal_groups.items()
        }
        qubits = {
            k: QuantumElementSerializer.to_dict(v, options)
            for k, v in obj.qubits.items()
        }
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "uid": uid,
                "servers": servers,
                "instruments": instruments,
                "physical_channel_groups": physical_channels_groups,
                "logical_signal_groups": logical_signal_groups,
                "qubits": qubits,
            },
        }

    @classmethod
    def from_dict_v3(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> DeviceSetup:
        se = serialized_data["__data__"]
        uid = se["uid"]
        servers = {
            k: _converter.structure(v, DataServerModel)
            for k, v in se["servers"].items()
        }
        instruments = [
            _converter.structure(instrument, ZIStandardInstrumentModel)
            for instrument in se["instruments"]
        ]
        physical_channel_groups = {
            k: _converter.structure(v, PhysicalChannelGroupModel)
            for k, v in se["physical_channel_groups"].items()
        }
        logical_signal_groups = {
            k: _converter.structure(v, LogicalSignalGroupModel)
            for k, v in se["logical_signal_groups"].items()
        }
        qubits = {
            k: QuantumElementSerializer.from_dict(v, options)
            for k, v in se["qubits"].items()
        }
        return DeviceSetup(
            uid=uid,
            servers=servers,
            instruments=instruments,
            physical_channel_groups=physical_channel_groups,
            logical_signal_groups=logical_signal_groups,
            qubits=qubits,
        )

    @classmethod
    def _rename_physical_channel_field(cls, device_setup_data: dict):
        """Rename `physical_channel` to `_physical_channel` for v2
        serializations where logical signal was saved using the former field
        name."""
        for group in device_setup_data["logical_signal_groups"].values():
            for logical_signal in group["logical_signals"].values():
                if "physical_channel" in logical_signal:
                    logical_signal["_physical_channel"] = logical_signal.pop(
                        "physical_channel"
                    )

    @classmethod
    def _remove_high_pass_clearing_v2(cls, device_setup_data: dict):
        for group in device_setup_data["logical_signal_groups"].values():
            for logical_signal in group["logical_signals"].values():
                signal_uid = logical_signal["path"]
                calibration_info = logical_signal["calibration"]
                remove_high_pass_clearing(signal_uid, calibration_info, _logger)
                physical_channel = logical_signal["_physical_channel"]
                calibration_info_physical = physical_channel["calibration"]
                remove_high_pass_clearing(
                    signal_uid, calibration_info_physical, _logger
                )

        for device in device_setup_data["physical_channel_groups"].values():
            for physical_channel in device["channels"].values():
                signal_uid = physical_channel["path"]
                calibration_info = physical_channel["calibration"]
                remove_high_pass_clearing(signal_uid, calibration_info, _logger)

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> DeviceSetup:
        se = serialized_data["__data__"]
        cls._rename_physical_channel_field(se)
        cls._remove_high_pass_clearing_v2(se)
        return cls.from_dict_v3(serialized_data, options)

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> DeviceSetup:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
