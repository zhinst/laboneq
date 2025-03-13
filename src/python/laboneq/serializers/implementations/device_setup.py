# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.implementations._models import (
    DataServerModel,
    ZIStandardInstrumentModel,
    LogicalSignalGroupModel,
    PhysicalChannelGroupModel,
    make_laboneq_converter,
)
from laboneq.serializers.implementations.quantum_element import QuantumElementSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)


@serializer(types=DeviceSetup, public=True)
class DeviceSetupSerializer(VersionedClassSerializer[DeviceSetup]):
    SERIALIZER_ID = "laboneq.serializers.implementations.DeviceSetupSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: DeviceSetup, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        converter = make_laboneq_converter()
        uid = obj.uid
        servers = {
            k: converter.unstructure(v, DataServerModel) for k, v in obj.servers.items()
        }
        instruments = [
            converter.unstructure(instrument, ZIStandardInstrumentModel)
            for instrument in obj.instruments
        ]
        physical_channels_groups = {
            k: converter.unstructure(v, PhysicalChannelGroupModel)
            for k, v in obj.physical_channel_groups.items()
        }
        logical_signal_groups = {
            k: converter.unstructure(v, LogicalSignalGroupModel)
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
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> DeviceSetup:
        se = serialized_data["__data__"]
        converter = make_laboneq_converter()
        uid = se["uid"]
        servers = {
            k: converter.structure(v, DataServerModel) for k, v in se["servers"].items()
        }
        instruments = [
            converter.structure(instrument, ZIStandardInstrumentModel)
            for instrument in se["instruments"]
        ]
        physical_channel_groups = {
            k: converter.structure(v, PhysicalChannelGroupModel)
            for k, v in se["physical_channel_groups"].items()
        }
        logical_signal_groups = {
            k: converter.structure(v, LogicalSignalGroupModel)
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
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> DeviceSetup:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
