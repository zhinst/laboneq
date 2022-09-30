# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import json

from laboneq.core.exceptions import LabOneQException
from laboneq.core.serialization.simple_serialization import (
    serialize_to_dict_with_ref,
    module_classes,
    deserialize_from_dict_with_ref,
)
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.device import Instrument
from laboneq.dsl.device import LogicalSignalGroup
from laboneq.dsl.device import Server
from laboneq.dsl.device.physical_channel_group import (
    PhysicalChannelGroup,
    PhysicalChannel,
)
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.section import Section
from laboneq.dsl.parameter import Parameter


class Serializer:
    @staticmethod
    def _entity_config():
        entity_classes = frozenset(
            (
                Section,
                Parameter,
                Pulse,
                Oscillator,
                MixerCalibration,
                Instrument,
                LogicalSignalGroup,
                PhysicalChannelGroup,
                PhysicalChannel,
                Server,
            )
        )
        entity_mapper = {
            "Section": "sections",
            "Pulse": "pulses",
            "Parameter": "parameters",
            "Oscillator": "oscillators",
            "MixerCalibration": "mixer_calibrations",
            "Instrument": "instruments",
            "LogicalSignalGroup": "logical_signal_groups",
            "PhysicalChannelGroup": "physical_channel_groups",
            "PhysicalChannel": "physical_channels",
            "Server": "servers",
        }

        return entity_classes, entity_mapper

    @staticmethod
    def to_json(serializable_object) -> str:
        if isinstance(serializable_object, dict):
            json_dump = json.dumps(serializable_object, sort_keys=True)
        else:

            entity_classes, entity_mapper = Serializer._entity_config()

            json_struct = serialize_to_dict_with_ref(
                serializable_object,
                entity_classes,
                entity_mapper,
                emit_enum_types=True,
            )
            json_dump = json.dumps(json_struct, sort_keys=True)
        return json_dump

    @staticmethod
    def to_dict(serializable_object) -> str:
        entity_classes, entity_mapper = Serializer._entity_config()
        json_dump = serialize_to_dict_with_ref(
            serializable_object, entity_classes, entity_mapper, emit_enum_types=True,
        )
        return json_dump

    @staticmethod
    def to_json_file(
        serializable_object, filename: str,
    ):
        json_string = Serializer.to_json(serializable_object)
        try:
            with open(filename, mode="w") as file:
                file.write(json_string)
        except IOError as e:
            raise LabOneQException() from e

    @staticmethod
    def _classes_by_short_name():
        dsl_modules = [
            "laboneq.dsl.experiment",
            "laboneq.dsl.calibration.oscillator",
            "laboneq.dsl.calibration.signal_calibration",
            "laboneq.dsl.result.results",
            "laboneq.dsl.parameter",
            "laboneq.dsl.calibration",
            "laboneq.dsl.device",
            "laboneq.dsl.device.server",
            "laboneq.dsl.device.servers.data_server",
            "laboneq.core.types.enums",
            "laboneq.core.types.compiled_experiment",
            "laboneq.core.types.device_output_signals",
            "laboneq.dsl.device.io_units.logical_signal",
            "laboneq.dsl.device.io_units.physical_channel",
            "laboneq.dsl.device.instruments",
            "laboneq.dsl.result.waveform",
        ]
        classes_by_fullname, classes_by_short_name = module_classes(dsl_modules)
        return classes_by_short_name

    @staticmethod
    def from_json(
        serialized_string: str, type_hint,
    ):
        if type_hint is dict:
            obj = json.loads(serialized_string)
        else:
            entity_classes, entity_mapper = Serializer._entity_config()
            serialized_form = json.loads(serialized_string)

            obj = deserialize_from_dict_with_ref(
                serialized_form,
                Serializer._classes_by_short_name(),
                entity_classes,
                entity_mapper,
            )

        return obj

    @staticmethod
    def load(
        data, type_hint,
    ):
        if type_hint is dict:
            obj = copy.deepcopy(data)
        else:
            entity_classes, entity_mapper = Serializer._entity_config()

            obj = deserialize_from_dict_with_ref(
                data, Serializer._classes_by_short_name(), entity_classes, entity_mapper
            )

        return obj

    @staticmethod
    def from_json_file(filename: str, type_hint):
        try:
            with open(filename, mode="r") as file:
                json_string = file.read()
        except IOError as e:
            raise LabOneQException(e.__repr__()) from e

        return Serializer.from_json(json_string, type_hint)
