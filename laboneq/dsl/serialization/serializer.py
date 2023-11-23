# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import OrderedDict
from itertools import chain
from typing import Dict

import orjson

from laboneq.core.exceptions import LabOneQException
from laboneq.core.serialization.simple_serialization import (
    deserialize_from_dict_with_ref,
    module_classes,
    serialize_to_dict_with_ref,
)
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.precompensation import Precompensation
from laboneq.dsl.device import Instrument, LogicalSignalGroup, Server
from laboneq.dsl.device.physical_channel_group import (
    PhysicalChannel,
    PhysicalChannelGroup,
)
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.section import Section
from laboneq.dsl.parameter import Parameter
from laboneq.dsl.prng import PRNGSample


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
                Precompensation,
                AmplifierPump,
                Instrument,
                LogicalSignalGroup,
                PhysicalChannelGroup,
                PhysicalChannel,
                Server,
                PRNGSample,
            )
        )
        entity_mapper = {
            "Section": "sections",
            "Pulse": "pulses",
            "Parameter": "parameters",
            "Oscillator": "oscillators",
            "MixerCalibration": "mixer_calibrations",
            "Precompensation": "precompensations",
            "Instrument": "instruments",
            "LogicalSignalGroup": "logical_signal_groups",
            "PhysicalChannelGroup": "physical_channel_groups",
            "PhysicalChannel": "physical_channels",
            "Server": "servers",
            "PRNGSample": "prng_samples",
        }

        return entity_classes, entity_mapper

    @staticmethod
    def to_json_struct(serializable_object, omit_none_fields=False) -> str:
        entity_classes, entity_mapper = Serializer._entity_config()

        json_struct = serialize_to_dict_with_ref(
            serializable_object,
            entity_classes,
            Serializer._classes_by_short_name(),
            entity_mapper,
            emit_enum_types=True,
            omit_none_fields=omit_none_fields,
        )

        return json_struct

    @staticmethod
    def to_json(serializable_object, omit_none_fields=False) -> str:
        options = orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY
        try:
            if isinstance(serializable_object, dict):
                json_dump = orjson.dumps(serializable_object, option=options)
            else:
                entity_classes, entity_mapper = Serializer._entity_config()

                json_struct = serialize_to_dict_with_ref(
                    serializable_object,
                    entity_classes,
                    Serializer._classes_by_short_name(),
                    entity_mapper,
                    emit_enum_types=True,
                    omit_none_fields=omit_none_fields,
                )
                json_dump = orjson.dumps(json_struct, option=options)
            return json_dump.decode()
        except TypeError as ex:
            raise LabOneQException(
                f"Serializing dictionaries with non integer keys is not supported: {ex}"
            ) from ex

    @staticmethod
    def to_dict(serializable_object, omit_none_fields=False) -> Dict:
        entity_classes, entity_mapper = Serializer._entity_config()

        return serialize_to_dict_with_ref(
            serializable_object,
            entity_classes,
            Serializer._classes_by_short_name(),
            entity_mapper,
            emit_enum_types=True,
            omit_none_fields=omit_none_fields,
        )

    @staticmethod
    def to_json_file(serializable_object, filename: str):
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
            "laboneq.dsl.quantum.quantum_element",
            "laboneq.dsl.quantum.quantum_operation",
            "laboneq.dsl.quantum.qubit",
            "laboneq.dsl.quantum.transmon",
            "laboneq.dsl.device.server",
            "laboneq.dsl.device.servers.data_server",
            "laboneq.core.serialization.simple_serialization",
            "laboneq.core.types.enums",
            "laboneq.core.types.compiled_experiment",
            "laboneq.data.scheduled_experiment",
            "laboneq.data.recipe",
            "laboneq.executor.executor",
            "laboneq.dsl.device.io_units.logical_signal",
            "laboneq.dsl.device.io_units.physical_channel",
            "laboneq.dsl.device.instruments",
            "laboneq.dsl.calibration.units",
        ]
        schedule_modules = [
            "laboneq.compiler.scheduler.scheduler",
            "laboneq.compiler.ir.ir",
        ]
        _, classes_by_short_name = module_classes(dsl_modules + schedule_modules)
        # TODO: remove this after migration to new data types is complete (?)
        _, classes_by_short_name_compilation_job = module_classes(
            ["laboneq.data.compilation_job"],
            class_names=[
                "DeviceInfoType",
                "AmplifierPumpInfo",
                "DeviceInfo",
                "OscillatorInfo",
                "SignalInfo",
                "MixerCalibrationInfo",
                "PrecompensationInfo",
                "PulseDef",
                "FollowerInfo",
                "AcquireInfo",
                "SignalRange",
                "Marker",
            ],
        )
        return OrderedDict(
            chain(
                classes_by_short_name.items(),
                classes_by_short_name_compilation_job.items(),
            )
        )

    @staticmethod
    # NOTE(mr): This can be removed after the legacy adapters have been removed, or, conversely after class names are unique in L1Q once more
    def _classes_by_short_name_ir():
        _, classes_by_short_name = module_classes(
            [
                "laboneq.compiler.ir.ir",
                "laboneq.compiler.ir.acquire_group_ir",
                "laboneq.compiler.ir.case_ir",
                "laboneq.compiler.ir.interval_ir",
                "laboneq.compiler.ir.loop_ir",
                "laboneq.compiler.ir.loop_iteration_ir",
                "laboneq.compiler.ir.match_ir",
                "laboneq.compiler.ir.oscillator_ir",
                "laboneq.compiler.ir.phase_reset_ir",
                "laboneq.compiler.ir.pulse_ir",
                "laboneq.compiler.ir.reserve_ir",
                "laboneq.compiler.ir.root_ir",
                "laboneq.compiler.ir.section_ir",
                "laboneq.data.compilation_job",
                "laboneq.data.calibration",
                "laboneq.core.types.enums",
                "laboneq._utils",
            ]
        )
        return OrderedDict(classes_by_short_name.items())

    @staticmethod
    def from_json(serialized_string: str, type_hint):
        if type_hint is dict:
            obj = orjson.loads(serialized_string)
        else:
            entity_classes, entity_mapper = Serializer._entity_config()
            serialized_form = orjson.loads(serialized_string)

            obj = deserialize_from_dict_with_ref(
                serialized_form,
                Serializer._classes_by_short_name_ir()
                if type_hint is not None and type_hint.__name__ == "IR"
                else Serializer._classes_by_short_name(),
                entity_classes,
                entity_mapper,
            )

        return obj

    @staticmethod
    def load(data, type_hint):
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
