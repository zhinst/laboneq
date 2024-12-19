# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from itertools import chain

from laboneq.core.serialization.simple_serialization import module_classes
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


def entity_config():
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


def classes_by_short_name() -> OrderedDict[str, type]:
    dsl_modules = [
        "laboneq.dsl.experiment",
        "laboneq.dsl.experiment.pulse",
        "laboneq.dsl.calibration.amplifier_pump",
        "laboneq.dsl.calibration.oscillator",
        "laboneq.dsl.calibration.signal_calibration",
        "laboneq.dsl.result.results",
        "laboneq.dsl.parameter",
        "laboneq.dsl.calibration",
        "laboneq.dsl.device",
        "laboneq.dsl.quantum.quantum_element",
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


# NOTE(mr): This can be removed after the legacy adapters have been removed, or, conversely after class names are unique in LabOne Q once more
def classes_by_short_name_ir() -> OrderedDict[str, type]:
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
