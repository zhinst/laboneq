# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""
Convenience header for the LabOne Q project.
"""

from laboneq import laboneq_logging, workflow
from laboneq.controller.api.async_remote_controller import AsyncRemoteController
from laboneq.controller.api.remote_controller import (
    RemoteController,
    get_instrument_topology,
)
from laboneq.controller.runtime_context import RuntimeContext
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.core.utilities.compile_experiment import (
    compile_experiment,
    laboneq_compile,
)
from laboneq.dsl import LinearSweepParameter, SweepParameter
from laboneq.dsl.calibration import (
    AmplifierPump,
    BounceCompensation,
    Calibratable,
    Calibration,
    CancellationSource,
    ExponentialCompensation,
    FIRCompensation,
    HighPassCompensation,
    MixerCalibration,
    Oscillator,
    OutputRoute,
    Precompensation,
    SignalCalibration,
    units,
)
from laboneq.dsl.coprocessor.builtins import (
    ensure,
    is_live,
    mark_stale,
    register_stream,
    render_layout,
    send,
    set_coprocessor_payload,
)
from laboneq.dsl.coprocessor.struct import Struct
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.device_setup_helper import DeviceSetupHelper
from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    QHUB,
    SHFPPC,
    SHFQA,
    SHFQC,
    SHFSG,
    UHFQA,
)
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    CarrierType,
    ExecutionType,
    ModulationType,
    PortMode,
    RepetitionMode,
    SectionAlignment,
    SectionTimingMode,
)
from laboneq.dsl.experiment import (
    AcquireLoopRt,
    Case,
    Experiment,
    ExperimentSignal,
    Match,
    Section,
    Sweep,
    pulse_library,
)
from laboneq.dsl.experiment import builtins_dsl as dsl
from laboneq.dsl.experiment.do_until import do_until
from laboneq.dsl.quantum import (
    QPU,
    QPUTopology,
    QuantumElement,
    QuantumParameters,
    QuantumPlatform,
    Qubit,
    QubitParameters,
    Transmon,
    TransmonParameters,
)
from laboneq.dsl.result import Results
from laboneq.dsl.result.logged_variable import LoggedVariable
from laboneq.dsl.session import Session
from laboneq.dsl.utils import has_onboard_lo
from laboneq.dsl.variable import Variable
from laboneq.dsl.variable.types import (
    Amplitude,
    DiscriminationDataPacked,
    Frequency,
    Int8,
    Int16,
    Int32,
    Int64,
    IqDataPacked,
    Phase,
    ScopeShot,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    WaveformUpdate,
)
from laboneq.openqasm3 import ExternResult, GateStore, exp_from_qasm, exp_from_qasm_list
from laboneq.pulse_sheet_viewer.pulse_sheet_viewer import show_pulse_sheet
from laboneq.serializers import from_dict, from_json, load, save, to_dict, to_json
from laboneq.simulator.output_simulator import OutputSimulator

__all__ = [
    "HDAWG",
    "PQSC",
    "QHUB",
    "QPU",
    "SHFPPC",
    "SHFQA",
    "SHFQC",
    "SHFSG",
    "UHFQA",
    "AcquireLoopRt",
    "AcquisitionType",
    "AmplifierPump",
    "Amplitude",
    "AsyncRemoteController",
    "AveragingMode",
    "BounceCompensation",
    "Calibratable",
    "Calibration",
    "CancellationSource",
    "CarrierType",
    "Case",
    "CompiledExperiment",
    "DeviceSetup",
    "DeviceSetupHelper",
    "DiscriminationDataPacked",
    "ExecutionType",
    "Experiment",
    "ExperimentSignal",
    "ExponentialCompensation",
    "ExternResult",
    "FIRCompensation",
    "Frequency",
    "GateStore",
    "HighPassCompensation",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "IqDataPacked",
    "LinearSweepParameter",
    "LoggedVariable",
    "Match",
    "MixerCalibration",
    "ModulationType",
    "Oscillator",
    "OutputRoute",
    "OutputSimulator",
    "Phase",
    "PortMode",
    "Precompensation",
    "QPUTopology",
    "QuantumElement",
    "QuantumParameters",
    "QuantumPlatform",
    "Qubit",
    "QubitParameters",
    "RemoteController",
    "RepetitionMode",
    "Results",
    "RuntimeContext",
    "ScopeShot",
    "Section",
    "SectionAlignment",
    "SectionTimingMode",
    "Session",
    "SignalCalibration",
    "Struct",
    "Sweep",
    "SweepParameter",
    "Transmon",
    "TransmonParameters",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Variable",
    "WaveformUpdate",
    "compile_experiment",
    "create_connection",
    "do_until",
    "dsl",
    "ensure",
    "exp_from_qasm",
    "exp_from_qasm_list",
    "from_dict",
    "from_json",
    "get_instrument_topology",
    "has_onboard_lo",
    "is_live",
    "laboneq_compile",
    "laboneq_logging",
    "load",
    "mark_stale",
    "pulse_library",
    "register_stream",
    "render_layout",
    "save",
    "send",
    "set_coprocessor_payload",
    "show_pulse_sheet",
    "to_dict",
    "to_json",
    "units",
    "workflow",
]
