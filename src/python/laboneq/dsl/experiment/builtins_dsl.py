# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q builtins recommended for use with the LabOne Q Applications library.

This is intended to be the equivalent of `laboneq.simple` for the LabOne Q
builtins, `laboneq.dsl.experiment.builtins`.
"""

__all__ = [  # noqa: RUF022
    # builtins:
    "acquire",
    "acquire_loop_rt",
    "active_section",
    "add",
    "call",
    "case",
    "delay",
    "experiment",
    "experiment_calibration",
    "for_each",
    "map_coprocessor",
    "map_signal",
    "match",
    "measure",
    "play",
    "reserve",
    "reset_oscillator_phase",
    "section",
    "set_node",
    "sweep",
    "sweep_range",
    "uid",
    # pulse_library:
    "pulse_library",
    # build experiments
    "add_quantum_elements",
    "add_signal",
    "qubit_experiment",
    # formatter:
    "handles",
    # core quantum
    "QuantumOperations",
    "create_pulse",
    "quantum_operation",
    # HQCS
    "Variable",
    "Struct",
    "register_stream",
    "set_coprocessor_payload",
    "send",
    "mark_stale",
    "is_live",
    "ensure",
    "do_until",
    "render_layout",
    "Phase",
    "Frequency",
    "Amplitude",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "DiscriminationDataPacked",
    "IqDataPacked",
    "ScopeShot",
    "WaveformUpdate",
    "LoggedVariable",
]

# HQCS coprocessor surface
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
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.build_experiment import (
    add_quantum_elements,
    add_signal,
    qubit_experiment,
)
from laboneq.dsl.experiment.builtins import (
    acquire,
    acquire_loop_rt,
    active_section,
    add,
    call,
    case,
    delay,
    experiment,
    experiment_calibration,
    for_each,
    map_coprocessor,
    map_signal,
    match,
    measure,
    play,
    reserve,
    reset_oscillator_phase,
    section,
    set_node,
    sweep,
    sweep_range,
    uid,
)
from laboneq.dsl.experiment.do_until import do_until
from laboneq.dsl.quantum.quantum_operations import (
    QuantumOperations,
    create_pulse,
    quantum_operation,
)
from laboneq.dsl.result.logged_variable import LoggedVariable
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
from laboneq.workflow import handles
