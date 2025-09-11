# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Various pulse definitions for quantum experiments."""

from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.quantum import QuantumElement


def drive_ge_rabi(qubit: QuantumElement):
    return pulse_library.drag(
        uid=f"drag_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        sigma=0.4,
        beta=0.2,
        amplitude=1,
    )


def drive_ge_pi_half(qubit: QuantumElement):
    """Ramsey drive pulse."""
    return pulse_library.drag(
        uid=f"ramsey_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        sigma=0.4,
        beta=0.2,
        amplitude=qubit.parameters.user_defined["amplitude_pi"] / 2,
    )


def qubit_spectroscopy_pulse(qubit):
    return pulse_library.const(
        uid=f"spectroscopy_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=0.8,
    )


def readout_pulse(qubit: QuantumElement):
    return pulse_library.const(
        uid=f"readout_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )


def integration_kernel(qubit: QuantumElement):
    return pulse_library.const(
        uid=f"integration_kernel_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=1,
    )
