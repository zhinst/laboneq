# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Simple Qubit class for simplifying device setup calibration
"""

from copy import deepcopy

from laboneq.core.types.enums import ModulationType
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.experiment import pulse_library


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_qubit_parameters(base_parameters, id):
    parameters = deepcopy(base_parameters)
    parameters["frequency"] = base_parameters["frequency"][id]
    parameters["readout_frequency"] = base_parameters["readout_frequency"][id]
    parameters["drive_lo_frequency"] = base_parameters["drive_lo_frequency"][id // 2]

    return parameters


class QubitParameters:
    def __init__(self, my_parameter_dict):
        for key in my_parameter_dict.keys():
            setattr(self, key, my_parameter_dict[key])


class QubitPulses:
    def __init__(self, id, parameters: QubitParameters):
        self.qubit_x90 = pulse_library.drag(
            uid=f"x90_q{id}",
            length=parameters.pulse_length,
            amplitude=parameters.pi_2_amplitude,
            sigma=0.3,
            beta=0.4,
        )
        self.qubit_x180 = pulse_library.drag(
            uid=f"x180_q{id}",
            length=parameters.pulse_length,
            amplitude=parameters.pi_amplitude,
            sigma=0.3,
            beta=0.4,
        )
        self.readout_pulse = pulse_library.const(
            uid=f"readout_pulse_q{id}",
            length=parameters.readout_length,
            amplitude=parameters.readout_amplitude,
        )
        self.readout_weights = pulse_library.const(
            uid=f"readout_weights_q{id}",
            length=parameters.readout_length,
            amplitude=1,
        )


class Qubit:
    def __init__(self, id, parameter_dict):
        self.id = id

        self.parameters = QubitParameters(parameter_dict)

        self.pulses = QubitPulses(self.id, self.parameters)


# define baseline signal calibration for a list of qubits
def define_calibration(qubits):
    calib = Calibration()

    for _, qubit in enumerate(qubits):
        calib[f"/logical_signal_groups/q{qubit.id}/drive_line"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=qubit.parameters.frequency,
                modulation_type=ModulationType.HARDWARE,
            ),
            local_oscillator=Oscillator(
                frequency=qubit.parameters.drive_lo_frequency,
            ),
            range=qubit.parameters.drive_range,
        )
        calib[f"/logical_signal_groups/q{qubit.id}/measure_line"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=qubit.parameters.readout_frequency,
                modulation_type=ModulationType.SOFTWARE,
            ),
            local_oscillator=Oscillator(
                frequency=qubit.parameters.readout_lo_frequency,
            ),
            range=qubit.parameters.readout_range_out,
        )
        calib[f"/logical_signal_groups/q{qubit.id}/acquire_line"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=qubit.parameters.readout_frequency,
                modulation_type=ModulationType.SOFTWARE,
            ),
            local_oscillator=Oscillator(
                frequency=qubit.parameters.readout_lo_frequency,
            ),
            range=qubit.parameters.readout_range_in,
            port_delay=qubit.parameters.readout_integration_delay,
        )

    return calib
