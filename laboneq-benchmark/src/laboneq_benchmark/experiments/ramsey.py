# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.dsl import LinearSweepParameter
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    SectionAlignment,
)
from laboneq.dsl.experiment import (
    Experiment,
    ExperimentSignal,
)
from laboneq.dsl.quantum import QuantumElement

from .pulses import (
    drive_ge_pi_half,
    integration_kernel,
    readout_pulse,
)


# function that returns an amplitude Rabi experiment
def ramsey_parallel(
    qubits: list[QuantumElement],
    number_average: int = 2**12,
    sweep_max: float = 15e-6,
    number_sweep_steps: int = 201,
    chunk_count: int = 1,
    do_readout: bool = True,
):
    # wrap the experiment definition in an inner function to be able to use it as a callable
    def inner():
        exp_ramsey = Experiment(
            uid="Qubit Spectroscopy",
            signals=[
                signal
                for signal_list in [
                    [
                        ExperimentSignal(
                            f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                        ),
                        ExperimentSignal(
                            f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                        ),
                        ExperimentSignal(
                            f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                        ),
                    ]
                    for qubit in qubits
                ]
                for signal in signal_list
            ],
        )
        delay_sweep = LinearSweepParameter(
            start=0, stop=sweep_max, count=number_sweep_steps
        )
        ## define Ramsey experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        with exp_ramsey.acquire_loop_rt(
            uid="ramsey_shots",
            count=number_average,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Ramsey time delays
            with exp_ramsey.sweep(
                uid="ramsey_sweep",
                parameter=delay_sweep,
                alignment=SectionAlignment.RIGHT,
                chunk_count=chunk_count,
            ):
                for qubit in qubits:
                    # play qubit excitation pulse - pulse amplitude is swept
                    ramsey_pulse = drive_ge_pi_half(qubit)
                    with exp_ramsey.section(
                        uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                    ):
                        exp_ramsey.play(signal=f"drive_{qubit.uid}", pulse=ramsey_pulse)
                        exp_ramsey.delay(signal=f"drive_{qubit.uid}", time=delay_sweep)
                        exp_ramsey.play(signal=f"drive_{qubit.uid}", pulse=ramsey_pulse)
                    if do_readout:
                        # readout pulse and data acquisition
                        with exp_ramsey.section(
                            uid=f"readout_{qubit.uid}",
                            play_after=f"{qubit.uid}_excitation",
                        ):
                            exp_ramsey.play(
                                signal=f"measure_{qubit.uid}",
                                pulse=readout_pulse(qubit),
                            )
                            exp_ramsey.acquire(
                                signal=f"acquire_{qubit.uid}",
                                handle=f"{qubit.uid}_rabi",
                                length=integration_kernel(qubit).length,
                            )
                        with exp_ramsey.section(
                            uid=f"readout_delay_{qubit.uid}",
                            play_after=f"readout_{qubit.uid}",
                        ):
                            exp_ramsey.delay(signal=f"measure_{qubit.uid}", time=100e-9)

        return exp_ramsey

    return inner
