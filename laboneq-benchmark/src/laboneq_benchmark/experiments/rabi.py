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
    drive_ge_rabi,
    integration_kernel,
    readout_pulse,
)


# Amplitude Rabi experiment including measurement
def amplitude_rabi_parallel(
    qubits: list[QuantumElement],
    sweep_max: float = 0.8,
    number_sweep_steps: int = 201,
    number_average: int = 2**12,
    chunk_count: int = 1,
    do_readout: bool = True,
):
    # wrap the experiment definition in an inner function to be able to use it as a callable
    def inner():
        exp_rabi = Experiment(
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
        amplitude_sweep = LinearSweepParameter(
            start=0, stop=sweep_max, count=number_sweep_steps
        )
        ## define Rabi experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=number_average,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi amplitudes
            with exp_rabi.sweep(
                uid="rabi_sweep", parameter=amplitude_sweep, chunk_count=chunk_count
            ):
                for qubit in qubits:
                    # qubit drive
                    with exp_rabi.section(
                        uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                    ):
                        exp_rabi.play(
                            signal=f"drive_{qubit.uid}",
                            pulse=drive_ge_rabi(qubit),
                            amplitude=amplitude_sweep,
                        )
                    if do_readout:
                        # measurement
                        with exp_rabi.section(
                            uid=f"readout_{qubit.uid}",
                            play_after=f"{qubit.uid}_excitation",
                        ):
                            exp_rabi.play(
                                signal=f"measure_{qubit.uid}",
                                pulse=readout_pulse(qubit),
                            )
                            exp_rabi.acquire(
                                signal=f"acquire_{qubit.uid}",
                                handle=f"{qubit.uid}_rabi",
                                length=integration_kernel(qubit).length,
                            )
                        with exp_rabi.section(
                            uid=f"readout_delay_{qubit.uid}",
                            play_after=f"readout_{qubit.uid}",
                        ):
                            exp_rabi.delay(signal=f"measure_{qubit.uid}", time=100e-9)

        return exp_rabi

    return inner
