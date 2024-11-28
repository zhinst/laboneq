# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.simple import (
    AcquisitionType,
    Experiment,
    LinearSweepParameter,
    pulse_library,
)


# simple qubit frequency spectroscopy used to demonstrate signal calibration in examples/02_logical_signals/02_calibration.ipynb
def make_reference_experiment():
    # define readout pulse
    readout_pulse = pulse_library.gaussian_square(
        uid="readout_pulse_q0", length=2.0e-6, amplitude=0.9, width=1.0e-6, sigma=0.2
    )
    # define drive pulse
    drive_pulse = pulse_library.const(
        uid="drive_spec_pulse_q0", length=1.0e-6, amplitude=0.9
    )

    # define frequency sweep parameter
    freq_sweep = LinearSweepParameter(
        uid="drive_freq_q0", start=-1e8, stop=1e8, count=1001
    )

    # initialize a qubit spectroscopy Experiment
    experiment_02 = Experiment(signals=["drive", "measure", "acquire"])

    # inner loop - real-time averaging - QA in integration mode
    with experiment_02.acquire_loop_rt(
        count=128, acquisition_type=AcquisitionType.INTEGRATION
    ):
        with experiment_02.sweep(uid="qfreq_sweep", parameter=freq_sweep):
            # qubit drive
            with experiment_02.section(uid="qubit_excitation"):
                experiment_02.play(signal="drive", pulse=drive_pulse)
            with experiment_02.section(
                uid="readout_section", play_after="qubit_excitation"
            ):
                # play readout pulse on measure line
                experiment_02.play(signal="measure", pulse=readout_pulse)
                # trigger signal data acquisition
                experiment_02.acquire(
                    signal="acquire",
                    handle="qb_spec",
                    kernel=readout_pulse,
                )
            # relax time after readout - for qubit relaxation to groundstate and signal processing
            with experiment_02.section(uid="relax", length=1e-6):
                experiment_02.reserve(signal="measure")

    return experiment_02
