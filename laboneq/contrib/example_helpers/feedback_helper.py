# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Common helper functions for setting up a threshold for state discrimination and for feedback experiments
"""

import numpy as np

from laboneq.core.types.enums import AcquisitionType, AveragingMode
from laboneq.dsl.calibration import SignalCalibration
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.builtins import (
    acquire,
    acquire_loop_rt,
    delay,
    experiment,
    experiment_calibration,
    for_each,
    map_signal,
    play,
    section,
)


## create readout pulse waveform with IQ encoded phase and optional software modulation
def complex_freq_phase(
    sampling_rate: float,
    length: float,
    freq: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.typing.ArrayLike:
    time_axis = np.linspace(0, length, int(length * sampling_rate))
    return amplitude * np.exp(1j * 2 * np.pi * freq * time_axis + 1j * phase)


@experiment(signals=["measure", "acquire"])
def exp_raw(measure_pulse, q0, pulse_len):
    """Experiment for raw signal acquisition from measure pulse"""
    map_signal("measure", q0["measure_line"])
    map_signal("acquire", q0["acquire_line"])

    with acquire_loop_rt(count=1024, acquisition_type=AcquisitionType.RAW):
        play(signal="measure", pulse=measure_pulse)
        acquire(signal="acquire", handle="raw", length=pulse_len)
        delay(signal="measure", time=10e-6)


@experiment(signals=["measure0", "measure1", "acquire"])
def exp_integration(measure0, measure1, q0, q1, samples_kernel, rotation_angle=0):
    """Experiment that plays two different measure pulses one after another and acquire
    results in single shot integration mode use custom integration kernel for data
    acquisition.
    """
    kernel = pulse_library.sampled_pulse_complex(
        samples_kernel * np.exp(1j * rotation_angle)
    )

    map_signal("measure0", q0["measure_line"])
    map_signal("measure1", q1["measure_line"])
    map_signal("acquire", q0["acquire_line"])

    with acquire_loop_rt(
        count=1024,
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        with section():
            play(signal="measure0", pulse=measure0)
            acquire(signal="acquire", handle="data0", kernel=kernel)
            delay(signal="measure0", time=10e-6)
        with section():
            play(signal="measure1", pulse=measure1)
            acquire(signal="acquire", handle="data1", kernel=kernel)
            delay(signal="measure1", time=10e-6)


@experiment(signals=["measure0", "measure1", "acquire"])
def exp_discrimination(
    measure0, measure1, q0, q1, samples_kernel, threshold=0, rotation_angle=0, num=50
):
    """Experiment to test state discrimination by playing two different measure pulses
    one ofter the other and acquiring the state readout.
    """
    kernel = pulse_library.sampled_pulse_complex(
        samples_kernel * np.exp(1j * rotation_angle)
    )

    map_signal("measure0", q0["measure_line"])
    map_signal("measure1", q1["measure_line"])
    map_signal("acquire", q0["acquire_line"])
    experiment_calibration()["acquire"] = SignalCalibration(threshold=threshold)

    with acquire_loop_rt(count=1024, acquisition_type=AcquisitionType.DISCRIMINATION):
        with for_each(range(num)):
            with section():
                play(signal="measure0", pulse=measure0)
                acquire(signal="acquire", handle="data0", kernel=kernel)
                delay(signal="measure0", time=1e-6)
            with section():
                play(signal="measure1", pulse=measure1)
                acquire(signal="acquire", handle="data1", kernel=kernel)
                delay(signal="measure1", time=1e-6)
