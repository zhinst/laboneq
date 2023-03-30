# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Common helper functions for setting up a threshold for state discrimination and for feedback experiments
"""

import numpy as np

from laboneq.core.types.enums import AcquisitionType, AveragingMode
from laboneq.dsl.calibration import SignalCalibration
from laboneq.dsl.experiment import Experiment, pulse_library
from laboneq.dsl.parameter import LinearSweepParameter


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


## Experiment for raw signal acquisition from measure pulse
def exp_raw(measure_pulse, q0, pulse_len):
    exp = Experiment(signals=["measure", "acquire"])
    exp.map_signal("measure", q0["measure_line"])
    exp.map_signal("acquire", q0["acquire_line"])

    with exp.acquire_loop_rt(count=1024, acquisition_type=AcquisitionType.RAW):
        exp.play(signal="measure", pulse=measure_pulse)
        exp.acquire(signal="acquire", handle="raw", length=pulse_len)
        exp.delay(signal="measure", time=10e-6)

    return exp


## Experiment that plays two different measure pulses one after another and acquire results in single shot integration mode
## use custom integration kernel for data acquisition
def exp_integration(measure0, measure1, q0, q1, samples_kernel, rotation_angle=0):
    kernel = pulse_library.sampled_pulse_complex(
        samples_kernel * np.exp(1j * rotation_angle)
    )

    exp = Experiment(signals=["measure0", "measure1", "acquire"])
    exp.map_signal("measure0", q0["measure_line"])
    exp.map_signal("measure1", q1["measure_line"])
    exp.map_signal("acquire", q0["acquire_line"])

    with exp.acquire_loop_rt(
        count=1024,
        averaging_mode=AveragingMode.SINGLE_SHOT,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        with exp.section():
            exp.play(signal="measure0", pulse=measure0)
            exp.acquire(signal="acquire", handle="data0", kernel=kernel)
            exp.delay(signal="measure0", time=10e-6)
        with exp.section():
            exp.play(signal="measure1", pulse=measure1)
            exp.acquire(signal="acquire", handle="data1", kernel=kernel)
            exp.delay(signal="measure1", time=10e-6)
    return exp


## Experiment to test state discrimination by playing two different measure pulses one ofter the other and acquiring the state readout
def exp_discrimination(
    measure0, measure1, q0, q1, samples_kernel, threshold=0, rotation_angle=0, num=50
):
    kernel = pulse_library.sampled_pulse_complex(
        samples_kernel * np.exp(1j * rotation_angle)
    )

    exp = Experiment(signals=["measure0", "measure1", "acquire"])
    exp.map_signal("measure0", q0["measure_line"])
    exp.map_signal("measure1", q1["measure_line"])
    exp.map_signal("acquire", q0["acquire_line"])

    exp.signals["acquire"].calibration = SignalCalibration(threshold=threshold)

    repeat = LinearSweepParameter(start=0, stop=num - 1, count=num)

    with exp.acquire_loop_rt(
        count=1024, acquisition_type=AcquisitionType.DISCRIMINATION
    ):
        with exp.sweep(parameter=repeat):
            with exp.section():
                exp.play(signal="measure0", pulse=measure0)
                exp.acquire(signal="acquire", handle="data0", kernel=kernel)
                # with exp.match_local(handle="data0"):
                #     with exp.case(0):
                #         exp.play()
                #     with exp.case(1):
                #         pass
                exp.delay(signal="measure0", time=100e-9)
            with exp.section():
                exp.play(signal="measure1", pulse=measure1)
                exp.acquire(signal="acquire", handle="data1", kernel=kernel)
                exp.delay(signal="measure1", time=100e-9)

    return exp
