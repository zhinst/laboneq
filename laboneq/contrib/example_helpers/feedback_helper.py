# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Common helper functions for setting up a threshold for state discrimination and for feedback experiments
"""

import numpy as np

from laboneq.core.types.enums import AcquisitionType
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.builtins import (
    acquire,
    acquire_loop_rt,
    delay,
    experiment,
    for_each,
    map_signal,
    play,
    section,
)


class state_emulation_pulse:
    """Class containing parameters for a pulse that can be used to emulate different qubit states"""

    def __init__(
        self,
        # qubit_state=0,
        pulse_length=400e-9,
        amplitude_increment=0.2,
        phase_increment=np.pi / 2.5,
    ):
        self.pulse_length = pulse_length
        self.pulse = pulse_library.const(length=pulse_length)
        self.amplitude_increment = amplitude_increment
        self.phase_increment = phase_increment

    def pulse_phase(self, qubit_state=0):
        return qubit_state * self.phase_increment

    def pulse_amplitude(self, qubit_state=0):
        return self.amplitude_increment * (qubit_state + 1)


def create_calibration_experiment(
    state_emulation_pulse,
    qubit_state,
    measure_signal,
    acquire_signal,
):
    """Experiment to calibrate state discrimination by playing a measurement pulse
    that emulates a certain qubit state response and acquiring the raw trace of the returned signal.
    """

    @experiment(signals=["measure", "acquire"])
    def exp():
        map_signal("measure", measure_signal)
        map_signal("acquire", acquire_signal)

        with acquire_loop_rt(count=1024, acquisition_type=AcquisitionType.RAW):
            play(
                signal="measure",
                pulse=state_emulation_pulse.pulse,
                phase=state_emulation_pulse.pulse_phase(qubit_state),
                amplitude=state_emulation_pulse.pulse_amplitude(qubit_state),
            )
            acquire(
                signal="acquire",
                handle="raw",
                length=state_emulation_pulse.pulse_length,
            )
            delay(signal="measure", time=1e-6)

    return exp()


def create_discrimination_experiment(
    measure_lines,
    acquire_line,
    kernels,
    state_emulation_pulse,
    num=10,
):
    """Experiment to test state discrimination by playing a number of different measure pulses
    one ofter the other and acquiring the state readout.
    """

    all_signals = [f"measure_{it}" for it in range(len(measure_lines))]
    all_signals.append("acquire")

    @experiment(signals=all_signals)
    def exp():
        for it, line in enumerate(measure_lines):
            map_signal(f"measure_{it}", line)
        map_signal("acquire", acquire_line)

        measure_pulse = state_emulation_pulse()

        with acquire_loop_rt(
            count=1024, acquisition_type=AcquisitionType.DISCRIMINATION
        ):
            with for_each(range(num)):
                for it, _ in enumerate(measure_lines):
                    with section(
                        uid=f"measure_{it}",
                        play_after=None if it == 0 else f"measure_{it-1}",
                    ):
                        play(
                            signal=f"measure_{it}",
                            pulse=measure_pulse.pulse,
                            phase=measure_pulse.pulse_phase(qubit_state=it),
                            amplitude=measure_pulse.pulse_amplitude(qubit_state=it),
                        )
                        acquire(signal="acquire", handle=f"data_{it}", kernel=kernels)
                        delay(signal=f"measure_{it}", time=0.5e-6)

    return exp()


def gaussian_envelope(
    centre=50e-9, sigma=20e-9, start_time=0, stop_time=100e-9, sampling_rate=2e9
):
    """Gaussian waveform envelope, to be used to construct piecewise modualted pulses"""
    times = np.linspace(
        start=start_time,
        stop=stop_time,
        num=int((stop_time - start_time) * sampling_rate),
    )
    return np.exp(-((times - centre) ** 2) / (2 * sigma**2))


def piecewise_modulated(
    piece_length=(100e-9, 100e-9),
    piece_frequency=(-200e6, 0),
    piece_amplitude=(0.6, 0.3),
    waveform_envelope=gaussian_envelope,
    sampling_rate=2e9,
):
    """Definition of a piecewise modulated sampled waveform.
    Each section of the waveform has a length in time, a modualtion frequency and an amplitude.
    All sections have the same envelope.
    """
    values = np.array([])
    piece_start = 0

    for amplitude, frequency, pulse_length in zip(
        piece_amplitude, piece_frequency, piece_length
    ):
        times = np.linspace(
            start=0,
            stop=pulse_length,
            num=int(pulse_length * sampling_rate),
        )
        values = np.append(
            values,
            amplitude
            * np.exp(-1j * 2 * np.pi * frequency * times)
            * waveform_envelope(
                start_time=piece_start,
                stop_time=piece_start + pulse_length,
                centre=piece_start + pulse_length / 2,
                sampling_rate=sampling_rate,
            ),
        )
        piece_start += pulse_length

    return values
