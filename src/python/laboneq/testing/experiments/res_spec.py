# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np
from numpy.typing import NDArray

from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Calibration,
    Experiment,
    ModulationType,
    Oscillator,
    SectionAlignment,
    SignalCalibration,
    SweepParameter,
)
from laboneq.simple import pulse_library as pl


def create_resonator_spectroscopy(settings: ResonatorSpectroscopySettings):
    """Create a pulsed resonator spectroscopy experiment parallelized to multiple qubits.

    The returned experiment will have signals of the form:

    * {qubit_id}_measure
    * {qubit_id}_acquire

    and acquisition handles of the form:

    * ac_{qubit_id}

    per qubit id provided in the `settings` object.

    Notes:

    1. Sweeps will be performed in real-time with local oscillator setting
       fixed per qubit, as provided in settings.

    """
    builder = ResonatorSpectroscopyBuilder()
    return builder.build(settings)


@attrs.frozen(kw_only=True)
class ResonatorSpectroscopySettings:
    """Settings for a resonator spectroscopy parallelized to multiple qubits.

    Attributes:
        qubit_ids: Labels for the qubits, e.g. ["q0", "q1", ...].
        n_average: Number of repetitions of the full experiment.  Only the
            final averaged results are kept.
        resonator_sweep_frequencies: Mapping from qubit id to the frequency of readout
            signals. These are the absolute frequencies of the signal that goes out
            the wire and that are swept to perform spectroscopy.
        readout_functional: Functional to be used for measure pulses and acquisition kernels.
        relaxation_length: Relaxation time.
        acquire_delay: Delay between the measure pulse and the beginning of the acquisitiong.
        qa_lo_frequencies: Mapping from readout signal (non-local) oscillator
            frequency in LabOne Q parlance. This determines the center
            frequency for the up/downconversion on the hardware.
        acquisition_type: Acquisition type of the real-time loop.
        averaging_type: Averaging type of the real-time loop.
        modulation_type: Modulation type for the intermediate frequency
            oscillator. WARNING: This may affect the pulse schedule.
    """

    qubit_ids: list[str]
    n_average: int
    resonator_sweep_frequencies: (
        dict[str, list[float]] | dict[str, NDArray[np.float64]]
    ) = attrs.field()
    readout_functional: pl.PulseFunctional
    relaxation_length: float
    acquire_delay: float
    qa_lo_frequencies: dict[str, float] = attrs.field()
    acquisition_type: AcquisitionType = attrs.field()
    averaging_mode: AveragingMode = attrs.field()
    modulation_type: ModulationType = attrs.field()

    @resonator_sweep_frequencies.validator  # type: ignore
    @qa_lo_frequencies.validator  # type: ignore
    def _has_qubit_ids(self, attribute: attrs.Attribute, value: dict):
        if set(value.keys()) != set(self.qubit_ids):
            raise ValueError(f"'{attribute.name}' keys must be equal to `qubit_ids`. ")


@attrs.define
class ResonatorSpectroscopyBuilder:
    __experiment_name__: ClassVar[str] = "ResonatorSpectroscopy"
    __required_signal_types__: ClassVar[frozenset[str]] = frozenset(
        {"measure", "acquire"}
    )

    def build(self, settings: ResonatorSpectroscopySettings) -> Experiment:
        signals = [
            f"{q}_{signal_type}"
            for q in settings.qubit_ids
            for signal_type in self.__required_signal_types__
        ]

        exp = Experiment(
            uid=self.__experiment_name__,
            name=self.__experiment_name__,
            signals=signals,
        )

        osc_frequency_sweep_parameters = {}

        for qid, freqs in settings.resonator_sweep_frequencies.items():
            values = np.array(
                [freq - settings.qa_lo_frequencies[qid] for freq in freqs]
            )
            osc_frequency_sweep_parameters[qid] = SweepParameter(values=values)

        with exp.acquire_loop_rt(
            count=settings.n_average,
            acquisition_type=settings.acquisition_type,
            averaging_mode=settings.averaging_mode,
        ):
            with exp.sweep(
                parameter=list(osc_frequency_sweep_parameters.values()),
                alignment=SectionAlignment.RIGHT,
            ):
                for qubit_id in settings.qubit_ids:
                    with exp.section(alignment=SectionAlignment.RIGHT) as excitation:
                        exp.play(
                            signal=f"{qubit_id}_measure",
                            pulse=settings.readout_functional,
                        )
                    with exp.section(play_after=excitation) as acquisition:
                        exp.acquire(
                            f"{qubit_id}_acquire",
                            handle=f"ac_{qubit_id}",
                            kernel=settings.readout_functional,
                        )
                    with exp.section(
                        length=settings.relaxation_length, play_after=acquisition
                    ):
                        exp.reserve(f"{qubit_id}_measure")
                        exp.reserve(f"{qubit_id}_acquire")

        exp.set_calibration(
            self._make_calibration(
                settings.qubit_ids,
                osc_frequency_sweep_parameters,
                settings.acquire_delay,
                settings.qa_lo_frequencies,
                settings.modulation_type,
            )
        )
        return exp

    def _make_calibration(
        self,
        qubit_ids: list[str],
        qa_if_frequency_parameters: dict[str, SweepParameter],
        acquire_delay: float,
        qa_lo_frequencies: dict[str, float],
        modulation_type: ModulationType,
    ):
        exp_cal = Calibration()

        for qubit_id in qubit_ids:
            flo_qa = qa_lo_frequencies[qubit_id]
            qa_if_freq_param = qa_if_frequency_parameters[qubit_id]
            exp_cal[f"{qubit_id}_measure"] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=qa_if_freq_param, modulation_type=modulation_type
                ),
                local_oscillator=Oscillator(frequency=flo_qa),
                delay_signal=acquire_delay,
            )

        return exp_cal
