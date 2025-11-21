# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np
from numpy.typing import NDArray

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.dsl.parameter import SweepParameter
from laboneq.simple import (
    AveragingMode,
    Calibration,
    Experiment,
    ModulationType,
    Oscillator,
    SectionAlignment,
    SignalCalibration,
)
from laboneq.simple import pulse_library as pl


def create_qubit_spectroscopy(settings: QubitSpectroscopySettings):
    """Create a Qubit Spectroscopy experiment parallelized to multiple qubits.

    The returned experiment will have signals of the form:

    * {qubit_id}_drive
    * {qubit_id}_measure
    * {qubit_id}_acquire

    and acquisition handles of the form:

    * ac_{qubit_id}

    per qubit id provided in the `settings` object.

    Notes:

    1. Sweeps will be performed in real-time with local oscillator setting
       fixed per qubit, as provided in settings.

    """
    builder = QubitSpectroscopyBuilder()
    return builder.build(settings)


@attrs.frozen(kw_only=True)
class QubitSpectroscopySettings:
    """Settings for a qubit spectroscopy experiment parallelized to
    multiple qubits.

    Attributes:
        qubit_ids: Labels for the qubits, e.g. ["q0", "q1", ...].
        n_average: Number of repetitions of the full experiment.  Only the final averaged
            results are kept.
        excitation_sweep_frequencies: Mapping from qubit id to the frequency of
            drive signals to be swept. These are the absolute frequencies of the
            signal that goes out the wire.
        excitation_pulse: Pulse used in drive signals.
        readout_functional: Functional to be used for measure pulses and acquisition kernels.
        relaxation_length: Relaxation time between iterations.
        readout_frequencies: Mapping from qubit id to the frequency of readout
            signals. These are the absolute frequencies of the signal that goes out
            the wire.
        sg_lo_frequencies: Mapping from qubit id to drive signal (non-local)
            oscillator frequencies in LabOne Q parlance. This determines the
            center frequency for the up/downconversion on the hardware.
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
    excitation_sweep_frequencies: (
        dict[str, list[float]] | dict[str, NDArray[np.float64]]
    ) = attrs.field()
    excitation_pulse: pl.PulseFunctional
    readout_functional: pl.PulseFunctional
    relaxation_length: float
    readout_frequencies: dict[str, float] = attrs.field()
    sg_lo_frequencies: dict[str, float] = attrs.field()
    qa_lo_frequencies: dict[str, float] = attrs.field()
    acquisition_type: AcquisitionType = attrs.field()
    averaging_mode: AveragingMode = attrs.field()
    modulation_type: ModulationType = attrs.field()

    @excitation_sweep_frequencies.validator  # type: ignore
    def _check_frequencies(self, attrib, value: dict):
        if not value:
            raise ValueError(f"'{attrib.name}' must contain at least one item.")

        lengths = [len(v) for v in value.values()]

        if any(l != lengths[0] for l in lengths[1:]):
            raise ValueError(f"'{attrib.name}' items must be of same size.")

    @excitation_sweep_frequencies.validator  # type: ignore
    @readout_frequencies.validator  # type: ignore
    @sg_lo_frequencies.validator  # type: ignore
    @qa_lo_frequencies.validator  # type: ignore
    def _has_qubit_ids(self, attribute: attrs.Attribute, value: dict):
        if set(value.keys()) != set(self.qubit_ids):
            raise ValueError(f"'{attribute.name}' keys must be equal to `qubit_ids`. ")


@attrs.define
class QubitSpectroscopyBuilder:
    __experiment_name__: ClassVar[str] = "QubitSpectroscopy"
    __required_signal_types__: ClassVar[frozenset[str]] = frozenset(
        {"drive", "measure", "acquire"}
    )

    def build(self, settings: QubitSpectroscopySettings) -> Experiment:
        signals = [
            f"{q}_{signal_type}"
            for q in settings.qubit_ids
            for signal_type in self.__required_signal_types__
        ]

        exp = Experiment(
            uid=self.__experiment_name__, name=self.__experiment_name__, signals=signals
        )

        osc_frequency_sweep_parameters = {}

        for qid, freqs in settings.excitation_sweep_frequencies.items():
            values = np.array(
                [freq - settings.sg_lo_frequencies[qid] for freq in freqs]
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
                    with exp.section(
                        alignment=SectionAlignment.RIGHT,
                    ) as excitation:
                        exp.play(
                            signal=f"{qubit_id}_drive",
                            pulse=settings.excitation_pulse,
                        )
                    with exp.section(play_after=excitation) as readout:
                        exp.play(
                            signal=f"{qubit_id}_measure",
                            pulse=settings.readout_functional,
                        )
                        exp.acquire(
                            signal=f"{qubit_id}_acquire",
                            handle=f"ac_{qubit_id}",
                            kernel=settings.readout_functional,
                        )
                    with exp.section(
                        length=settings.relaxation_length, play_after=readout
                    ):
                        exp.reserve(f"{qubit_id}_measure")
                        exp.reserve(f"{qubit_id}_drive")

        exp.set_calibration(
            self._make_calibration(
                settings.qubit_ids,
                osc_frequency_sweep_parameters,
                settings.sg_lo_frequencies,
                settings.readout_frequencies,
                settings.qa_lo_frequencies,
                settings.modulation_type,
            )
        )
        return exp

    def _make_calibration(
        self,
        qubit_ids: list[str],
        sg_if_frequency_parameters: dict[str, SweepParameter],
        sg_lo_frequencies: dict[str, float],
        readout_frequencies: dict[str, float],
        qa_lo_frequencies: dict[str, float],
        modulation_type: ModulationType,
    ):
        exp_cal = Calibration()

        for qubit_id in qubit_ids:
            flo_sg, f_ro, flo_qa = (
                sg_lo_frequencies[qubit_id],
                readout_frequencies[qubit_id],
                qa_lo_frequencies[qubit_id],
            )
            sg_if_freq_param = sg_if_frequency_parameters[qubit_id]
            exp_cal[f"{qubit_id}_drive"] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=sg_if_freq_param, modulation_type=modulation_type
                ),
                local_oscillator=Oscillator(frequency=flo_sg),
            )

            exp_cal[f"{qubit_id}_measure"] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=f_ro - flo_qa, modulation_type=modulation_type
                ),
                local_oscillator=Oscillator(frequency=flo_qa),
            )

        return exp_cal
