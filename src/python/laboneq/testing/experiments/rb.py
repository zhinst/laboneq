# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import attrs
import numpy as np
import numpy.random as nprnd

from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    PauliGateMap,
    clifford_parametrized,
    generate_play_rb_pulses,
)
from laboneq.simple import (
    Calibration,
    Experiment,
    Oscillator,
    SectionAlignment,
    SignalCalibration,
    SweepParameter,
)

if TYPE_CHECKING:
    from laboneq.simple import (
        AcquisitionType,
        AveragingMode,
        ModulationType,
    )
    from laboneq.simple import pulse_library as pl

MAX_PAULI_GATE_PER_CLIFFORD = 3
assert max(map(len, clifford_parametrized)) == MAX_PAULI_GATE_PER_CLIFFORD


def create_rb_experiment(settings: RBExperimentSettings):
    """Create a randomized benchmarking experiment parallelized to multiple qubits.

    The returned experiment will have the signals:

    * {qubit_id}_drive
    * {qubit_id}_measure
    * {qubit_id}_acquire

    and acquisition handles:

    * ac_{qubit_id}

    per item in `settings.qubit_id`.

    The pseudo-code for the RB experiment returned:
    ```
    n_iter = len(sequence_lengths)
    n_qubits = len(qubits)

    # Prepare random clifford gates
    clifford_sequences = {}
    for m in range(n_iter):
        for k in range(n_repeat):
            for qubit in qubits:
                clifford_sequences[(m, k, qubit)] = pick_randomly(
                    clifford_set_for_1_qubit, sequence_lengths[m]
                )

    _results = array(dimensions=(n_average, n_iter, n_repeat, n_qubits))
    for n in range(n_average):
        for m, seq_length in enumerate(sequence_lengths):
            for k in range(n_repeat):
                for i, qubit in enumerate(qubits):
                    clifford_sequence = clifford_sequences[(m, k, qubit)]
                    apply(clifford_sequence, to=qubit)
                    _results[n, m, k, i] = measure(qubit)
    results = average(_results, over_dimension=0)
    ```
    The nesting shown is the one of `AveragingMode.CYCLIC`. Under
    `AveragingMode.SEQUENTIAL` the averaging loop is the innermost loop instead of the
    outermost one, so each sequence is averaged before the next one is played.

    """
    builder = RBExperimentBuilder()
    return builder.build(settings)


@attrs.frozen
class RBExperimentSettings:
    """Settings for a single-qubit randomized benchmark (RB) experiment parallelized to
    multiple qubits.

    Definitions:
        - A RB sequence of length `m` is an ordered collection of `m` randomly chosen
          Clifford gates, recovery gate, and a measurement; all scheduled to be applied
          in parallel on qubits with ids `qubit_ids`.
        - Repetition `k` is the k'th instance of RB sequence of length `m`.
        - Iteration `m` is `n_repeat` repetitions of a RB sequence of length `m`.
        - Shot `n` is the n'th "copy" of `len(sequence_lengths)` iterations.

    !!! tip
        To optimize use of waveform memory, ensure all lengths are a multiple of the
        system timing grid.

    Important to note here is that all shots are identical.

    To generate the random numbers, `numpy.random.default_rng` is used with the seed attribute.

    Attributes:
        seed: A non-negative integer for seeding the PRNG.
        qubit_ids: Labels for the qubits, e.g. ["q0", "q1", ...].
        sequence_lengths: An ordered set of numbers to use as sequence length at each
            RB iteration.  E.g., `sequence_lengths=[1, 2]`, implies 2 iteration RB with 1
            and 2 randomly chosen Clifford gates applied in each iteration.
        n_average: Number of repetitions of the full experiment.  Only the final averaged
            results are kept.
        n_repeat: Number of repetitions per sequence length.  Each repetition
            will yield a sample in the results.
        relaxation_length: Relaxation time between iterations.
        excitation_length: Duration of each drive (gate) pulse. Used to build the
            Clifford gate pulse map.
        pi_pulse_amp: Amplitude of the pi-rotation drive pulse.
        pi_half_pulse_amp: Amplitude of the pi/2-rotation drive pulse.
        prereadout_delay: Delay to apply before readout pulse.
        excitation_frequencies: Drive frequency of each qubit, keyed by qubit id.
        readout_frequencies: Readout frequency of each qubit, keyed by qubit id.
        sg_lo_frequencies: Local-oscillator frequency of each qubit's drive (SG)
            channel, keyed by qubit id. The drive IF is set to
            `excitation_frequency - sg_lo_frequency`.
        qa_lo_frequencies: Local-oscillator frequency of each qubit's readout (QA)
            channel, keyed by qubit id. The readout IF is set to
            `readout_frequency - qa_lo_frequency`.
        set_local_oscillator: Whether to set the local oscillators in the calibration.
        measure_pulse: Pulse used in readout signals.
        acquire_kernel: Integration kernel used in acquisition.
        acquisition_type: Acquisition mode used by the real-time acquire loop.
        averaging_mode: Averaging mode used by the real-time acquire loop.
        sg_modulation_type: Oscillator modulation type applied to the drive (SG) IF
            oscillator.
        qa_modulation_type: Oscillator modulation type applied to the readout (QA) IF
            oscillator.
        sg_range: Range to set in the SignalCalibration of the drive (SG) channel.
            If set to None, uses the default settings.
        chunk_count: Number of chunks to split the sequence sweep into. Chunking
            carries more sequences than fit into a single program, but not every setup
            can execute the chunks: a UHFQA cannot, nor can an SHFQA acquiring in
            `AcquisitionType.RAW`. Each chunk is averaged on its own.
        auto_chunking: Whether the compiler picks the chunk count itself, taking
            `chunk_count` as its starting guess.
    """

    seed: int = attrs.field(validator=attrs.validators.ge(0))
    qubit_ids: list[str] = attrs.field(validator=attrs.validators.min_len(1))
    sequence_lengths: list[int] = attrs.field(validator=attrs.validators.min_len(1))
    n_average: int
    n_repeat: int = attrs.field(validator=attrs.validators.ge(1))
    relaxation_length: float = attrs.field(validator=attrs.validators.gt(0))
    excitation_length: float
    pi_pulse_amp: float
    pi_half_pulse_amp: float
    prereadout_delay: float = attrs.field(
        validator=attrs.validators.ge(0)
    )  # !FIXME_CLARIFY_UC
    excitation_frequencies: dict[str, float]
    readout_frequencies: dict[str, float]
    sg_lo_frequencies: dict[str, float]
    qa_lo_frequencies: dict[str, float]
    measure_pulse: pl.PulseFunctional
    acquire_kernel: pl.PulseFunctional
    acquisition_type: AcquisitionType = attrs.field()
    averaging_mode: AveragingMode = attrs.field()
    sg_modulation_type: ModulationType
    qa_modulation_type: ModulationType
    sg_range: float | None
    chunk_count: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    auto_chunking: bool = False
    set_local_oscillator: bool = True


@attrs.define
class RBExperimentBuilder:
    """A class for constructing single-qubit randomized benchmarking experiments.

    The RB experiment constructed by instances of this class can be understood in terms
    of the following pseudo-code:
    ```
    n_iter = len(sequence_lengths)
    n_qubits = len(qubits)

    # Prepare random clifford gates
    clifford_sequences = {}
    for m in range(n_iter):
        for k in range(n_repeat):
            for qubit in qubits:
                clifford_sequences[(m, k, qubit)] = pick_randomly(
                    clifford_set_for_1_qubit, sequence_lengths[m]
                )

    _results = array(dimensions=(n_average, n_iter, n_repeat, n_qubits))
    for n in range(n_average):
        for m, seq_length in enumerate(sequence_lengths):
            for k in range(n_repeat):
                for i, qubit in enumerate(qubits):
                    clifford_sequence = clifford_sequences[(m, k, qubit)]
                    apply(clifford_sequence, to=qubit)
                    _results[n, m, k, i] = measure(qubit)
    results = average(_results, over_dimension=0)
    ```
    The nesting shown is the one of `AveragingMode.CYCLIC`. Under
    `AveragingMode.SEQUENTIAL` the averaging loop is the innermost loop instead of the
    outermost one, so each sequence is averaged before the next one is played.
    The real execution is much more complex than what this snippet suggests, but this is
    still useful for understanding which parts are randomized and how results are
    obtained.

    """

    __experiment_name__: ClassVar[str] = "RB"
    __required_signal_types__: ClassVar[frozenset[str]] = frozenset(
        {"drive", "measure", "acquire"}
    )

    def build(self, settings: RBExperimentSettings) -> Experiment:
        signals = [
            f"{q}_{signal_type}"
            for q in settings.qubit_ids
            for signal_type in self.__required_signal_types__
        ]
        exp = Experiment(
            uid=self.__experiment_name__, name=self.__experiment_name__, signals=signals
        )
        rng = nprnd.default_rng(settings.seed)
        gate_map = PauliGateMap.make(
            settings.excitation_length,
            settings.pi_pulse_amp,
            settings.pi_half_pulse_amp,
        )

        # All RB sequences, indexed as (sequence_length, repetition) pairs. The RNG is
        # drawn in this order below, so the order is part of what a seed pins down.
        sequences = [
            (sequence_length, repetition)
            for sequence_length in settings.sequence_lengths
            for repetition in range(settings.n_repeat)
        ]
        index = SweepParameter(uid="rb_index", values=np.arange(len(sequences)))

        with exp.acquire_loop_rt(
            uid="rt_loop",
            count=settings.n_average,
            averaging_mode=settings.averaging_mode,
            acquisition_type=settings.acquisition_type,
            reset_oscillator_phase=False,
        ):
            with exp.sweep(
                uid="rb_sweep",
                parameter=index,
                chunk_count=settings.chunk_count,
                auto_chunking=settings.auto_chunking,
            ):
                # The composite pulse of a sequence of Cliffords, one case per sequence.
                # A match on a sweep parameter is resolved at compile time, so a chunk
                # holds only the cases it plays.
                with exp.match(sweep_parameter=index):
                    for i, (sequence_length, _repetition) in enumerate(sequences):
                        with exp.case(i):
                            with exp.section(
                                uid=f"cliffords_{i}",
                                alignment=SectionAlignment.RIGHT,
                            ):
                                for qubit_id in settings.qubit_ids:
                                    generate_play_rb_pulses(
                                        exp,
                                        f"{qubit_id}_drive",
                                        sequence_length,
                                        clifford_parametrized,
                                        gate_map=gate_map,
                                        rng=rng,
                                    )
                # Readout and data acquisition at the end of every sequence
                with exp.section(uid="readout"):
                    for qubit_id in settings.qubit_ids:
                        exp.reserve(
                            signal=f"{qubit_id}_drive",
                        )
                        # add a delay before the readout pulse
                        exp.delay(
                            signal=f"{qubit_id}_measure",
                            time=settings.prereadout_delay,
                        )
                        # play readout pulse
                        exp.play(
                            signal=f"{qubit_id}_measure",
                            pulse=settings.measure_pulse,
                        )
                        # signal data acquisition
                        exp.acquire(
                            signal=f"{qubit_id}_acquire",
                            handle=f"ac_{qubit_id}",
                            kernel=settings.acquire_kernel,
                        )
                with exp.section(
                    uid="relaxation",
                    length=settings.relaxation_length,
                ):
                    for qubit_id in settings.qubit_ids:
                        exp.reserve(
                            signal=f"{qubit_id}_drive",
                        )
                        exp.reserve(
                            signal=f"{qubit_id}_measure",
                        )
                        exp.reserve(
                            f"{qubit_id}_acquire",
                        )

        exp.set_calibration(
            self._make_calibration(
                settings.qubit_ids,
                settings.excitation_frequencies,
                settings.sg_lo_frequencies,
                settings.readout_frequencies,
                settings.qa_lo_frequencies,
                settings.sg_range,
                settings.sg_modulation_type,
                settings.qa_modulation_type,
                settings.set_local_oscillator,
            )
        )
        return exp

    def _make_calibration(
        self,
        qubit_ids: list[str],
        drive_frequencies: dict[str, float],
        sg_lo_frequencies: dict[str, float],
        readout_frequencies: dict[str, float],
        qa_lo_frequencies: dict[str, float],
        drive_range: float | None,
        sg_modulation_type: ModulationType,
        qa_modulation_type: ModulationType,
        set_local_oscillator: bool,
    ):
        exp_cal = Calibration()

        for qubit_id in qubit_ids:
            f_drive, flo_sg, f_ro, flo_qa = (
                drive_frequencies[qubit_id],
                sg_lo_frequencies[qubit_id],
                readout_frequencies[qubit_id],
                qa_lo_frequencies[qubit_id],
            )

            exp_cal[f"{qubit_id}_drive"] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=f_drive - flo_sg, modulation_type=sg_modulation_type
                ),
                local_oscillator=(
                    Oscillator(frequency=flo_sg) if set_local_oscillator else None
                ),
                range=drive_range,
            )

            exp_cal[f"{qubit_id}_measure"] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=f_ro - flo_qa, modulation_type=qa_modulation_type
                ),
                local_oscillator=(
                    Oscillator(frequency=flo_qa) if set_local_oscillator else None
                ),
            )
        return exp_cal
