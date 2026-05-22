# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import attrs
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
        prereadout_delay: Delay to apply before readout pulse.
        measure_pulse: Pulse used in readout signals.
        acquire_kernel: Integration kernel used in acquisition.

    Properties:
        excitation_length: Duration of gate pulses. Automatically calculated from gate map.
        cliffords_section_length: Section length where the Clifford gates are applied.

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
    modulation_type: ModulationType

    cliffords_section_length: float = attrs.field(init=False)

    @cliffords_section_length.default  # type: ignore
    def _calc_max_sequence_duration(self):
        """Calculate the duration of the maximum possible sequence assuming the longest
        Clifford gate is repeated `n_repeat` times by `max(sequence_lengths)`."""
        max_clifford_count = max(self.sequence_lengths) + 1  # +1 for recovery
        return max_clifford_count * MAX_PAULI_GATE_PER_CLIFFORD * self.excitation_length


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

        # Construct a RB experiment with identical structure to the example notebook
        with exp.acquire_loop_rt(
            uid="rt_loop",
            count=settings.n_average,
            averaging_mode=settings.averaging_mode,
            acquisition_type=settings.acquisition_type,
            reset_oscillator_phase=False,
        ):
            # inner loop - sweep over sequence lengths
            for length in settings.sequence_lengths:
                # innermost loop - different random sequences for each length
                for k in range(settings.n_repeat):
                    # RB block

                    # generate composite pulse representing the sequence of Cliffords
                    with exp.section(
                        uid=f"cliffords_{length}_{k}",
                        length=settings.cliffords_section_length,
                        alignment=SectionAlignment.RIGHT,
                    ):
                        for qubit_id in settings.qubit_ids:
                            generate_play_rb_pulses(
                                exp,
                                f"{qubit_id}_drive",
                                length,
                                clifford_parametrized,
                                gate_map=gate_map,
                                rng=rng,
                            )
                    # readout and data acquisition
                    with exp.section(uid=f"readout_{length}_{k}"):
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
                        uid=f"relaxation_{length}_{k}",
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
                settings.modulation_type,
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
        modulation_type: ModulationType,
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
                    frequency=f_drive - flo_sg, modulation_type=modulation_type
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
