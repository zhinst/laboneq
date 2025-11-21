# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal
from functools import partial
from typing import ClassVar

import attrs
import numpy as np
from numpy import random as nprnd

from laboneq.simple import (
    Calibration,
    Experiment,
    Oscillator,
    SectionAlignment,
    Session,
    SignalCalibration,
    SweepParameter,
)
from laboneq.simple import pulse_library as pl


def create_clops(settings: CLOPSExperimentSettings, session: Session) -> Experiment:
    """Create a Circuit Layer Operations Per Second (CLOPS) experiment.

    The returned experiment will have signals of the form:

    * {qubit_id}_drive
    * {qubit_id}_measure
    * {qubit_id}_acquire
    * {qubit_id}_flux

    and acquisition handles of the form:

    * ac_{qubit_id}

    per qubit id provided in the `settings` object.

    Notes:

    1. The permutation layer is implemented as an empty section.

    2. The SU(4) gates acting on the qubit pairs are represented with
       2 drive pulses followed by 2 flux pulses and 2 drive pulses again. Each
       of the 2 pulse sections have one pulse for one qubit and another for
       the second qubit.

    3. Circuit update iterations are implemented in near-time.

    4. All drive and flux pulses are gaussian with a sigma in seconds given as
       1/6th of the excitation length.

    5. All measure pulses and acquire kernels are rectangular.
    """

    builder = CLOPSExperimentBuilder()
    return builder.build(settings=settings, session=session)


def calculate_clops(
    n_average: int, n_template: int, n_repeat: int, depth: int, total_runtime: float
) -> float:
    """Calculate CLOPS.

    Arguments:
        n_average: Number of averages the experiment had.
        n_template: Number of circuit templates (i.e. M)
        n_repeat: Number of randomized template updates (i.e. K)
        depth: Depth of the QV circuit.
        total_runtime: Total execution time, excluding initial compilation and configuration step.
    """
    n_pattern = n_template * n_repeat
    return n_pattern * n_average * depth / total_runtime


@attrs.frozen(kw_only=True)
class CLOPSExperimentSettings:
    """Settings for a CLOPS experiment.

    Attributes:
        seed: A non-negative integer for seeding the PRNG.
        qubit_ids: Labels for the qubits, e.g. ["q0", "q1", ...]. CLOPS protocol is
            defined for a minimum of two qubits.
        n_average: Number of repetitions for each template experiment.  Only the final
            averaged results are kept.
        n_template: Number of circuit templates.
        n_repeat: Number of repetitions per template. Each circuit template is updated a
            total of `n_repeat` times. The permutation layers in each QV layer is fixed
            during these parameter updates.
        qv_layer_length: Length of the QV layer section in seconds.
        relaxation_length: Relaxation time between iterations.
        readout_length: Length of the measurement pulse and the acquisition kernel.
        excitation_frequencies: Frequency of drive signals. These are the absolute
            frequencies of the signal that goes out the wire.
        readout_frequencies: Frequency of readout signals. These are the absolute
            frequencies of the signal that goes out the wire.
        sg_lo_frequencies: Drive signal (non-local) oscillator frequencies in
            LabOne Q parlance. This determines the center frequency for the
            up/downconversion on the hardware.
        qa_lo_frequencies: Readout signal (non-local) oscillator frequency in
            LabOne Q parlance. This determines the center frequency for the
            up/downconversion on the hardware.

    Properties:
        depth: Circuit depth of a single Quantum Volume (QV) experiment. Always equal to
            `n_qubit` as CLOPS is defined for square (number of QV layers = number of
            qubits) circuits.

    """

    seed: int = attrs.field()
    qubit_ids: list[str] = attrs.field(validator=attrs.validators.min_len(2))
    n_average: int
    n_template: int = attrs.field(validator=attrs.validators.ge(1))
    n_repeat: int = attrs.field(validator=attrs.validators.ge(1))
    qv_layer_length: float = attrs.field(validator=attrs.validators.ge(0))
    relaxation_length: float = attrs.field(validator=attrs.validators.gt(0))
    readout_length: float = attrs.field(validator=attrs.validators.gt(0))
    excitation_length: float = attrs.field(validator=attrs.validators.gt(0))
    excitation_frequencies: dict[str, float] = attrs.field()
    readout_frequencies: dict[str, float] = attrs.field()
    sg_lo_frequencies: dict[str, float] = attrs.field()
    qa_lo_frequencies: dict[str, float] = attrs.field()

    # Auto-initialized to ensure square circuits, can be changed if needed in future.
    depth: int = attrs.field(init=False)

    @depth.default  # type: ignore
    def _set_depth(self):
        return len(self.qubit_ids)

    @excitation_frequencies.validator  # type: ignore
    @readout_frequencies.validator  # type: ignore
    @sg_lo_frequencies.validator  # type: ignore
    @qa_lo_frequencies.validator  # type: ignore
    def _has_qubit_ids(self, attribute: attrs.Attribute, value: dict):
        if set(value.keys()) != set(self.qubit_ids):
            raise ValueError(f"'{attribute.name}' keys must be equal to `qubit_ids`. ")


@attrs.define
class CLOPSExperimentBuilder:
    """Experiment builder for the CLOPS experiment.

    Currently, the permutation layer is left empty as it is strictly application
    specific. The SU(4) gate section is implemented with an ad-hoc set of pulses on
    qubit drive and flux lines and the circuit update iterations are implemented in
    near-time.

    Recommended way to instantiate this object is through `.from_settings` method.

    Currently the SU(4) gates acting on the qubit pairs are represented with 2 drive
    pulses followed by 2 flux pulses and 2 drive pulses again. Each of the 2 pulse
    sections have one pulse for one qubit and another for the second qubit.
    """

    __experiment_name__: ClassVar[str] = "CLOPS"
    __required_signal_types__: ClassVar[frozenset[str]] = frozenset(
        {"drive", "measure", "acquire", "flux"}
    )

    def build(
        self,
        *,
        settings: CLOPSExperimentSettings,
        session: Session,
    ) -> Experiment:
        signals = [
            f"{q}_{signal_type}"
            for q in settings.qubit_ids
            for signal_type in self.__required_signal_types__
        ]
        exp = Experiment(
            uid=self.__experiment_name__, name=self.__experiment_name__, signals=signals
        )
        self._populate_clops_experiment(exp, settings)

        rng = nprnd.default_rng(settings.seed)
        self._prepare_session(
            session,
            settings.qubit_ids,
            settings.depth,
            settings.excitation_length,
            rng,
        )

        self._calibrate_experiment(
            exp,
            settings.qubit_ids,
            settings.excitation_frequencies,
            settings.sg_lo_frequencies,
            settings.readout_frequencies,
            settings.qa_lo_frequencies,
        )
        return exp

    def _calibrate_experiment(
        self,
        exp: Experiment,
        qubit_ids: list[str],
        drive_frequencies: dict[str, float],
        sg_lo_frequencies: dict[str, float],
        readout_frequencies: dict[str, float],
        qa_lo_frequencies: dict[str, float],
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
                oscillator=Oscillator(frequency=f_drive - flo_sg),
                local_oscillator=Oscillator(frequency=flo_sg),
            )

            exp_cal[f"{qubit_id}_measure"] = SignalCalibration(
                oscillator=Oscillator(frequency=f_ro - flo_qa),
                local_oscillator=Oscillator(frequency=flo_qa),
            )
        exp.set_calibration(exp_cal)

    def _calculate_readout_entropy(
        self, readout_results: dict[str, float | np.float64]
    ) -> int:
        """Calculates a representative entropy useful to calculate the seed function for
        the next gate randomization update. Called exactly once per near-time iteration,
        with the results exclusively from the latest measurement operation over all
        qubits.

        Arguments:
            readout_results: Dictionary mapping qubit ids to the readout result.

        Returns:
            entropy: Sum of all significant digits in all qubit readout measurement results.
        """
        # TODO: Currently a placeholder calculation, should be user-defined.

        # decimal significant digits in a complex number
        sum_digits = lambda f: sum(Decimal.from_float(f).as_tuple().digits)
        sum_digits_complex = lambda c: sum_digits(c.real) + sum_digits(c.imag)
        entropy = sum(
            [
                sum_digits_complex(result)  # 0 if result == NaN
                for result in readout_results.values()
            ]
        )
        return entropy

    def _populate_clops_experiment(
        self, exp: Experiment, settings: CLOPSExperimentSettings
    ):
        measure_pulse = pl.const(length=settings.readout_length)
        acquire_kernel = pl.const(length=settings.readout_length)
        template_id_handle = SweepParameter(
            uid="template_id", values=range(settings.n_template)
        )
        repetition_id_handle = SweepParameter(
            uid="repetition_id", values=range(settings.n_repeat)
        )
        n_average = settings.n_average
        depth = settings.depth
        with exp.sweep(uid="template_loop", parameter=template_id_handle):
            exp.call(
                "update_circuit_template",
                template_id=template_id_handle,
            )
            with exp.sweep(uid="su4_update_loop", parameter=repetition_id_handle):
                exp.call(  # relies on last measurement results
                    "update_circuit_parameters",
                    template_id=template_id_handle,
                    repetition_id=repetition_id_handle,
                )
                with exp.acquire_loop_rt(uid="rt_loop", count=n_average):
                    with exp.section(uid="qv_circuit_gates"):
                        for layer_index in range(depth):
                            self._add_qv_layer(
                                exp,
                                layer_index,
                                settings.qubit_ids,
                                settings.qv_layer_length,
                                excitation_length=settings.excitation_length,
                            )

                    with exp.section(uid="readout", play_after="qv_circuit_gates"):
                        for qubit_id in settings.qubit_ids:
                            exp.play(
                                signal=f"{qubit_id}_measure",
                                pulse=measure_pulse,
                            )
                            exp.acquire(
                                signal=f"{qubit_id}_acquire",
                                handle=f"ac_{qubit_id}",
                                kernel=acquire_kernel,
                            )

                    with exp.section(uid="relaxation"):
                        for qubit_id in settings.qubit_ids:
                            exp.delay(
                                signal=f"{qubit_id}_measure",
                                time=settings.relaxation_length,
                            )

    def _add_qv_layer(
        self,
        exp: Experiment,
        layer_index: int,
        qubit_ids: list[str],
        qv_length: float,
        excitation_length: float,
    ):
        permutation_layer_uid = f"permutation_layer_{layer_index}"
        su4_layer_uid = f"su4_layer_{layer_index}"
        with exp.section(
            uid=f"qv_layer_{layer_index}",
            length=qv_length,
            alignment=SectionAlignment.RIGHT,
        ):
            with exp.section(
                uid=permutation_layer_uid, alignment=SectionAlignment.RIGHT
            ):
                self._add_permutation_layer(exp)
            with exp.section(
                uid=su4_layer_uid,
                play_after=permutation_layer_uid,
                alignment=SectionAlignment.RIGHT,
            ):
                # SU(4) layer has depth//2 SU(4) gates
                for q_u, q_d in zip(*[iter(qubit_ids)] * 2):
                    # drops last for odd number of qubits (last SU4 sits idle in each)
                    self._add_su4_gate(exp, layer_index, q_u, q_d, excitation_length)

    def _add_permutation_layer(self, exp: Experiment):
        # Permutation layer implementation is specific to the qubit architecture
        # TODO: Find a way to allow users to define this layer.
        pass

    def _update_circuit_template(
        self, rng: nprnd.Generator, session: Session, template_id: int
    ) -> int:
        return 0

    def _update_circuit_parameters(
        self,
        rng: nprnd.Generator,
        qubit_ids: list[str],
        circuit_depth: int,
        excitation_length: float,
        session: Session,
        template_id: int,
        repetition_id: int,
    ) -> int:
        """Used to randomize circuit parameters at the end of each near-time iteration.

        Returns:
            entropy: The seed used for the PRNG when randomizing gates.
        """
        results = session.results.acquired_results

        latest_results = {
            qubit_id: results[f"ac_{qubit_id}"].data[tuple(last_nt_step)]
            for qubit_id in qubit_ids
            if (last_nt_step := session.results.get_last_nt_step(f"ac_{qubit_id}"))
        }

        result_entropy = self._calculate_readout_entropy(latest_results)
        # this will be simply 0 at first call when there is no result available.

        # mix the result entropy with the provided rng state
        rng = np.random.Generator(rng.bit_generator.jumped(result_entropy))

        # entropy can be directly used as seed to reproduce the state of the PRNG.
        # So we can still reproduce even though experiment may not be deterministic
        entropy = rng.bit_generator.seed_seq.entropy

        self._randomize_su4_gate_pulses(
            session, rng, qubit_ids, circuit_depth, excitation_length
        )
        return entropy  # can be accessed later from session.results.user_func_results

    def _add_su4_gate(
        self,
        exp: Experiment,
        layer_index: int,
        q_u: str,
        q_d: str,
        length: float,
    ):
        # All SU(4) gates have the same decomposition.
        # TODO: Allow users to define the decomposition.

        pulse_factory = pl.gaussian
        q_u_drive = f"{q_u}_drive"
        q_d_drive = f"{q_d}_drive"
        q_u_flux = f"{q_u}_flux"
        q_d_flux = f"{q_d}_flux"

        with exp.section(alignment=SectionAlignment.RIGHT):
            exp.play(
                signal=q_u_drive,
                pulse=pulse_factory(f"{q_u}_d{layer_index}_L", length=length),
            )
            exp.play(
                signal=q_d_drive,
                pulse=pulse_factory(f"{q_d}_d{layer_index}_L", length=length),
            )
        with exp.section(alignment=SectionAlignment.RIGHT):
            exp.reserve(q_u_drive)
            exp.reserve(q_d_drive)
            exp.play(
                signal=q_u_flux,
                pulse=pulse_factory(f"{q_u}_d{layer_index}_F", length=length),
            )
            exp.play(
                signal=q_d_flux,
                pulse=pulse_factory(f"{q_d}_d{layer_index}_F", length=length),
            )
        with exp.section(alignment=SectionAlignment.RIGHT):
            exp.reserve(q_u_flux)
            exp.reserve(q_d_flux)
            exp.play(
                signal=q_u_drive,
                pulse=pulse_factory(f"{q_u}_d{layer_index}_R", length=length),
            )
            exp.play(
                signal=q_d_drive,
                pulse=pulse_factory(f"{q_d}_d{layer_index}_R", length=length),
            )

    def _randomize_su4_gate_pulses(
        self,
        session: Session,
        rng: nprnd.Generator,
        qubit_ids: list[str],
        circuit_depth: int,
        excitation_length: float,
    ):
        pulse_factory = pl.gaussian
        discriminators = "LFR"  # adhoc pulse discrimination suffixes we used
        n_discriminators = len(discriminators)
        n_qubit = len(qubit_ids)

        amplitudes = rng.random(size=(circuit_depth, n_qubit, n_discriminators))
        phasors = np.exp(
            1j * 2 * np.pi * rng.random(size=(circuit_depth, n_qubit, n_discriminators))
        )

        for layer_index in range(circuit_depth):
            for qubit_index in range(n_qubit // 2):
                qubit_id = qubit_ids[qubit_index]
                for disc_index, discriminator in enumerate(discriminators):
                    pulse_uid = f"{qubit_id}_d{layer_index}_{discriminator}"
                    amplitude = amplitudes[layer_index, qubit_index, disc_index]
                    phasor = phasors[layer_index, qubit_index, disc_index]
                    session.replace_pulse(
                        pulse_uid,
                        pulse_factory(
                            uid=pulse_uid,
                            length=excitation_length,
                            amplitude=amplitude * phasor,
                        ),
                    )

    def _prepare_session(
        self,
        session: Session,
        qubit_ids: list[str],
        circuit_depth: int,
        excitation_length: float,
        rng: np.random.Generator,
    ) -> Session:
        """Prepares the session object for the near-time operations necessary for the
        CLOPS experiment.

        Arguments:
            session: Session object to prepare.
            experiment: Experiment built with this builder to prepare the session object
                against.

        Returns:
            session: Session object with the necessary near-time callback functions
                registered.

        """

        # Separating the PRNGs for circuit template updates and the gate randomization
        # simplifies tracking the random numbers deterministically by decoupling the
        # randomization for these two different processes.
        rng_templates, rng_gates = rng.spawn(2)
        update_circuit_template = partial(
            CLOPSExperimentBuilder._update_circuit_template,
            self,
            rng_templates,
        )
        update_circuit_parameters = partial(
            CLOPSExperimentBuilder._update_circuit_parameters,
            self,
            rng_gates,
            qubit_ids,
            circuit_depth,
            excitation_length,
        )
        session.register_neartime_callback(
            update_circuit_template, "update_circuit_template"
        )
        session.register_neartime_callback(
            update_circuit_parameters, "update_circuit_parameters"
        )

        return session
