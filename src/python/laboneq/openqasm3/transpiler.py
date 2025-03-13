# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable
import logging
from contextlib import nullcontext
from itertools import chain

from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.dsl import enums, parameter, quantum
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.experiment import Experiment, Section
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.openqasm3 import openqasm3_importer
from laboneq.openqasm3.options import MultiProgramOptions, SingleProgramOptions
from laboneq.openqasm3 import device

_logger = logging.getLogger(__name__)


def _calibration_from_qubits(
    qubits: list[QuantumElement],
) -> dict[str,]:
    """Return the calibration objects from a list of qubits."""
    calibration = {}
    for qubit in qubits:
        if isinstance(qubit, QuantumElement):
            calibration.update(qubit.calibration())
        else:
            # handle lists or tuples of qubits:
            for q in qubit:
                calibration.update(q.calibration())
    return calibration


def _copy_set_frequency_calibration(implicit_calibration: Calibration, exp: Experiment):
    """Copy set_frequency values from the implicit visitor calibration to the experiment."""
    # TODO: This function should be removed and instead the experiment calibration should be
    #       accessible in the visitor so that it can be modified directly if needed.
    #       Possibly we could do this using a `set_frequency` quantum operation.
    exp_calibration = exp.get_calibration()
    for signal, signal_calibration in implicit_calibration.items():
        exp_signal_calibration = exp_calibration.get(signal)
        if exp_signal_calibration is None or exp_signal_calibration.oscillator is None:
            raise ValueError(
                f"Sweeping or setting the frequency of signal {signal!r}"
                f" requires a signal calibration with oscillator to be set."
            )
        assert signal_calibration.oscillator is not None
        exp_signal_calibration.oscillator.frequency = (
            signal_calibration.oscillator.frequency
        )


class OpenQASMTranspiler:
    """OpenQASM transpiler for LabOne Q.

    Translates OpenQASM program(s) to LabOne Q constructs.
    Does not perform any optimization steps.

    Arguments:
        qpu: Quantum processing unit for which the programs are
            transpiled to.

            Qubits and quantum operations used within the programs
            must exist in the `qpu`.
    """

    def __init__(self, qpu: quantum.QPU):
        self.qpu = qpu
        self._qubit_signals: set[str] = set(
            chain.from_iterable([q.signals.values() for q in self.qpu.qubits])
        )

    def section(
        self,
        program: str,
        qubit_map: dict[str, quantum.QuantumElement | list[quantum.QuantumElement]],
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable | device.Port] | None = None,
    ) -> Section:
        """Transpile a program into an experiment section.

        Useful when only the OpenQASM program part is needed as an `Section`.

        To create a full `Experiment`, use either `.experiment()` or
        `.batch_experiment()`.

        Arguments:
            program:
                OpenQASM program.
            qubit_map:
                A map from OpenQASM qubit names to LabOne Q DSL Qubit objects in the `QPU`.

                The qubit values can be either be qubit `uid`s or direct qubit objects.
                Mapped qubit registers must be a list of qubits.

                When `QuantumElement` is supplied, it is used directly as long as the
                matching `uid` exists in `QPU`.
            inputs:
                Inputs to the program.

                Supports also OpenPulse implicit input waveforms.

                Input waveforms declared in the legacy API `exp_from_qasm(waveforms=...)`
                are also treated as inputs.
            externs:
                A mapping for program extern definitions.

                Externs may be either functions (Python `callable`s) or
                ports on qubit signals.

        Returns:
            a `Section` which contains the operations in the program.

        Raises:
            ValueError: Supplied qubit(s) or mapped ports does not exists in the QPU.
            OpenQasmException: The program cannot be transpiled.
            TypeError: Extern value is not of type `callable` or `Port`.
        """
        qubit_map = self._create_qubit_register(qubit_map)
        externs = self._preprocess_externs(externs or {})
        importer = openqasm3_importer.OpenQasm3Importer(
            qops=self.qpu.quantum_operations,
            qubits=openqasm3_importer.define_qubit_names(qubit_map),
            inputs=inputs,
            externs=externs,
        )
        return importer(text=program).section

    def experiment(
        self,
        program: str,
        qubit_map: dict[
            str, str | list[str] | quantum.QuantumElement | list[quantum.QuantumElement]
        ],
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable | device.Port] | None = None,
        options: SingleProgramOptions | dict | None = None,
    ) -> Experiment:
        """Transpile QASM program into an LabOne Q experiment.

        Arguments:
            program:
                OpenQASM program.
            qubit_map:
                A map from OpenQASM qubit names to LabOne Q DSL Qubit objects in the `QPU`.

                The qubit values can be either be qubit `uid`s or direct qubit objects.
                Mapped qubit registers must be a list of qubits.

                When `QuantumElement` is supplied, it is used directly as long as the
                matching `uid` exists in `QPU`.
            inputs:
                Inputs to the program.

                Supports also OpenPulse implicit input waveforms.

                Input waveforms declared in the legacy API `exp_from_qasm(waveforms=...)`
                are also treated as inputs.
            externs:
                A mapping for program extern definitions.

                Externs may be either functions (Python `callable`s) or
                ports on qubit signals.
            options:
                Optional settings for the LabOne Q Experiment.

                Default: [SingleProgramOptions]()

                Accepts also a dictionary with following items:

                **count**:
                    The number of acquire iterations.

                **acquisition_mode**:
                    The mode of how to average the acquired data.

                **acquisition_type**:
                    The type of acquisition to perform.
                    The acquisition type may also be specified within the
                    OpenQASM program using `pragma zi.acquisition_type raw`,
                    for example.
                    If an acquisition type is passed here, it overrides
                    any value set by a pragma.
                    If the acquisition type is not specified, it defaults
                    to [enums.AcquisitionType.INTEGRATION]().

                **reset_oscillator_phase**:
                    When true, reset all oscillators at the start of every
                    acquisition loop iteration.

        Returns:
            A LabOne Q Experiment.

        Raises:
            ValueError: Supplied qubit(s) or mapped ports does not exists in the QPU.
            OpenQasmException: The program cannot be transpiled.
            TypeError: Extern value is not of type `callable` or `Port`.
        """
        qubit_map = self._create_qubit_register(qubit_map)
        if isinstance(options, dict):
            options = SingleProgramOptions(**options)
        else:
            options = options or SingleProgramOptions()
        # TODO: Replace with proper qubit register / mapping
        importer = openqasm3_importer.OpenQasm3Importer(
            qops=self.qpu.quantum_operations,
            qubits=qubit_map,
            inputs=inputs,
            externs=self._preprocess_externs(externs or {}),
        )
        ret = importer(text=program)
        qasm_section = ret.section
        importer_acquire_loop_options = ret.acquire_loop_options
        acquisition_type = options.acquisition_type
        if "acquisition_type" in importer_acquire_loop_options:
            importer_acquisition_type = importer_acquire_loop_options[
                "acquisition_type"
            ]
            if acquisition_type is None:
                acquisition_type = importer_acquisition_type
            else:
                _logger.warning(
                    f"Overriding the acquisition type supplied via a pragma "
                    f"({importer_acquisition_type}) with: {acquisition_type}"
                )
        if acquisition_type is None:
            acquisition_type = enums.AcquisitionType.INTEGRATION

        # TODO: feed qubits directly to experiment when feature is implemented
        exp = Experiment(signals=_experiment_signals(qubit_map))

        calibration = Calibration(_calibration_from_qubits(qubit_map.values()))
        exp.set_calibration(calibration)

        with exp.acquire_loop_rt(
            count=options.count,
            averaging_mode=options.averaging_mode,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=options.reset_oscillator_phase,
        ) as loop:
            loop.add(qasm_section)

        _copy_set_frequency_calibration(ret.implicit_calibration, exp)
        return exp

    def batch_experiment(
        self,
        programs: list[str],
        qubit_map: dict[
            str, str | list[str], quantum.QuantumElement | list[quantum.QuantumElement]
        ],
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable | device.Port] | None = None,
        options: MultiProgramOptions | dict | None = None,
    ) -> Experiment:
        """Batch a list of OpenQASM programs into a LabOne Q experiment.

        The list of programs are processed into a single LabOne Q experiment that
        executes them sequentially.

        At this time, the QASM programs should not include any measurements. By default, we automatically
        append a measurement of all qubits to the end of each program. This can be controlled
        via `add_measurement` option.

        Note: Using `set_frequency` or specifying the acquisition type via a
        `pragma zi.acquisition_type` statement within an OpenQASM program is not
        supported in batch mode. It will log a warning if these are encountered.

        The supplied mapping parameters for OpenQASM definitions are shared among the programs.
        Individual parameter mapping per program cannot be done.

        Arguments:
            programs:
                List of OpenQASM program.
            qubit_map:
                A map from OpenQASM qubit names to LabOne Q DSL Qubit objects in the `QPU`.

                The qubit values can be either be qubit `uid`s or direct qubit objects.
                Mapped qubit registers must be a list of qubits.

                When `QuantumElement` is supplied, it is used directly as long as the
                matching `uid` exists in `QPU`.
            inputs:
                Inputs to the program.

                Supports also OpenPulse implicit input waveforms.

                Input waveforms declared in the legacy API `exp_from_qasm(waveforms=...)`
                are also treated as inputs.
            externs:
                A mapping for program extern definitions.

                Externs may be either functions (Python `callable`s) or
                ports on qubit signals.
            options:
                Optional settings for the LabOne Q Experiment.

                Default: [MultiProgramOptions]()

                Accepts also a dictionary with the following items:

                **count**:
                    The number of acquire iterations.

                **acquisition_mode**:
                    The mode of how to average the acquired data.

                **acquisition_type**:
                    The type of acquisition to perform.
                    The acquisition type may also be specified within the
                    OpenQASM program using `pragma zi.acquisition_type raw`,
                    for example.
                    If an acquisition type is passed here, it overrides
                    any value set by a pragma.
                    If the acquisition type is not specified, it defaults
                    to [enums.AcquisitionType.INTEGRATION]().

                **reset_oscillator_phase**:
                    When true, reset all oscillators at the start of every
                    acquisition loop iteration.

                **repetition_time**:
                    The length that any single program is padded to.
                    The duration between the reset and the final readout is fixed and must be specified as
                    `repetition_time`. It must be chosen large enough to accommodate the longest of the
                    programs. The `repetition_time` parameter is also required if the resets are
                    disabled. In a future version we hope to make an explicit `repetition_time` optional.

                **batch_execution_mode**:
                    The execution mode for the sequence of programs. Can be any of the following.

                * `nt`: The individual programs are dispatched by software.
                * `pipeline`: The individual programs are dispatched by the sequence pipeliner.
                * `rt`: All the programs are combined into a single real-time program.

                `rt` offers the fastest execution, but is limited by device memory.
                In comparison, `pipeline` introduces non-deterministic delays between
                programs of up to a few 100 microseconds. `nt` is the slowest.

                **add_reset**:
                    If `True`, an active reset operation is added to the beginning of each program.

                Note: Requires `reset(qubit)` operation to be defined for each qubit.

                **add_measurement**:
                    If `True`, add measurement at the end of each program for all qubits used.

                Note: Requires `measure(qubit, handle: str)` operation to be defined for each qubit, where `handle`
                is the key specified for the qubit in the `qubit_map` parameter (e.g. `q0`) or
                `<key>[N]` in the case of an qubit register as a list of qubits (e.g. `q[0]`, `q[1]`, ..., `q[N]`,
                where `N` represents the qubit index in the supplied register).

                **pipeline_chunk_count**:
                    The number of pipeline chunks to divide the experiment into.

        Returns:
            A LabOne Q Experiment.

        Raises:
            ValueError: Supplied qubit(s) or mapped ports does not exists in the QPU.
            OpenQasmException: The program cannot be transpiled.
            TypeError: Extern value is not of type `callable` or `Port`.
        """
        qubit_map = self._create_qubit_register(qubit_map)
        if isinstance(options, dict):
            options = MultiProgramOptions(**options)
        else:
            options = options or MultiProgramOptions()

        batch_execution_mode = options.batch_execution_mode
        pipeline_chunk_count = options.pipeline_chunk_count
        importer = openqasm3_importer.OpenQasm3Importer(
            qops=self.qpu.quantum_operations,
            qubits=qubit_map,
            inputs=inputs,
            externs=self._preprocess_externs(externs or {}),
        )

        if batch_execution_mode == "pipeline":
            if pipeline_chunk_count is None:
                pipeline_chunk_count = len(programs)
            if len(programs) % pipeline_chunk_count != 0:
                # The underlying limitation is that the structure of the acquisitions
                # must be the same in each chunk, because the compiled experiment
                # recipe only supplies the acquisition information once, rather than
                # once per chunk. Once the acquisition information has been moved to
                # per-chunk execution information and the controller updated to apply
                # this, then this restriction can be removed.
                raise ValueError(
                    f"Number of programs ({len(programs)}) not divisible"
                    f" by pipeline_chunk_count ({pipeline_chunk_count})",
                )

        exp = Experiment(signals=_experiment_signals(qubit_map))

        calibration = Calibration(_calibration_from_qubits(qubit_map.values()))
        exp.set_calibration(calibration)

        experiment_index = parameter.LinearSweepParameter(
            uid="index",
            start=0,
            stop=len(programs) - 1,
            count=len(programs),
        )

        if batch_execution_mode == "nt":
            maybe_nt_sweep = exp.sweep(experiment_index)
        else:
            maybe_nt_sweep = nullcontext()

        with maybe_nt_sweep:
            with exp.acquire_loop_rt(
                count=options.count,
                averaging_mode=options.averaging_mode,
                acquisition_type=options.acquisition_type
                or enums.AcquisitionType.INTEGRATION,
                reset_oscillator_phase=options.reset_oscillator_phase,
            ):
                sweep_kwargs = {}
                if batch_execution_mode != "nt":
                    if batch_execution_mode == "pipeline":
                        # pipelined sweep with specified programs per chunk
                        sweep_kwargs["chunk_count"] = pipeline_chunk_count
                    maybe_rt_sweep = exp.sweep(experiment_index, **sweep_kwargs)
                else:
                    maybe_rt_sweep = nullcontext()

                with maybe_rt_sweep:
                    if options.add_reset:
                        with exp.section(uid="qubit reset") as reset_section:
                            for qubit in _flatten_qubits(qubit_map):
                                reset_section.add(
                                    self.qpu.quantum_operations["reset"](qubit)
                                )

                    with exp.section(
                        alignment=enums.SectionAlignment.RIGHT,
                        length=options.repetition_time,
                    ):
                        with exp.match(
                            sweep_parameter=experiment_index,
                        ):
                            for i, program in enumerate(programs):
                                with exp.case(i) as c:
                                    ret = importer(text=program)
                                    qasm_section = ret.section
                                    importer_implicit_calibration = (
                                        ret.implicit_calibration
                                    )
                                    importer_acquire_loop_options = (
                                        ret.acquire_loop_options
                                    )
                                    if importer_implicit_calibration:
                                        _logger.warning(
                                            "Implicit calibration (e.g. use of set_frequency in an OpenQASM program) is not supported in multi program experiments."
                                        )
                                    if importer_acquire_loop_options:
                                        _logger.warning(
                                            "OpenQASM setting of acquire loop parameters via pragmas is not supported in multi program experiments."
                                        )
                                    c.add(qasm_section)

                    # read out all qubits
                    if options.add_measurement:
                        with exp.section(uid="qubit_readout") as readout_section:
                            for qubit in _flatten_qubits(qubit_map):
                                readout_section.add(
                                    self.qpu.quantum_operations["measure"](
                                        qubit, handle=qubit.uid
                                    )
                                )
                                if options.add_reset:
                                    with exp.section():
                                        # The next shot will immediately start with an active reset.
                                        # SHFQA needs some time to process previous results before
                                        # it can trigger the next measurement. So we add a delay
                                        # here to have sufficient margin between the two readouts.
                                        # In the future, we'll ideally not have resort to two
                                        # measurements (one for readout, one for reset) in the
                                        # first place.

                                        # TODO: Remove this magic number.
                                        exp.delay(qubit.signals["measure"], 500e-9)
            return exp

    def _preprocess_externs(
        self, externs: dict[str, str | tuple[str, str]]
    ) -> dict[str, str]:
        out = {}
        for name, function_or_port in externs.items():
            if isinstance(function_or_port, device.Port):
                uid = function_or_port.qubit
                signal = function_or_port.signal
                qubit = self.qpu.qubit_by_uid(uid)
                if signal in qubit.signals:
                    # Partial path
                    # Qubit signal lookup does not work with full paths (e.g. alias drive/drive_line)
                    out[name] = qubit.signals[signal]
                elif signal in self._qubit_signals:
                    out[name] = signal
                else:
                    msg = f"Invalid port mapping. Signal {function_or_port} could not be found within qubits."
                    raise ValueError(msg) from None
            elif callable(function_or_port):
                out[name] = function_or_port
            else:
                raise TypeError("Externs must be either a `callable` or an `Port`.")
        return out

    def _get_qubit(
        self, uid_or_qubit: str | quantum.QuantumElement
    ) -> quantum.QuantumElement:
        try:
            if isinstance(uid_or_qubit, quantum.QuantumElement):
                # Will raise KeyError if qubit does not exists.
                # Use the supplied qubit if the UID exists in QPU.
                self.qpu.qubit_by_uid(uid_or_qubit.uid)
                return uid_or_qubit
            return self.qpu.qubit_by_uid(uid_or_qubit)
        except KeyError as error:
            msg = f"Qubit {uid_or_qubit} does not exist in the QPU."
            raise ValueError(msg) from error

    def _create_qubit_register(
        self,
        qubit_map: dict[
            str, str | list[str] | quantum.QuantumElement | list[quantum.QuantumElement]
        ],
    ):
        mapped_qubits = {}
        for name, qubit in qubit_map.items():
            if not isinstance(qubit, list):
                mapped_qubits[name] = self._get_qubit(qubit)
            else:
                mapped_qubits[name] = [self._get_qubit(q) for q in qubit]
        return mapped_qubits


def _experiment_signals(qubit_register: dict) -> list[ExperimentSignal]:
    """Return a list of experiment signals from all qubits in the register.

    Arguments:
        qubit_register: A dictionary representing a qubit register.
        {"q0": q0, "qreg0": [q1, q2]}

    Returns:
        A list of experiment signals.

    Raises:
        LabOneQException: Signal with the same id is already assigned.
    """

    signals = []
    for qubit in qubit_register.values():
        if isinstance(qubit, list):
            nested_qubits = qubit
        else:
            nested_qubits = [qubit]
        for q in nested_qubits:
            for exp_signal in q.experiment_signals():
                if exp_signal in signals:
                    msg = f"Signal with id {exp_signal.uid} already assigned."
                    raise LabOneQException(msg)
                signals.append(exp_signal)
    return signals


def _flatten_qubits(qubit_register: dict) -> list[QuantumElement]:
    """Return a list of all qubits in the register.

    Arguments:
        qubit_register: A dictionary representing a qubit register.
        {"q0": q0, "qreg0": [q1, q2]}

    Returns:
        A flattened list of qubits.
    """
    qubits = []
    for qubit in qubit_register.values():
        if isinstance(qubit, list):
            qubits.extend(qubit)
        else:
            qubits.append(qubit)
    return qubits
