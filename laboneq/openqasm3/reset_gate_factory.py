# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

from laboneq.dsl.experiment.acquire import Acquire
from laboneq.dsl.experiment.play_pulse import PlayPulse
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.section import Case, Match, Section
from laboneq.dsl.experiment.utils import id_generator

if TYPE_CHECKING:
    from laboneq.openqasm3.openqasm3_importer import GateStore

# todo: The following four blocks shall be moved once the corresponding task is to be
#  implemented.


class SignalLineType(Enum):
    """An enum for the different signal line types."""

    MEASURE = "measure"
    ACQUIRE = "acquire"
    DRIVE = "drive"


def default_signal_mapper(signal_line_type: SignalLineType, qubit_index: int) -> str:
    """Default signal mapper.

    Args:
        signal_line_type: The signal line type to map.
        qubit_index: The index of the qubit to map the signal for.

    Returns:
        The mapped signal.
    """
    return f"q{qubit_index}_{signal_line_type.value}"


def create_measure_gate(
    qubit_index: int,
    measure_pulse: Pulse,
    acquire_kernel: Pulse,
    handle: Optional[str] = None,
    signal_mapper: Callable[[SignalLineType, int], str] = default_signal_mapper,
) -> Callable[[], Section]:
    """Creates a measure gate factory.

    Args:
        qubit_index: The index of the qubit to measure.
        measure_pulse: The pulse to use for measure pulse.
        acquire_kernel: The pulse to use as integration kernel.
        handle: The handle to use for the acquire, auto-generated if not provided.
        signal_mapper: A callable to general a signal id from the line type and qubit
            index

    Returns:
        A section modelling the measure gate.
    """

    def impl():
        uid = id_generator(f"measure_q{qubit_index}")
        nonlocal handle
        if handle is None:
            handle = uid

        measure_section = Section(uid=uid)
        measure_section.play(
            signal=signal_mapper(SignalLineType.MEASURE, qubit_index),
            pulse=measure_pulse,
        )
        measure_section.acquire(
            signal=signal_mapper(SignalLineType.ACQUIRE, qubit_index),
            kernel=acquire_kernel,
            handle=handle if handle is not None else uid,
        )
        return measure_section

    return impl


def create_x_gate(
    qubit_index: int,
    x_pulse: Pulse,
    angle_deg: float,
    signal_mapper: Callable[[SignalLineType, int], str] = default_signal_mapper,
) -> Callable[[], Section]:
    """Creates an x gate.

    Args:
        qubit_index: The index of the qubit to apply the x gate to.
        x_pulse: The pulse to use for the x gate.

    Returns:
        A section modelling the x gate.
    """

    def impl():
        uid = id_generator(f"x{angle_deg}_q{qubit_index}")

        x_section = Section(uid=uid)
        x_section.play(
            signal=signal_mapper(SignalLineType.DRIVE, qubit_index), pulse=x_pulse
        )

        return x_section

    return impl


def create_reset_gate(
    qubit_index: int,
    local: bool,
    reuse_measurement_handle: bool,
    gate_store: GateStore,
) -> Callable[[], Section]:
    """Creates a reset gate.

    Args:
        qubit_index: The index of the qubit to apply the reset gate to.
        local: Whether to use local feedback (SHFQC only) or go via the PQSC.
        reuse_measurement_handle: Whether to reuse the handle used for other
          measurements of this qubit or create a new handle
        gate_store: A dictionary of sections to use for the measure and flip pulse.

    Returns:
        A section modelling the reset gate.
    """

    def impl():
        uid = id_generator(f"reset_q{qubit_index}_l{local}_r{reuse_measurement_handle}")
        handle = None

        acquire_section = gate_store.lookup_gate("measure", (qubit_index,))
        acquire_section.uid = uid + "_acquire"
        for c in acquire_section.children:
            if isinstance(c, Acquire):
                if reuse_measurement_handle:
                    handle = c.handle
                else:
                    handle = id_generator(f"reset_q{qubit_index}")
                    c.handle = handle

        if handle is None:
            raise RuntimeError(
                f"No valid acquire operation in measurement gate for q{qubit_index}."
            )

        case0 = Case(uid=id_generator("NOOP"), state=0)
        case1 = Case.from_section(
            gate_store.lookup_gate("x180", (qubit_index,)), state=1
        )
        signals = {p.signal for p in case1.children if isinstance(p, PlayPulse)}
        for s in signals:
            acquire_section.reserve(signal=s)

        match_section = Match(uid=uid + "_match", handle=handle, local=local)
        match_section.add(case0)
        match_section.add(case1)

        reset_section = Section(uid=uid)
        reset_section.add(acquire_section)
        reset_section.add(match_section)

        return reset_section

    return impl