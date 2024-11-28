# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Build DSL experiments that use quantum operations on qubits."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, overload

from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import builtins
from laboneq.dsl.quantum import QuantumElement
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from laboneq.simple import (
        Experiment,
        ExperimentSignal,
    )


class ExperimentBuilder:
    """A builder for functions that create DSL experiments for qubits.

    The builder takes care of creating the DSL `Experiment` object
    including:

    - giving the experiment a name
    - adding the qubit signals as experiment signals
    - adding the qubit calibration to the experiment calibration

    By default, the builder calls `exp_func` in an experiment context
    and returns the resulting experiment.

    If `context=False`, the builder calls `exp_func` and passes the
    experiment object directly as the first argument.

    If needed, the experiment calibration may be accessed within
    `exp_func` using [laboneq.dsl.experiment.builtins.experiment_calibration]()
    when `context=True`.

    The set of qubits is detected by inspecting the arguments supplied when the
    builder is called and consists of:

    - any qubit supplied directly as an argument
    - all the qubits in any list or tuple supplied directly as an argument

    Qubits in tuples or lists containing a mix of qubits and non-qubits are
    ignored.

    During the detection, any instance of [laboneq.simple.QuantumElement]()
    or its sub-classes is considered a qubit.

    Arguments:
        exp_func:
            The function that builds the experiment (or None).

            When calling `__call__` the `*args` and `**kw`
            are passed directly to `exp_func`.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.
        context:
            If true, `exp_func` is called inside an experiment context
            using `exp_func(*args, **kw)`.

            If false, `exp_func` is passed the experiment as the first
            argument instead using `exp_func(exp, *args, **kw)` and no
            experiment context is active.

    Examples:
        ```python
        # Using the builder to create an experiment context:


        def rx_exp(qops, q, angle):
            qops.rx(q, angle)


        rx = ExperimentBuilder(rx_exp, name="rx")
        exp = rx(qops, q0, 0.5 * np.pi)

        # Using the builder without an experiment context:


        def rx_contextless(exp, qops, q, angle):
            exp.add(qops.rx(q, angle))


        rx = ExperimentBuilder(rx_contextless, name="rx", context=False)
        exp = rx(qops, q0, 0.5 * np.pi)
        ```
    """

    def __init__(
        self, exp_func: Callable, name: str | None = None, *, context: bool = True
    ):
        if name is None:
            name = exp_func.__name__
        self.exp_func = exp_func
        self.name = name
        self.context = context

    def __call__(self, *args, **kw):
        """Build the experiment.

        Arguments:
            *args:
                Positional arguments to pass to `exp_func`.
            **kw:
                Keyword arguments to pass to `exp_func`.

        Returns:
            A LabOne Q experiment.
        """
        qubits = _qubits_from_args_and_kws(args, kw)
        signals = _exp_signals_from_qubits(qubits)
        calibration = _calibration_from_qubits(qubits)

        with builtins.experiment(uid=self.name, signals=signals) as exp:
            exp_calibration = builtins.experiment_calibration()
            exp_calibration.calibration_items.update(calibration)
            if self.context:
                self.exp_func(*args, **kw)

        if not self.context:
            self.exp_func(exp, *args, **kw)

        return exp


T = ParamSpec("T")


@overload
def qubit_experiment(
    exp_func: None = ..., name: str | None = ..., context: bool = ...
) -> Callable[[Callable[T]], Callable[T, Experiment]]: ...


@overload
def qubit_experiment(
    exp_func: Callable[T], name: str | None = ..., context: bool = ...
) -> Callable[T, Experiment]: ...


def qubit_experiment(
    exp_func: Callable[T] | None = None,
    name: str | None = None,
    *,
    context: bool = True,
) -> Callable[T, Experiment] | Callable[[Callable[T]], Callable[T, Experiment]]:
    """Decorator for functions that build experiments for qubits.

    Arguments:
        exp_func:
            The wrapped function that builds the experiment.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.
        context:
            If true, the decorated function `exp_func` is called
            inside an experiment context using `exp_func(*args, **kw)`.

            If false, the decorated function `exp_func` is passed the
            experiment as the first argument instead using
            `exp_func(exp, *args, **kw)` and no experiment context is
            active.

    Returns:
        If `exp_func` is given, returns `exp_func` wrapped in an `ExperimentBuilder`.
        Otherwise returns a partial evaluation of `qubit_experiment` with the other
        parameters already set.

    Examples:
        ```python
        @qubit_experiment
        def rx_exp(qops, q, angle):
            qops.rx(q, angle)


        @qubit_experiment(name="rx_exp")
        def my_exp(qops, q, angle):
            qops.rx(q, angle)


        @qubit_experiment(context=False)
        def my_exp_2(exp, qops, q, andle):
            exp.add(qops.rx(q, angle))
        ```
    """
    if exp_func is None:
        return functools.partial(qubit_experiment, name=name, context=context)

    builder = ExperimentBuilder(exp_func, name=name, context=context)

    @functools.wraps(exp_func)
    def build_qubit_experiment(*args, **kw) -> Experiment:
        return builder(*args, **kw)

    return build_qubit_experiment


def build(
    exp_func: Callable, *args, name: str | None = None, context: bool = True, **kw
) -> Experiment:
    """Build an experiment that accepts qubits as arguments.

    Arguments:
        exp_func:
            The function that builds the experiment.
            The `*args` and `**kw` are passed directly to
            this function.
        name:
            A name for this experiment. Defaults to
            `exp_func.__name__` if no name is given.
        context:
            If true, `exp_func` is called inside an experiment context
            using `exp_func(*args, **kw)`.

            If false, `exp_func` is passed the experiment as the first
            argument instead, using `exp_func(exp, *args, **kw)` and no
            experiment context is active.
        *args (tuple):
            Positional arguments to pass to `exp_func`.
        **kw (dict):
            Keyword arguments to pass to `exp_func`.

    Returns:
        A LabOne Q experiment.

    Examples:
        ```python
        def rx_exp(qops, q, angle):
            qops.rx(q, angle)


        exp = build(rx_exp, q0, 0.5 * np.pi)
        ```
    """
    builder = ExperimentBuilder(exp_func, name=name, context=context)
    return builder(*args, **kw)


def _is_qubit(obj: object) -> bool:
    """Return True if an object is a qubit."""
    return isinstance(obj, QuantumElement)


def _is_qubit_list_or_tuple(obj: object) -> bool:
    """Return True if an object is a list or tuple of qubits."""
    if not isinstance(obj, (tuple, list)) or not obj:
        return False
    # all(...) short-circuits so long lists of qubits are not
    # traversed:
    return all(isinstance(x, QuantumElement) for x in obj)


def _qubits_from_args(args: tuple[object]) -> list[QuantumElement]:
    """Return a list of qubits found in a list of arguments."""
    qubits = []
    for arg in args:
        if _is_qubit(arg):
            qubits.append(arg)
        elif _is_qubit_list_or_tuple(arg):
            qubits.extend(arg)
    return qubits


def _qubits_from_args_and_kws(
    args: tuple[object],
    kws: dict[str, object],
) -> list[QuantumElement]:
    """Return a list of qubits found in either positional or keyword arguments."""
    qubits = _qubits_from_args(args)
    qubits.extend(_qubits_from_args(kws.values()))
    return qubits


def _exp_signals_from_qubits(qubits: list[QuantumElement]) -> list[ExperimentSignal]:
    """Return a list of experiment signals from a list of qubits."""
    signals = []
    for qubit in qubits:
        for exp_signal in qubit.experiment_signals():
            if exp_signal in signals:
                msg = f"Signal with id {exp_signal.uid} already assigned."
                raise LabOneQException(msg)
            signals.append(exp_signal)
    return signals


def _calibration_from_qubits(
    qubits: list[QuantumElement],
) -> dict[str,]:
    """Return the calibration objects from a list of qubits."""
    calibration = {}
    for qubit in qubits:
        calibration.update(qubit.calibration())
    return calibration
