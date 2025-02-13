# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Core classes for defining sets of quantum operations on qubits."""

from __future__ import annotations

import contextlib
import functools
import inspect
import textwrap
from numbers import Number
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
from laboneq.dsl.experiment import builtins, pulse_library
from laboneq.dsl.experiment.build_experiment import _qubits_from_args
from laboneq.dsl.parameter import Parameter
from laboneq.core.types.enums.execution_type import ExecutionType
from laboneq.core.utilities.highlight import pygmentize


if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.simple import (
        QuantumElement,
        Section,
    )


def quantum_operation(
    f: Callable | None = None, *, broadcast: bool = True, neartime: bool = False
) -> Callable:
    """Decorator that marks an method as a quantum operation.

    Methods marked as quantum operations are moved into the `BASE_OPS` dictionary
    of the `QuantumOperations` class they are defined in at the end of class
    creation.

    Functions marked as quantum operations take the `QuantumOperations` instance
    they are registered on as their initial `self` parameter.

    Arguments:
        f:
            The method to decorate. `f` should take as positional arguments the
            qubits to operate on. It may take additional arguments that
            specify other parameters of the operation.
        broadcast:
            If true, the operation may be broadcast across multiple qubits
            using `.broadcast`. Broadcasting requires that all operation
            arguments do not accept sequences of values (when not broadcasting).
        neartime:
            If true, the operation is marked as a near-time operation. Its
            section is set to near-time and no qubit signals are reserved.

    Returns:
        If `f` is given, returns the decorated method.
        Otherwise returns a partial evaluation of `quantum_operation` with
        the other arguments set.
    """
    if f is None:
        return functools.partial(
            quantum_operation, broadcast=broadcast, neartime=neartime
        )
    f._quantum_op = _QuantumOperationMarker(broadcast=broadcast, neartime=neartime)
    return f


class _QuantumOperationMarker:
    """A marker indicating that a function is a quantum operation.

    Arguments:
        broadcast:
            If true, the operation may be broadcast across multiple qubits
            using `.broadcast`. Broadcasting requires that all operation
            arguments do not accept sequences of values (when not broadcasting).
        neartime:
            If true, the operation is marked as a near-time operation. Its
            section is set to near-time and no qubit signals are reserved.
    """

    def __init__(self, *, broadcast: bool = True, neartime: bool = False):
        self.broadcast = broadcast
        self.neartime = neartime


class _PulseCache:
    """A cache for pulses to ensure that each unique pulse is only created once."""

    GLOBAL_CACHE: ClassVar[dict[tuple, Pulse]] = {}

    def __init__(self, cache: dict | None = None):
        if cache is None:
            cache = {}
        self.cache = cache

    @classmethod
    def experiment_or_global_cache(cls) -> _PulseCache:
        """Return an pulse cache.

        If there is an active experiment context, return its cache. Otherwise
        return the global pulse cache.
        """
        context = builtins.current_experiment_context()
        if context is None:
            return cls(cls.GLOBAL_CACHE)
        if not hasattr(context, "_pulse_cache"):
            context._pulse_cache = cls()
        return context._pulse_cache

    @classmethod
    def reset_global_cache(cls) -> None:
        cls.GLOBAL_CACHE.clear()

    def _parameter_value_key(self, key: str, value: object) -> object:
        if isinstance(value, Parameter):
            return (value.uid, tuple(value.values))
        if isinstance(value, list):
            if all(isinstance(x, Number) for x in value):
                return tuple(value)
            raise ValueError(
                f"Pulse parameter {key!r} is a list of values that are not all numbers."
                " It cannot be cached by create_pulse(...)."
            )
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number) and len(value.shape) == 1:
                return tuple(value)
            raise ValueError(
                f"Pulse parameter {key!r} is a numpy array whose values are not all"
                " numbers or whose dimension is not one. It cannot be cached by"
                " create_pulse(...)."
            )
        return value

    def _key(self, name: str, function: str, parameters: dict) -> tuple:
        parameters = {k: self._parameter_value_key(k, v) for k, v in parameters.items()}
        return (name, function, tuple(sorted(parameters.items())))

    def get(self, name: str, function: str, parameters: dict) -> Pulse | None:
        """Return the cache pulse or `None`."""
        key = self._key(name, function, parameters)
        return self.cache.get(key, None)

    def store(self, pulse: Pulse, name: str, function: str, parameters: dict) -> None:
        """Store the given pulse in the cache."""
        key = self._key(name, function, parameters)
        self.cache[key] = pulse


def create_pulse(
    parameters: dict,
    overrides: dict | None = None,
    name: str | None = None,
) -> Pulse:
    """Create a pulse from the given parameters and parameter overrides.

    The parameters are dictionary that contains:

      - a key `"function"` that specifies which function from the LabOne Q
        `pulse_library` to use to construct the pulse. The function may
        either be the name of a registered pulse functional or
        `"sampled_pulse"` which uses `pulse_library.sampled_pulse`.
      - any other parameters required by the given pulse function.

    Arguments:
        parameters:
            A dictionary of pulse parameters. If `None`, then the overrides
            must completely define the pulse.
        overrides:
            A dictionary of overrides for the pulse parameters.
            If the overrides changes the pulse function, then the
            overrides completely replace the existing pulse parameters.
            Otherwise they extend or override them.
            The dictionary of overrides may contain sweep parameters.
        name:
            The name of the pulse. This is used as a prefix to generate the
            pulse `uid`.

    Returns:
        pulse:
            The pulse described by the parameters.
    """
    if overrides is None:
        overrides = {}
    if "function" in overrides and overrides["function"] != parameters["function"]:
        parameters = overrides.copy()
    else:
        parameters = {**parameters, **overrides}

    function = parameters.pop("function")

    if function == "sampled_pulse":
        # special case the sampled_pulse function that is not registered as a
        # pulse functional:
        pulse_function = pulse_library.sampled_pulse
    else:
        try:
            pulse_function = pulse_library.pulse_factory(function)
        except KeyError as err:
            raise ValueError(f"Unsupported pulse function {function!r}.") from err

    if name is None:
        name = "unnamed"

    pulse_cache = _PulseCache.experiment_or_global_cache()
    pulse = pulse_cache.get(name, function, parameters)
    if pulse is None:
        pulse = pulse_function(uid=builtins.uid(name), **parameters)
        pulse_cache.store(pulse, name, function, parameters)

    return pulse


class QuantumOperations:
    """Quantum operations for a given qubit type.

    Attributes:
        QUBIT_TYPES:
            (class attribute) The classes of qubits supported by this set of
            operations. The value may be a single class or a tuple of classes.
        BASE_OPS:
            (class attribute) A dictionary of names and functions that define
            the base operations provided.
    """

    QUBIT_TYPES: type[QuantumElement] | tuple[type[QuantumElement]] | None = None
    BASE_OPS: dict[str, Callable] = None

    def __init__(self):
        if self.QUBIT_TYPES is None:
            raise ValueError(
                "Sub-classes of QuantumOperations must set the supported QUBIT_TYPES.",
            )

        self._ops = {}

        for name, f in self.BASE_OPS.items():
            self.register(f, name=name)

    def __init_subclass__(cls, **kw):
        """Move any quantum operations into BASE_OPS."""
        if cls.BASE_OPS is None:
            cls.BASE_OPS = {}

        # Collect quantum operations from the class and its parent classes
        base_ops = {}
        for base in reversed(cls.mro()):
            ops = getattr(base, "BASE_OPS", None)
            if ops is None:
                continue
            base_ops.update(
                {k: v for k, v in ops.items() if getattr(v, "_quantum_op", False)}
            )

        quantum_ops = {
            k: v for k, v in cls.__dict__.items() if getattr(v, "_quantum_op", False)
        }

        for k in quantum_ops:
            delattr(cls, k)
        cls.BASE_OPS = {**base_ops, **quantum_ops}

        super().__init_subclass__(**kw)

    def __getattr__(self, name: str):
        """Retrieve an operation."""
        op = self._ops.get(name, None)
        if op is None:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}",
            )
        return op

    def __getitem__(self, name: str):
        """Retrieve an operation."""
        return self._ops[name]

    def __setitem__(self, name: str, f: Operation | Callable):
        """Replace or register an operation."""
        if isinstance(f, Operation):
            self._ops[name] = f
        else:
            self.register(f, name=name)

    def __contains__(self, name: str):
        """Return true if the set of operations contains the given name."""
        return name in self._ops

    def __dir__(self):
        """Return the attributes of these quantum operations."""
        return sorted(super().__dir__() + list(self._ops.keys()))

    def keys(self) -> list[str]:
        """Return the names of the registered quantum operations."""
        return sorted(self._ops.keys())

    def register(self, f: Callable, name: str | None = None) -> None:
        """Registers a quantum operation.

        The given operation is wrapped in a `Operation` instance
        and added to this set of operations.

        Arguments:
            f:
                The function to register as a quantum operation.

                The first parameter of `f` should be the set of quantum
                operations to use. This allows `f` to use other quantum
                operations if needed.

                The qubits `f` operates on must be passed as positional
                arguments, not keyword arguments.

                Additional non-qubit arguments may be passed to `f` as
                either positional or keyword arguments.
            name:
                The name of the operation. Defaults to `f.__name__`.

        Example:
            Create a custom operation function, register and
            call it:

            ```python
            qop = qpu.quantum_operations


            def custom_op(qop, q, amplitude):
                pulse = ...
                play(
                    q.signals["drive"],
                    amplitude=amplitude,
                    pulse=pulse,
                )


            qop.register(custom_op)
            qop.custom_op(q, amplitude=0.5)
            ```

            In the example above the `qop` argument to `custom_op`
            is unused, but `custom_op` could call another quantum
            operation using, e.g., `qop.x90(q)`, if needed.
        """
        name = name if name is not None else f.__name__
        self._ops[name] = Operation(f, name, self)


class Operation:
    """An operation on one or more qubits.

    Arguments:
        op:
            The callable that implements the operation.
            `op` should take as positional arguments the
            qubits to operate on. It may take additional
            arguments that specify other parameters of the
            operation.
        op_name:
            The name of the operation (usually the same as that
            of the implementing function).
        quantum_ops:
            The quantum operations object the operation is for.
    """

    def __init__(
        self,
        op: Callable,
        op_name: str,
        quantum_ops: QuantumOperations,
    ):
        self._op = op
        self._op_name = op_name
        self._op_marker = getattr(op, "_quantum_op", _QuantumOperationMarker())
        self._quantum_ops = quantum_ops
        self.__doc__ = self._op.__doc__

    def __call__(self, *args, **kw) -> Section | list[Section]:
        """Build a section using the operation.

        The operation is called in the context of a pre-built
        section instance.

        The UID of the section is generated with the name of the operation
        as a prefix and a unique count as a suffix.

        If the operation can be broadcast and any of the positional arguments
        are lists or tuples, then the operation will be broadcast.

        Broadcasting is currently an experimental feature.

        Arguments:
            *args:
                Positional arguments for the operation.
            **kw:
                Keyword parameters for the operation.

        Returns:
            A LabOne Q section built by the operation, or a list of
            LabOne Q sections if the operation is broadcast.
        """
        return self._call(args, kw)

    def __repr__(self) -> str:
        return (
            f"Operation(op={self._op}, op_name={self._op_name},"
            f" neartime={self._op_marker.neartime!r},"
            f" supports_broadcast={self._op_marker.broadcast!r})"
        )

    def omit_section(self, *args: object, **kw: object) -> None | list[None]:
        """Calls the operation but *without* building a new section.

        Omitting the section causes the contents of the operation to be added directly
        to the existing section context when the operation is called.

        This is intended to reduce the number of sections created when one operation
        consists entirely of calling another operation. In other cases it should be used
        with care since omitting a section may affect the generated signals.

        Broadcasting an operation while omitting sections is supported but not advised
        without extreme care. Such a broadcast will omit the section of each contained
        operation and return a list consisting of a number `None` values equal to the
        size of the broadcast. The contents of all the individual operations will be
        added directly to the existing section context.

        Arguments:
            *args:
                Positional arguments to the operation.
            **kw:
                Keyword parameters for the operation.

        Returns:
            None.

        Raises:
            LabOneQException:
                If no active section context exists.
        """
        self._call(args, kw, omit_section=True)

    def omit_reserves(self, *args: object, **kw: object) -> Section | list[Section]:
        """Calls the operation but *without* reserving the qubit signals.

        This should be used with care. Reserving the signals ensures that operations on
        the same qubit do not overlap. When the reserves are omitted, the caller must
        take care themselves that any overlaps of operations on the same qubit are
        avoided or intended.

        Broadcasting an operation while omitting reserves is supported. Each operation
        in the broadcast will omit its reserves.

        Arguments:
            *args:
                Positional arguments to the operation.
            **kw:
                Keyword parameters for the operation.

        Returns:
            A LabOne Q section containing the operation.
        """
        return self._call(args, kw, omit_reserves=True)

    def _duplicate_qubit_uids(self, qubits: list[QuantumElement]) -> list[str]:
        """Return a list of sorted duplicate qubit uids."""
        seen_uids = set()
        duplicate_qubit_uids = set()
        for q in qubits:
            if q.uid in seen_uids:
                duplicate_qubit_uids.add(q.uid)
            else:
                seen_uids.add(q.uid)
        return sorted(duplicate_qubit_uids)

    def _broadcast_call(
        self,
        args: tuple,
        kw: dict,
        *,
        omit_section: bool = False,
        omit_reserves: bool = False,
    ) -> list[Section] | list[None]:
        """Broadcast the operation with the supplied parameters and additional options.

        Arguments:
            args:
                Positional arguments to the operation.
            kw:
                Keyword parameters for the operation.
            omit_section:
                The `omit_section` flag is passed to each individual operation
                within the broadcast.
            omit_reserves:
                The `omit_reserves` flag is passed to each individual operation
                within the broadcast.

        Returns:
            If omit_section is false, a list of LabOne Q sections for each of the
            broadcast operations.
            If omit_section is true, a list of `None`s for each of the broadcast
            operations.
        """
        # Broadcasting requires that all operation arguments do not accept lists
        # or tuples of values (when not broadcasting), so we find all such lists
        # or tuples in args and kwargs and treat those as broadcast.

        if not self._op_marker.broadcast:
            raise ValueError(
                f"Quantum operation {self._op_name!r} does not support broadcasting."
            )

        args_map = {
            i: arg for i, arg in enumerate(args) if isinstance(arg, (tuple, list))
        }

        if not args_map:
            raise ValueError(
                f"Quantum operation {self._op_name!r} was being broadcast but"
                f" no lists or tuples were found to broadcast over."
            )

        first_arg = next(iter(args_map))
        broadcast_length = len(args_map[first_arg])

        kw_map = {k: v for k, v in kw.items() if isinstance(v, (tuple, list))}

        invalid_args = [
            (i, v) for i, v in args_map.items() if len(v) != broadcast_length
        ]
        if invalid_args:
            summary = ", ".join(
                f"arg[{i}] has length {len(v)}" for i, v in invalid_args
            )
            raise ValueError(
                f"Quantum operation {self._op_name!r} was being broadcast with length"
                f" {broadcast_length} but the following positional arguments have"
                f" different lengths: {summary}"
            )

        invalid_kw = [k for k, v in kw_map.items() if len(v) != broadcast_length]
        if invalid_kw:
            summary = ", ".join(
                f"kw[{k!r}] has length {len(kw_map[k])}" for k in invalid_kw
            )
            raise ValueError(
                f"Quantum operation {self._op_name!r} was being broadcast with length"
                f" {broadcast_length} but the following keyword arguments have"
                f" different lengths: {summary}"
            )

        qubits = _qubits_from_args(args)
        duplicate_qubit_uids = self._duplicate_qubit_uids(qubits)
        if duplicate_qubit_uids:
            duplicate_uids = ", ".join(duplicate_qubit_uids)
            raise ValueError(
                f"Quantum operation {self._op_name!r} was given the following"
                f" non-unique qubits as arguments when being broadcast:"
                f" {duplicate_uids}"
            )

        op_sections = []

        for i in range(broadcast_length):
            call_args = [
                args_map[j][i] if j in args_map else args[j] for j in range(len(args))
            ]
            call_kw = {k: kw_map[k][i] if k in kw_map else kw[k] for k in kw}
            op_sections.append(
                self._single_call(
                    call_args,
                    call_kw,
                    omit_section=omit_section,
                    omit_reserves=omit_reserves,
                )
            )

        return op_sections

    def _single_call(
        self,
        args: tuple,
        kw: dict,
        *,
        omit_section: bool = False,
        omit_reserves: bool = False,
    ) -> Section | None:
        """Calls the operation with the supplied parameters and additional options.

        Arguments:
            args:
                Positional arguments to the operation.
            kw:
                Keyword parameters for the operation.
            omit_section:
                If omit_section is true, the operation is added to the existing
                section context and no new section is created.
            omit_reserves:
                If omit_reserves is true, the qubit signal lines are not
                reserved.

        Returns:
            If omit_section is false, a LabOne Q section containing the operation.
            If omit_section is true, no section is returned and the operation is
            added to the existing section context.
            If broadcasting, a list of created sections or a list of `None` values
            is returned.
        """
        qubits = _qubits_from_args(args)
        qubits_with_incorrect_type = [
            q.uid for q in qubits if not isinstance(q, self._quantum_ops.QUBIT_TYPES)
        ]

        if qubits_with_incorrect_type:
            if isinstance(self._quantum_ops.QUBIT_TYPES, type):
                supported_qubit_types = self._quantum_ops.QUBIT_TYPES.__name__
            else:
                supported_qubit_types = ", ".join(
                    x.__name__ for x in self._quantum_ops.QUBIT_TYPES
                )
            unsupported_qubits = ", ".join(qubits_with_incorrect_type)
            raise TypeError(
                f"Quantum operation {self._op_name!r} was passed the following"
                f" qubits that are not of a supported qubit type: {unsupported_qubits}."
                f" The supported qubit types are: {supported_qubit_types}.",
            )

        duplicate_qubit_uids = self._duplicate_qubit_uids(qubits)
        if duplicate_qubit_uids:
            duplicate_uids = ", ".join(duplicate_qubit_uids)
            raise ValueError(
                f"Quantum operation {self._op_name!r} was given the following"
                f" non-unique qubits as arguments: {duplicate_uids}"
            )

        section_name = "_".join([self._op_name] + [q.uid for q in qubits])
        execution_type = ExecutionType.NEAR_TIME if self._op_marker.neartime else None
        reserve_signals = not (
            omit_reserves or omit_section or self._op_marker.neartime
        )

        if not omit_section:
            maybe_section = builtins.section(
                name=section_name,
                execution_type=execution_type,
            )
        else:
            maybe_section = contextlib.nullcontext()

        with maybe_section as op_section:
            if reserve_signals:
                self._reserve_signals(qubits)
            self._op(self._quantum_ops, *args, **kw)

        return op_section

    def _call(
        self,
        args: tuple,
        kw: dict,
        *,
        omit_section: bool = False,
        omit_reserves: bool = False,
    ) -> Section | None | list[Section] | list[None]:
        """Calls the operation with the supplied parameters and additional options.

        If the operation can be broadcast and any of the positional arguments
        are lists or tuples, then the operation will be broadcast.

        Arguments:
            args:
                Positional arguments to the operation.
            kw:
                Keyword parameters for the operation.
            omit_section:
                If omit_section is true, the operation is added to the existing
                section context and no new section is created.
                If the operation is being broadcast, this flag is passed on to
                the call to each individual operation.
            omit_reserves:
                If omit_reserves is true, the qubit signal lines are not
                reserved.
                If the operation is being broadcast, this flag is passed on to
                the call to each individual operation.

        Returns:
            If omit_section is false, a LabOne Q section containing the operation.
            If omit_section is true, no section is returned and the operation is
            added to the existing section context.
        """
        if self._op_marker.broadcast and any(
            isinstance(arg, (list, tuple)) for arg in args
        ):
            return self._broadcast_call(
                args, kw, omit_section=omit_section, omit_reserves=omit_reserves
            )

        return self._single_call(
            args, kw, omit_section=omit_section, omit_reserves=omit_reserves
        )

    def _reserve_signals(self, qubits: list[QuantumElement]) -> None:
        """Reserve all the signals of a list of qubits."""
        for q in qubits:
            for signal in q.signals.values():
                builtins.reserve(signal)

    @property
    def op(self) -> Callable:
        """Return the implementation of the operation.

        Returns:
            The function implementing the operation.
        """
        return self._op

    @property
    def neartime(self) -> bool:
        """Return whether the operation is neartime or not.

        Returns:
            True if the operation is neartime. False otherwise.
        """
        return self._op_marker.neartime

    @property
    def supports_broadcast(self) -> bool:
        """Return whether the operation supports broadcasting over qubits.

        Returns:
            True if the operation can be broadcast over qubits. False otherwise.
        """
        return self._op_marker.broadcast

    @property
    @pygmentize
    def src(self) -> str:
        """Return the source code of the underlying operation.

        Returns:
            The source code of the underlying operation.
        """
        src = inspect.getsource(self._op)
        return textwrap.dedent(src)
