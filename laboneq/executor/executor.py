# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Iterator, List

import numpy.typing as npt

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode


class LoopType(Enum):
    SWEEP = auto()
    AVERAGE = auto()
    HARDWARE = auto()


class LoopingMode(Enum):
    EXECUTE = auto()
    ONCE = auto()


LOOP_INDEX = "_loop_index"


class ExecutionScope:
    def __init__(self, parent: ExecutionScope, root: ExecutorBase):
        self._parent = parent
        self._root = root
        self._variables: Dict[str, Any] = {}

    def set_variable(self, name: str, value: Any):
        self._variables[name] = value

    def resolve_variable(self, name):
        if name in self._variables:
            return self._variables[name]
        val = None if self._parent is None else self._parent.resolve_variable(name)
        if val is None:
            raise LabOneQException(f"Reference to unknown parameter '{name}'")
        return val

    def make_sub_scope(self) -> ExecutionScope:
        return ExecutionScope(self, self._root)

    @property
    def root(self):
        return self._root


class Statement(ABC):
    @abstractmethod
    def run(self, scope: ExecutionScope):
        pass


class NoOperation(Statement):
    def run(self, scope: ExecutionScope):
        pass


class Sequence(Statement):
    def __init__(self):
        self._sequence: List[Statement] = []

    def append_statement(self, statement: Statement):
        self._sequence.append(statement)

    def run(self, scope: ExecutionScope):
        for statement in self._sequence:
            statement.run(scope)


class Nop(Statement):
    def run(self, scope: ExecutionScope):
        pass


class ExecSet(Statement):
    def __init__(self, path, val):
        self._path = path
        self._val = val

    def run(self, scope: ExecutionScope):
        val = scope.resolve_variable(self._val) if type(self._val) is str else self._val
        scope.root.set_handler(self._path, val)


class ExecUserCall(Statement):
    def __init__(self, func_name: str, args: Dict[str, Any]):
        self._func_name = func_name
        self._args = args

    def run(self, scope: ExecutionScope):
        resolved_args = {}
        for name, val in self._args.items():
            resolved_args[name] = (
                scope.resolve_variable(val) if type(val) is str else val
            )
        scope.root.user_func_handler(self._func_name, resolved_args)


class ExecAcquire(Statement):
    def __init__(self, handle: str, signal: str, parent_uid: str):
        self._handle = handle
        self._signal = signal
        self._parent_uid = parent_uid

    def run(self, scope: ExecutionScope):
        scope.root.acquire_handler(self._handle, self._signal, self._parent_uid)


class SetSoftwareParamLinear(Statement):
    def __init__(self, name: str, start: float, step: float, axis_name: str = None):
        self._name = name
        self._start = start
        self._step = step
        self._axis_name = axis_name

    def run(self, scope: ExecutionScope):
        index = scope.resolve_variable(LOOP_INDEX)
        value = self._start + self._step * index
        scope.set_variable(self._name, value)
        scope.root.set_sw_param_handler(self._name, index, value, self._axis_name, None)


class SetSoftwareParam(Statement):
    def __init__(self, name: str, values: npt.ArrayLike, axis_name: str = None):
        self._name = name
        self._values = values
        self._axis_name = axis_name

    def run(self, scope: ExecutionScope):
        index = max(0, min(scope.resolve_variable(LOOP_INDEX), len(self._values) - 1))
        value = self._values[index]
        scope.set_variable(self._name, value)
        scope.root.set_sw_param_handler(
            self._name, index, value, self._axis_name, self._values
        )


class ForLoop(Statement):
    def __init__(
        self, count: int, body: Sequence, loop_type: LoopType = LoopType.SWEEP
    ):
        self._count = count
        self._body = body
        self._loop_type = loop_type

    def _loop_iterator(self, scope: ExecutionScope) -> Iterator[int]:
        if scope.root.looping_mode == LoopingMode.EXECUTE:
            if self._loop_type == LoopType.HARDWARE:
                yield 0
            else:
                for i in range(self._count):
                    yield i
        elif scope.root.looping_mode == LoopingMode.ONCE:
            yield 0
        else:
            raise LabOneQException(f"Unknown looping mode '{scope.root.looping_mode}'")

    def run(self, scope: ExecutionScope):
        sub_scope = scope.make_sub_scope()
        for i in self._loop_iterator(scope):
            scope.root.for_loop_handler(self._count, i, True)
            sub_scope.set_variable(LOOP_INDEX, i)
            self._body.run(sub_scope)
            scope.root.for_loop_handler(None, None, False)


class ExecRT(ForLoop):
    def __init__(
        self,
        count: int,
        body: Sequence,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        super().__init__(count, body, LoopType.AVERAGE)
        self._uid = uid
        self._averaging_mode = averaging_mode
        self._acquisition_type = acquisition_type

    def run(self, scope: ExecutionScope):
        scope.root.rt_handler(
            self._count, self._uid, self._averaging_mode, self._acquisition_type, True
        )
        if scope.root.looping_mode == LoopingMode.EXECUTE:
            pass
        elif scope.root.looping_mode == LoopingMode.ONCE:
            super().run(scope)
        else:
            raise LabOneQException(f"Unknown looping mode '{scope.root.looping_mode}'")
        scope.root.rt_handler(None, None, None, None, False)


class ExecutorBase:
    """Base class for the concrete executor.

    Subclass this base class into a concrete executor. Override the necessary
    `*_handler(self, ...)` methods, and implement the desired execution behavior.

    Attributes
    ----------
    looping_mode : LoopingMode
        Controls the way loops are being executed.

    Methods
    -------
    *_handler(self, ...)
        These methods are being called during execution on respective events.
        By default do nothing. Override and implement in your concrete executor
        class as needed.
    """

    def __init__(self, looping_mode: LoopingMode = LoopingMode.EXECUTE):
        self.looping_mode: LoopingMode = looping_mode

    def set_handler(self, path: str, value):
        pass

    def user_func_handler(self, func_name: str, args: Dict[str, Any]):
        pass

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        pass

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: npt.ArrayLike
    ):
        pass

    def for_loop_handler(self, count: int, index: int, enter: bool):
        pass

    def rt_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        enter: bool,
    ):
        pass

    def run(self, root_sequence: Statement):
        """Start execution of the provided sequence."""
        scope = ExecutionScope(None, self)
        root_sequence.run(scope)


class ExecutionFactory:
    def __init__(self):
        self._root_sequence = Sequence()
        self._current_scope = self._root_sequence

    def _append_statement(self, statement: Statement):
        self._current_scope.append_statement(statement)

    def _sub_scope(self, generator, *args):
        new_scope = Sequence()
        saved_scope = self._current_scope
        self._current_scope = new_scope
        generator(*args)
        self._current_scope = saved_scope
        return new_scope
