# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, Flag, auto
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode


class LoopFlags(Flag):
    NONE = 0
    AVERAGE = auto()
    PIPELINE = auto()

    @property
    def is_average(self) -> bool:
        return bool(self & LoopFlags.AVERAGE)

    @property
    def is_pipeline(self) -> bool:
        return bool(self & LoopFlags.PIPELINE)


class LoopingMode(Enum):
    NEAR_TIME_ONLY = auto()
    ONCE = auto()


LOOP_INDEX = "_loop_index"


class ExecutionScope:
    def __init__(self, parent: ExecutionScope | None, root: ExecutorBase):
        self._parent = parent
        self._is_real_time: bool = False if parent is None else parent._is_real_time
        self._root = root
        self._variables: dict[str, Any] = {}

    def enter_real_time(self):
        self._is_real_time = True

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


class Sequence(Statement):
    def __init__(self, sequence=None):
        self.sequence: list[Statement] = sequence or []

    def append_statement(self, statement: Statement):
        self.sequence.append(statement)

    def run(self, scope: ExecutionScope):
        for statement in self.sequence:
            statement.run(scope)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is Sequence:
            return self.sequence == other.sequence
        return NotImplemented

    def __repr__(self):
        return f"Sequence({self.sequence})"


class Nop(Statement):
    def run(self, scope: ExecutionScope):
        pass

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is Nop:
            return True
        return NotImplemented

    def __repr__(self):
        return "Nop()"


class ExecSet(Statement):
    def __init__(self, path: str, val: Any):
        self.path = path
        self.val = val

    def run(self, scope: ExecutionScope):
        val = (
            scope.resolve_variable(self.val) if isinstance(self.val, str) else self.val
        )
        scope.root.set_handler(self.path, val)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecSet:
            return (self.path, self.val) == (other.path, other.val)
        return NotImplemented

    def __repr__(self):
        return f"ExecSet({repr(self.path)}, {repr(self.val)})"


class ExecUserCall(Statement):
    def __init__(self, func_name: str, args: dict[str, Any]):
        self.func_name = func_name
        self.args = args

    def run(self, scope: ExecutionScope):
        resolved_args = {}
        for name, val in self.args.items():
            resolved_args[name] = (
                scope.resolve_variable(val) if type(val) is str else val
            )
        scope.root.user_func_handler(self.func_name, resolved_args)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecUserCall:
            return (self.func_name, self.args) == (other.func_name, other.args)
        return NotImplemented

    def __repr__(self):
        return f"ExecUserCall({repr(self.func_name)}, {repr(self.args)})"


class ExecAcquire(Statement):
    def __init__(self, handle: str, signal: str, parent_uid: str):
        self.handle = handle
        self.signal = min(signal) if isinstance(signal, list) else signal
        self.parent_uid = parent_uid

    def run(self, scope: ExecutionScope):
        scope.root.acquire_handler(self.handle, self.signal, self.parent_uid)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecAcquire:
            return (self.handle, self.signal, self.parent_uid) == (
                other.handle,
                other.signal,
                other.parent_uid,
            )
        return NotImplemented

    def __repr__(self):
        return f"ExecAcquire({repr(self.handle)}, {repr(self.signal)}, {repr(self.parent_uid)})"


class SetSoftwareParamLinear(Statement):
    def __init__(self, name: str, start: float, step: float, axis_name: str = None):
        self.name = name
        self.start = start
        self.step = step
        self.axis_name = axis_name

    def run(self, scope: ExecutionScope):
        index = scope.resolve_variable(LOOP_INDEX)
        value = self.start + self.step * index
        scope.set_variable(self.name, value)
        scope.root.set_sw_param_handler(self.name, index, value, self.axis_name, None)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is SetSoftwareParamLinear:
            return (self.name, self.start, self.step, self.axis_name) == (
                other.name,
                other.start,
                other.step,
                other.axis_name,
            )
        return NotImplemented

    def __repr__(self):
        return f"SetSoftwareParamLinear({self.name}, {self.start}, {self.step}, {repr(self.axis_name)})"


class SetSoftwareParam(Statement):
    def __init__(self, name: str, values: npt.ArrayLike, axis_name: str = None):
        self.name = name
        self.values = values
        self.axis_name = axis_name

    def run(self, scope: ExecutionScope):
        index = max(0, min(scope.resolve_variable(LOOP_INDEX), len(self.values) - 1))
        value = self.values[index]
        scope.set_variable(self.name, value)
        scope.root.set_sw_param_handler(
            self.name, index, value, self.axis_name, self.values
        )

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is SetSoftwareParam:
            return (self.name, self.axis_name) == (
                other.name,
                other.axis_name,
            ) and np.allclose(self.values, other.values)

        return NotImplemented

    def __repr__(self):
        return f"SetSoftwareParam({repr(self.name)}, {repr(self.values)}, {repr(self.axis_name)})"


class ForLoop(Statement):
    def __init__(
        self,
        count: int,
        body: Sequence,
        loop_flags: LoopFlags = LoopFlags.NONE,
    ):
        self.count = count
        self.body = body
        self.loop_flags = loop_flags

    def _loop_iterator(self, scope: ExecutionScope) -> Iterator[int]:
        if scope.root.looping_mode == LoopingMode.NEAR_TIME_ONLY:
            if scope._is_real_time:
                yield 0
            else:
                for i in range(self.count):
                    yield i
        elif scope.root.looping_mode == LoopingMode.ONCE:
            yield 0
        else:
            raise LabOneQException(f"Unknown looping mode '{scope.root.looping_mode}'")

    def run(self, scope: ExecutionScope):
        sub_scope = scope.make_sub_scope()
        for i in self._loop_iterator(scope):
            with scope.root.for_loop_handler(self.count, i, self.loop_flags):
                sub_scope.set_variable(LOOP_INDEX, i)
                self.body.run(sub_scope)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ForLoop:
            return (self.count, self.body, self.loop_flags) == (
                other.count,
                other.body,
                other.loop_flags,
            )
        return NotImplemented

    def __repr__(self):
        return f"ForLoop({self.count}, {self.body}, {self.loop_flags})"


class ExecRT(ForLoop):
    def __init__(
        self,
        count: int,
        body: Sequence,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        super().__init__(count=count, body=body, loop_flags=LoopFlags.AVERAGE)
        self.uid = uid
        self.averaging_mode = averaging_mode
        self.acquisition_type = acquisition_type

    def run(self, scope: ExecutionScope):
        with scope.root.rt_handler(
            self.count, self.uid, self.averaging_mode, self.acquisition_type
        ):
            if scope.root.looping_mode == LoopingMode.NEAR_TIME_ONLY:
                pass
            elif scope.root.looping_mode == LoopingMode.ONCE:
                sub_scope = scope.make_sub_scope()
                sub_scope.enter_real_time()
                super().run(sub_scope)
            else:
                raise LabOneQException(
                    f"Unknown looping mode '{scope.root.looping_mode}'"
                )

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecRT:
            return (
                self.count,
                self.body,
                self.uid,
                self.averaging_mode,
                self.acquisition_type,
            ) == (
                other.count,
                other.body,
                other.uid,
                other.averaging_mode,
                other.acquisition_type,
            )
        return NotImplemented

    def __repr__(self):
        return f"ExecRT({self.count}, {self.body}, {repr(self.uid)}, {self.averaging_mode}, {self.acquisition_type})"


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

    def __init__(self, looping_mode: LoopingMode = LoopingMode.NEAR_TIME_ONLY):
        self.looping_mode: LoopingMode = looping_mode

    def set_handler(self, path: str, value):
        pass

    def user_func_handler(self, func_name: str, args: dict[str, Any]):
        pass

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        pass

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: npt.ArrayLike
    ):
        pass

    @contextmanager
    def for_loop_handler(self, count: int, index: int, loop_flags: LoopFlags):
        yield

    @contextmanager
    def rt_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        yield

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
