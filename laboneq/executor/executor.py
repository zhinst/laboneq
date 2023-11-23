# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from laboneq._utils import UIDReference
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


class StatementType(Enum):
    Set = auto()
    NTCallback = auto()
    Acquire = auto()
    SetSwParam = auto()
    ForLoopEntry = auto()
    ForLoopExit = auto()
    RTEntry = auto()
    RTExit = auto()


@dataclass
class Notification:
    statement_type: StatementType
    args: dict[str, Any]


class Statement(ABC):
    @abstractmethod
    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        ...


class Sequence(Statement):
    def __init__(self, sequence=None):
        self.sequence: list[Statement] = sequence or []

    def append_statement(self, statement: Statement):
        self.sequence.append(statement)

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        for statement in self.sequence:
            yield from statement.run(scope)

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is Sequence:
            return self.sequence == other.sequence
        return NotImplemented

    def __repr__(self):
        return f"Sequence({self.sequence})"


class Nop(Statement):
    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        yield from ()

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

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        val = (
            scope.resolve_variable(self.val) if isinstance(self.val, str) else self.val
        )
        yield Notification(
            statement_type=StatementType.Set, args=dict(path=self.path, value=val)
        )

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecSet:
            return (self.path, self.val) == (other.path, other.val)
        return NotImplemented

    def __repr__(self):
        return f"ExecSet({repr(self.path)}, {repr(self.val)})"


class ExecNeartimeCall(Statement):
    def __init__(self, func_name: str, args: dict[str, Any]):
        self.func_name = func_name
        self.args = args

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        resolved_args = {}
        for name, val in self.args.items():
            if isinstance(val, UIDReference):
                resolved_args[name] = scope.resolve_variable(val.uid)
            else:
                resolved_args[name] = val
        yield Notification(
            statement_type=StatementType.NTCallback,
            args=dict(func_name=self.func_name, args=resolved_args),
        )

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is ExecNeartimeCall:
            return (self.func_name, self.args) == (other.func_name, other.args)
        return NotImplemented

    def __repr__(self):
        return f"ExecNeartimeCall({repr(self.func_name)}, {repr(self.args)})"


class ExecAcquire(Statement):
    def __init__(self, handle: str, signal: str, parent_uid: str):
        self.handle = handle
        self.signal = min(signal) if isinstance(signal, list) else signal
        self.parent_uid = parent_uid

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        yield Notification(
            statement_type=StatementType.Acquire,
            args=dict(
                handle=self.handle, signal=self.signal, parent_uid=self.parent_uid
            ),
        )

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
    def __init__(
        self, name: str, start: float, step: float, axis_name: str | None = None
    ):
        self.name = name
        self.start = start
        self.step = step
        self.axis_name = axis_name

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        index = scope.resolve_variable(LOOP_INDEX)
        value = self.start + self.step * index
        scope.set_variable(self.name, value)
        yield Notification(
            statement_type=StatementType.SetSwParam,
            args=dict(
                name=self.name,
                index=index,
                value=value,
                axis_name=self.axis_name,
                values=None,
            ),
        )

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
    def __init__(self, name: str, values: npt.ArrayLike, axis_name: str | None = None):
        self.name = name
        self.values = values
        self.axis_name = axis_name

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        index = max(0, min(scope.resolve_variable(LOOP_INDEX), len(self.values) - 1))
        value = self.values[index]
        scope.set_variable(self.name, value)
        yield Notification(
            statement_type=StatementType.SetSwParam,
            args=dict(
                name=self.name,
                index=index,
                value=value,
                axis_name=self.axis_name,
                values=self.values,
            ),
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

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        sub_scope = scope.make_sub_scope()
        for i in self._loop_iterator(scope):
            yield Notification(
                statement_type=StatementType.ForLoopEntry,
                args=dict(count=self.count, index=i, loop_flags=self.loop_flags),
            )
            sub_scope.set_variable(LOOP_INDEX, i)
            yield from self.body.run(sub_scope)
            yield Notification(
                statement_type=StatementType.ForLoopExit,
                args=dict(count=self.count, index=i, loop_flags=self.loop_flags),
            )

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

    def run(self, scope: ExecutionScope) -> Iterator[Notification]:
        yield Notification(
            statement_type=StatementType.RTEntry,
            args=dict(
                count=self.count,
                uid=self.uid,
                averaging_mode=self.averaging_mode,
                acquisition_type=self.acquisition_type,
            ),
        )
        if scope.root.looping_mode == LoopingMode.NEAR_TIME_ONLY:
            pass
        elif scope.root.looping_mode == LoopingMode.ONCE:
            sub_scope = scope.make_sub_scope()
            sub_scope.enter_real_time()
            yield from super().run(sub_scope)
        else:
            raise LabOneQException(f"Unknown looping mode '{scope.root.looping_mode}'")
        yield Notification(
            statement_type=StatementType.RTExit,
            args=dict(
                count=self.count,
                uid=self.uid,
                averaging_mode=self.averaging_mode,
                acquisition_type=self.acquisition_type,
            ),
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
        self._handlers_map = {
            StatementType.Set: self.set_handler,
            StatementType.NTCallback: self.nt_callback_handler,
            StatementType.Acquire: self.acquire_handler,
            StatementType.SetSwParam: self.set_sw_param_handler,
            StatementType.ForLoopEntry: self.for_loop_entry_handler,
            StatementType.ForLoopExit: self.for_loop_exit_handler,
            StatementType.RTEntry: self.rt_entry_handler,
            StatementType.RTExit: self.rt_exit_handler,
        }

    def set_handler(self, path: str, value):
        pass

    def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        pass

    def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        pass

    def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: npt.ArrayLike
    ):
        pass

    def for_loop_entry_handler(self, count: int, index: int, loop_flags: LoopFlags):
        pass

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        pass

    def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        pass

    def rt_exit_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        pass

    def run(self, root_sequence: Statement):
        """Start execution of the provided sequence."""
        scope = ExecutionScope(None, self)
        for notification in root_sequence.run(scope):
            self._handlers_map[notification.statement_type](**notification.args)


class AsyncExecutorBase:
    """Async version of ExecutorBase. See ExecutorBase for details."""

    def __init__(self, looping_mode: LoopingMode = LoopingMode.NEAR_TIME_ONLY):
        self.looping_mode: LoopingMode = looping_mode
        self._handlers_map = {
            StatementType.Set: self.set_handler,
            StatementType.NTCallback: self.nt_callback_handler,
            StatementType.Acquire: self.acquire_handler,
            StatementType.SetSwParam: self.set_sw_param_handler,
            StatementType.ForLoopEntry: self.for_loop_entry_handler,
            StatementType.ForLoopExit: self.for_loop_exit_handler,
            StatementType.RTEntry: self.rt_entry_handler,
            StatementType.RTExit: self.rt_exit_handler,
        }

    async def set_handler(self, path: str, value):
        pass

    async def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        pass

    async def acquire_handler(self, handle: str, signal: str, parent_uid: str):
        pass

    async def set_sw_param_handler(
        self, name: str, index: int, value: float, axis_name: str, values: npt.ArrayLike
    ):
        pass

    async def for_loop_entry_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        pass

    async def for_loop_exit_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        pass

    async def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        pass

    async def rt_exit_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        pass

    async def run(self, root_sequence: Statement):
        """Start execution of the provided sequence."""
        scope = ExecutionScope(None, self)
        for notification in root_sequence.run(scope):
            await self._handlers_map[notification.statement_type](**notification.args)


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
