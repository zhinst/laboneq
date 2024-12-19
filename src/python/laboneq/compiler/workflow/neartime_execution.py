# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import logging
from builtins import frozenset
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
import time

from laboneq.compiler import CompilerSettings
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.workflow import rt_linker
from laboneq.compiler.common.iface_compiler_output import RTCompilerOutputContainer
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompiler
from laboneq.compiler.workflow.rt_linker import CombinedRTCompilerOutputContainer
from laboneq.executor.executor import (
    ExecRT,
    ExecutorBase,
    LoopFlags,
    LoopingMode,
    Sequence,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray

_logger = logging.getLogger(__name__)


@dataclass
class IterationStep:
    # The number of iterations in this loop
    count: int

    # The index of this iteration (aka the iteration in this loop)
    index: int

    # The values of the near-time parameters for this iteration, not including
    # parameters from the parent loop
    parameter_values: Dict[str, Any]


@dataclass
class IterationStack:
    _stack: List[IterationStep] = field(default_factory=list)

    def push(self, count: int, index: int, parameter_values: Dict[str, Any]):
        self._stack.append(IterationStep(count, index, parameter_values))

    def pop(self):
        return self._stack.pop()

    def nt_loop_indices(self):
        return tuple(step.index for step in self._stack)

    def nt_parameter_values(self):
        return {k: v for step in self._stack for k, v in step.parameter_values.items()}

    def set_parameter_value(self, name: str, value: Any):
        self._stack[-1].parameter_values[name] = value

    def current_index_flat(self):
        index_flat = 0
        for step in self._stack:
            index_flat *= step.count
            index_flat += step.index
        return index_flat

    def total_count(self):
        count = 1
        for step in self._stack:
            count *= step.count
        return count


def legacy_execution_program():
    """Near-time seqc reloading not supported in JSON mode.

    Use dummy execution to emulate legacy behaviour."""

    # `None` as placeholder is acceptable here. Currently the executor requires none of
    # these.
    return ExecRT(
        count=1, body=Sequence(), uid="", acquisition_type=None, averaging_mode=None
    )


class NtCompilerExecutorDelegate(abc.ABC):
    @abc.abstractmethod
    def __init__(self, settings: CompilerSettings):
        raise NotImplementedError

    @abc.abstractmethod
    def after_compilation_run(self, new: RTCompilerOutputContainer, indices: list[int]):
        raise NotImplementedError

    @abc.abstractmethod
    def after_final_run(self, combined: CombinedRTCompilerOutputContainer):
        raise NotImplementedError


class NtCompilerExecutor(ExecutorBase):
    _delegates_types: list[type[NtCompilerExecutorDelegate]] = []

    def __init__(self, rt_compiler: RealtimeCompiler, settings: CompilerSettings):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self._rt_compiler = rt_compiler
        self._settings = settings
        self._iteration_stack = IterationStack()

        self._compiler_output_by_param_values: Dict[
            frozenset, RTCompilerOutputContainer
        ] = {}
        self._last_compiler_output: Optional[RTCompilerOutputContainer] = None
        self._required_parameters: Optional[Set[str]] = None
        self._combined_compiler_output: Optional[CombinedRTCompilerOutputContainer] = (
            None
        )

        self._delegates = [
            Delegate(self._settings) for Delegate in self._delegates_types
        ]

        self._skipped_last_compilation = False

    @classmethod
    def register_hook(cls, delegate_class: type[NtCompilerExecutorDelegate]):
        cls._delegates_types.append(delegate_class)

    def set_sw_param_handler(
        self,
        name: str,
        index: int,
        value: float,
        axis_name: str,
        values: NumPyArray,
    ):
        super().set_sw_param_handler(name, index, value, axis_name, values)
        self._iteration_stack.set_parameter_value(name, value)

    def for_loop_entry_handler(self, count: int, index: int, loop_flags: LoopFlags):
        self._iteration_stack.push(count, index, {})
        if loop_flags.is_pipeline:
            self._iteration_stack.set_parameter_value("__pipeline_index", index)

    def for_loop_exit_handler(self, count: int, index: int, loop_flags: LoopFlags):
        self._iteration_stack.pop()

    def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode,
        acquisition_type,
    ):
        time_start = time.perf_counter()
        if self._required_parameters is not None:
            # We already know what subset of the near-time parameters are required
            # by the real-time sequence. If we already have a compiler output for
            # that state, we can skip the compilation.
            requested_values = self._frozen_required_parameters()
            if requested_values in self._compiler_output_by_param_values:
                new_compiler_output = self._compiler_output_by_param_values[
                    requested_values
                ]

                self._last_compiler_output = new_compiler_output
                rt_linker.repeat_previous(
                    self._combined_compiler_output, self._last_compiler_output
                )
                if not self._skipped_last_compilation:
                    _logger.info("Skipping compilation for next step(s)...")
                self._skipped_last_compilation = True

                return

        # We don't have a compiler output for this state yet, so we need to compile
        parameter_store = ParameterStore(self._iteration_stack.nt_parameter_values())
        tracker = parameter_store.create_tracker()
        new_compiler_output = self._rt_compiler.run(parameter_store)

        if self._required_parameters is None:
            self._required_parameters = tracker.queries()
        else:
            assert self._required_parameters == tracker.queries()

        requested_values = self._frozen_required_parameters()

        self._compiler_output_by_param_values[requested_values] = new_compiler_output

        nt_step_indices = list(self._iteration_stack.nt_loop_indices())

        # Assemble the combined compiler output
        if self._combined_compiler_output is None:
            self._combined_compiler_output = rt_linker.from_single_run(
                new_compiler_output,
                nt_step_indices,
            )
        else:
            rt_linker.merge_compiler_runs(
                self._combined_compiler_output,
                new_compiler_output,
                self._last_compiler_output,
                nt_step_indices,
            )

        for delegate in self._delegates:
            delegate.after_compilation_run(new_compiler_output, nt_step_indices)

        self._last_compiler_output = new_compiler_output

        time_delta = time.perf_counter() - time_start

        this_index = self._iteration_stack.current_index_flat()
        total_count = self._iteration_stack.total_count()
        self._skipped_last_compilation = False
        _logger.info(
            f"Completed compilation step {this_index + 1} of {total_count}. [{time_delta:.3f} s]"
        )

    def _frozen_required_parameters(self):
        return frozenset(
            (k, v)
            for k, v in self._iteration_stack.nt_parameter_values().items()
            if k in self._required_parameters
        )

    def combined_compiler_output(self):
        return self._combined_compiler_output

    def finalize(self):
        if self._combined_compiler_output is not None:
            for delegate in self._delegates:
                delegate.after_final_run(self._combined_compiler_output)
