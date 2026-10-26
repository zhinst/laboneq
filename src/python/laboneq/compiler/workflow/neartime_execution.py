# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from laboneq.compiler.workflow import rt_linker
from laboneq.compiler.workflow.realtime_compiler import compile_realtime
from laboneq.executor.executor import (
    ExecutorBase,
    LoopingMode,
)

from .parameter_store import ParameterStore

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_compiler_output import RTCompilerOutputContainer
    from laboneq.compiler.workflow.rt_linker import CombinedRTCompilerOutputContainer
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
    parameter_values: dict[str, Any]


@dataclass
class IterationStack:
    _stack: list[IterationStep] = field(default_factory=list)

    def push(self, count: int, index: int, parameter_values: dict[str, Any]):
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


class NtCompilerExecutor(ExecutorBase):
    def __init__(
        self,
        experiment: compiler_rs.ProcessedExperiment,
        chunk_count: int | None = None,
        backend: str = "qccs",
    ):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self._experiment = experiment
        self._chunk_count = chunk_count
        self._backend = backend
        self._iteration_stack = IterationStack()

        # Compiler outputs, grouped by the near-time parameters their compilation
        # read, then keyed by the values of those parameters
        self._compiler_outputs: dict[
            frozenset[str],
            dict[frozenset[tuple[str, Any]], RTCompilerOutputContainer],
        ] = {}
        self._last_compiler_output: RTCompilerOutputContainer | None = None
        self._combined_compiler_output: CombinedRTCompilerOutputContainer | None = None

        self._skipped_last_compilation = False

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

    def for_loop_entry_handler(self, count: int, index: int):
        self._iteration_stack.push(count, index, {})

    def for_loop_exit_handler(self, count: int, index: int):
        self._iteration_stack.pop()

    def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode,
        acquisition_type,
    ):
        if self._chunk_count is None:
            self._rt_entry_handler()
        else:
            for i in range(self._chunk_count):
                self._iteration_stack.push(self._chunk_count, i, {})
                self._iteration_stack.set_parameter_value("__chunk_index", i)
                self._iteration_stack.set_parameter_value(
                    "__chunk_count", self._chunk_count
                )
                self._rt_entry_handler()
                self._iteration_stack.pop()

    def _rt_entry_handler(
        self,
    ):
        time_start = time.perf_counter()
        nt_step_indices = list(self._iteration_stack.nt_loop_indices())
        nt_parameter_values = self._iteration_stack.nt_parameter_values()
        cached_compiler_output = self._cached_compiler_output(nt_parameter_values)
        if cached_compiler_output is not None:
            self._last_compiler_output = cached_compiler_output
            rt_linker.repeat_run(
                self._combined_compiler_output,
                self._last_compiler_output,
                nt_step_indices,
            )
            if not self._skipped_last_compilation:
                _logger.info("Skipping compilation for next step(s)...")
            self._skipped_last_compilation = True

            return

        # We don't have a compiler output for this state yet, so we need to compile
        parameter_store = ParameterStore[str, float](nt_parameter_values)
        tracker = parameter_store.create_tracker()
        new_compiler_output = compile_realtime(
            experiment=self._experiment,
            near_time_parameters=parameter_store,
            backend=self._backend,
        )

        used_parameters = frozenset(tracker.queries())
        self._compiler_outputs.setdefault(used_parameters, {})[
            _cache_key(nt_parameter_values, used_parameters)
        ] = new_compiler_output

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

        self._last_compiler_output = new_compiler_output

        time_delta = time.perf_counter() - time_start

        this_index = self._iteration_stack.current_index_flat()
        total_count = self._iteration_stack.total_count()
        self._skipped_last_compilation = False
        _logger.info(
            f"Completed compilation step {this_index + 1} of {total_count}. [{time_delta:.3f} s]"
        )

    def _cached_compiler_output(
        self, nt_parameter_values: dict[str, Any]
    ) -> RTCompilerOutputContainer | None:
        for used_parameters, compiler_outputs in self._compiler_outputs.items():
            compiler_output = compiler_outputs.get(
                _cache_key(nt_parameter_values, used_parameters)
            )
            if compiler_output is not None:
                return compiler_output
        return None

    def combined_compiler_output(self):
        return self._combined_compiler_output


def _cache_key(
    values: dict[str, Any], names: frozenset[str]
) -> frozenset[tuple[str, Any]]:
    return frozenset((k, v) for k, v in values.items() if k in names)
