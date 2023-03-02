# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from laboneq.core.types.enums import ExecutionType
from laboneq.executor import executor

if TYPE_CHECKING:
    from laboneq.dsl import SweepParameter
    from laboneq.dsl.experiment import Experiment, Operation, Section, Sweep


class ExecutionFactoryFromExperiment(executor.ExecutionFactory):
    def make(self, experiment: Experiment) -> executor.Statement:
        self._handle_children(experiment.sections, experiment.uid)
        return self._root_sequence

    def _handle_children(
        self, children: List[Union[Operation, Section]], parent_uid: str
    ):
        from laboneq.dsl.experiment import (
            AcquireLoopNt,
            AcquireLoopRt,
            Operation,
            Sweep,
        )

        for child in children:
            if isinstance(child, Operation):
                self._append_statement(
                    self._statement_from_operation(child, parent_uid)
                )
            elif isinstance(child, AcquireLoopNt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    executor.ForLoop(child.count, loop_body, executor.LoopType.AVERAGE)
                )
            elif isinstance(child, AcquireLoopRt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    executor.ExecRT(
                        count=child.count,
                        body=loop_body,
                        uid=child.uid,
                        averaging_mode=child.averaging_mode,
                        acquisition_type=child.acquisition_type,
                    )
                )
            elif isinstance(child, Sweep):
                count = len(child.parameters[0].values)
                loop_body = self._sub_scope(self._handle_sweep, child)
                loop_type = (
                    executor.LoopType.HARDWARE
                    if child.execution_type == ExecutionType.REAL_TIME
                    else executor.LoopType.SWEEP
                )
                self._append_statement(executor.ForLoop(count, loop_body, loop_type))
            else:
                sub_sequence = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(sub_sequence)

    def _handle_sweep(self, sweep: Sweep):
        for parameter in sweep.parameters:
            self._append_statement(self._statement_from_param(parameter))
        self._handle_children(sweep.children, sweep.uid)

    def _statement_from_param(self, parameter: SweepParameter):
        return executor.SetSoftwareParam(
            parameter.uid, parameter.values, parameter.axis_name
        )

    def _statement_from_operation(self, operation, parent_uid: str):
        from laboneq.dsl.experiment import Acquire, Call, Delay, PlayPulse, Reserve, Set

        if isinstance(operation, Call):
            return executor.ExecUserCall(operation.func_name, operation.args)
        if isinstance(operation, Set):
            return executor.ExecSet(operation.path, operation.value)
        if isinstance(operation, PlayPulse):
            return executor.Nop()
        if isinstance(operation, Delay):
            return executor.Nop()
        if isinstance(operation, Reserve):
            return executor.Nop()
        if isinstance(operation, Acquire):
            return executor.ExecAcquire(operation.handle, operation.signal, parent_uid)
        return executor.Nop()
