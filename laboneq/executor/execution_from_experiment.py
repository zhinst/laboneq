# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import List, Union, TYPE_CHECKING

from .executor import *

from laboneq.core.types.enums import ExecutionType

if TYPE_CHECKING:
    from laboneq.dsl.experiment import Experiment, Operation, Section, Sweep
    from laboneq.dsl import SweepParameter


class ExecutionFactoryFromExperiment(ExecutionFactory):
    def make(self, experiment: Experiment) -> Statement:
        self._handle_children(experiment.sections, experiment.uid)
        return self._root_sequence

    def _handle_children(
        self, children: List[Union[Operation, Section]], parent_uid: str
    ):
        from laboneq.dsl.experiment import (
            Operation,
            AcquireLoopNt,
            AcquireLoopRt,
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
                    ForLoop(child.count, loop_body, LoopType.AVERAGE)
                )
            elif isinstance(child, AcquireLoopRt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    ExecRT(
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
                    LoopType.HARDWARE
                    if child.execution_type == ExecutionType.REAL_TIME
                    else LoopType.SWEEP
                )
                self._append_statement(ForLoop(count, loop_body, loop_type))
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
        return SetSoftwareParam(parameter.uid, parameter.values, parameter.axis_name)

    def _statement_from_operation(self, operation, parent_uid: str):
        from laboneq.dsl.experiment import (
            Call,
            Set,
            PlayPulse,
            Delay,
            Reserve,
            Acquire,
        )

        if isinstance(operation, Call):
            return ExecUserCall(operation.func_name, operation.args)
        if isinstance(operation, Set):
            return ExecSet(operation.path, operation.value)
        if isinstance(operation, PlayPulse):
            return Nop()
        if isinstance(operation, Delay):
            return Nop()
        if isinstance(operation, Reserve):
            return Nop()
        if isinstance(operation, Acquire):
            return ExecAcquire(operation.handle, operation.signal, parent_uid)
        return Nop()
