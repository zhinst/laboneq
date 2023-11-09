# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from laboneq.core.exceptions import LabOneQException
from laboneq.executor import executor

if TYPE_CHECKING:
    from laboneq.dsl import SweepParameter
    from laboneq.dsl.experiment import Experiment, Operation, Section, Sweep


class ExecutionFactoryFromExperiment(executor.ExecutionFactory):
    def __init__(self):
        super().__init__()
        self._chunked_sweep = None

    def make(self, experiment: Experiment) -> executor.Statement:
        self._chunked_sweep = self.analyze_pipeline(experiment)
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
                    executor.ForLoop(
                        count=child.count,
                        body=loop_body,
                        loop_flags=executor.LoopFlags.AVERAGE,
                    )
                )
            elif isinstance(child, AcquireLoopRt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                loop = executor.ExecRT(
                    count=child.count,
                    body=loop_body,
                    uid=child.uid,
                    averaging_mode=child.averaging_mode,
                    acquisition_type=child.acquisition_type,
                )
                if self._chunked_sweep is not None:
                    # add 'fake' pipeline loop
                    loop = self._make_pipelined(loop)
                self._append_statement(loop)
            elif isinstance(child, Sweep):
                count = len(child.parameters[0].values)
                loop_body = self._sub_scope(self._handle_sweep, child)
                self._append_statement(executor.ForLoop(count=count, body=loop_body))
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
        from laboneq.dsl.experiment import (
            Acquire,
            Call,
            Delay,
            PlayPulse,
            Reserve,
            SetNode,
        )

        if isinstance(operation, Call):
            return executor.ExecNeartimeCall(operation.func_name, operation.args)
        if isinstance(operation, SetNode):
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

    def _make_pipelined(self, averaging_loop: executor.Statement):
        return executor.ForLoop(
            count=self._chunked_sweep.chunk_count,
            body=averaging_loop,
            loop_flags=executor.LoopFlags.PIPELINE,
        )

    @staticmethod
    def analyze_pipeline(experiment: Experiment):
        from laboneq.dsl.experiment import AcquireLoopRt, Section, Sweep

        rt_averaging_loop = None
        chunked_sweep = None

        def visit(section: Section, inside_rt=False):
            nonlocal rt_averaging_loop, chunked_sweep
            if isinstance(section, AcquireLoopRt):
                if rt_averaging_loop is not None:
                    raise LabOneQException("Found multiple RT averaging loops")
                rt_averaging_loop = section
                inside_rt = True
            if isinstance(section, Sweep):
                if section.chunk_count > 1:
                    if chunked_sweep is not None:
                        raise LabOneQException("Found multiple chunked sweeps")
                    if not inside_rt:
                        raise LabOneQException(
                            "Chunking of sweeps is only supported for real-time execution"
                        )
                    chunked_sweep = section
            for child in section.children:
                if isinstance(child, Section):
                    visit(child, inside_rt)

        for c in experiment.sections:
            # depth-first search
            visit(c)

        return chunked_sweep
