# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.prng import PRNG
from laboneq.data.experiment_description import (
    Acquire,
    AcquireLoopNt,
    AcquireLoopRt,
    Call,
    Delay,
    Experiment,
    Operation,
    PlayPulse,
    Reserve,
    Section,
    SetNode,
    Sweep,
    PrngLoop,
)
from laboneq.data.parameter import LinearSweepParameter, SweepParameter
from laboneq.data.prng import PRNGSample
from laboneq.executor import executor


class ExecutionFactoryFromNewExperiment(executor.ExecutionFactory):
    def __init__(self):
        super().__init__()
        self._chunked_sweep = None

    def make(self, experiment: Experiment) -> executor.Statement:
        self._chunked_sweep = self.analyze_pipeline(experiment)
        self._handle_children(experiment.sections, experiment.uid)
        return self._root_sequence

    def _handle_children(self, children: list[Operation | Section], parent_uid: str):
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
                parameter = child.parameters[0]
                if isinstance(parameter, LinearSweepParameter):
                    count = parameter.count
                else:
                    assert isinstance(parameter, SweepParameter)
                    count = len(parameter.values)
                loop_body = self._sub_scope(self._handle_sweep, child)
                self._append_statement(executor.ForLoop(count=count, body=loop_body))
            elif isinstance(child, PrngLoop):
                prng_sample = child.prng_sample
                count = prng_sample.count
                loop_body = self._sub_scope(self._handle_prng_loop, child)
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

    def _handle_prng_loop(self, loop: PrngLoop):
        self._append_statement(self._statement_from_param(loop.prng_sample))
        self._handle_children(loop.children, loop.uid)

    def _statement_from_param(
        self, parameter: SweepParameter | LinearSweepParameter | PRNGSample
    ):
        if isinstance(parameter, SweepParameter):
            values = parameter.values
            axis_name = parameter.axis_name
        elif isinstance(parameter, LinearSweepParameter):
            values = np.linspace(parameter.start, parameter.stop, parameter.count)
            axis_name = parameter.axis_name
        elif isinstance(parameter, PRNGSample):
            prng = parameter.prng
            prng_sim = PRNG(seed=prng.seed, upper=prng.range - 1)
            values = np.array([next(prng_sim) for _ in range(parameter.count)])
            axis_name = parameter.uid
        else:
            raise TypeError(f"Unrecognized parameter type: {type(parameter)}")
        return executor.SetSoftwareParam(
            name=parameter.uid, values=values, axis_name=axis_name
        )

    def _statement_from_operation(self, operation, parent_uid: str):
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
    def analyze_pipeline(experiment: Experiment) -> Section | None:
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
