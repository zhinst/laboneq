# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from laboneq.core.utilities.prng import PRNG
from laboneq.data.experiment_description import (
    Acquire,
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


class ExecutionFactoryFromExperiment(executor.ExecutionFactory):
    def __init__(self):
        super().__init__()
        self._chunk_count: int = 1

    def make(self, experiment: Experiment, chunk_count: int = 1) -> executor.Statement:
        self._chunk_count = chunk_count
        self._handle_children(experiment.sections, experiment.uid)
        return self._root_sequence

    def _handle_children(self, children: list[Operation | Section], parent_uid: str):
        for child in children:
            if isinstance(child, Operation):
                self._append_statement(
                    self._statement_from_operation(child, parent_uid)
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
            assert operation.signal is not None
            return executor.ExecAcquire(operation.handle, operation.signal, parent_uid)
        return executor.Nop()
