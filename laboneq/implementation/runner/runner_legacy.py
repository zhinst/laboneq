# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from laboneq import controller as ctrl
from laboneq.data.execution_payload import (
    ExecutionPayload,
    LoopType,
    NearTimeOperation,
    NearTimeOperationType,
    NearTimeProgram,
    TargetSetup,
)
from laboneq.data.experiment_results import ExperimentResults
from laboneq.executor import executor
from laboneq.interfaces.runner.runner_api import RunnerAPI
from laboneq.interfaces.runner.runner_control_api import RunnerControlAPI

_logger = logging.getLogger(__name__)


class RunnerLegacy(RunnerAPI, RunnerControlAPI):
    """
    This the core implementation of the experiment runner.
    """

    def __init__(self):
        self._job_queue = []
        self._job_results = {}
        self._connected = False
        self._controller = None

    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        _logger.debug(f"Connecting to TargetSetup {setup.uid}")
        run_parameters = ctrl.ControllerRunParameters()
        run_parameters.dry_run = do_emulation
        run_parameters.ignore_version_mismatch = False

        controller = ctrl.Controller(
            run_parameters=run_parameters,
            target_setup=setup,
            user_functions={},
        )
        controller.connect()
        self._controller = controller
        self._connected = True

    def submit_execution_payload(self, job: ExecutionPayload):
        """
        Submit an experiment run job.
        """
        job_id = len(self._job_queue)
        queue_entry = {"job_id": job_id, "job": job}

        self._job_queue.append(queue_entry)
        if not self._connected:
            self.connect(job.target_setup)

        self._controller.execute_compiled(job)
        controller_results = self._controller._results
        self._job_results[job_id] = ExperimentResults(
            acquired_results=controller_results.acquired_results,
            user_func_results=controller_results.user_func_results,
            execution_errors=controller_results.execution_errors,
        )

        return job_id

    def run_job_status(self, job_id: str):
        """
        Get the status of an  experiment run job.
        """
        return next(j for j in self._job_queue if j["job_id"] == job_id)

    def run_job_result(self, job_id: str) -> ExperimentResults:
        """
        Get the result of an experiment run job. Blocks until the result is available.
        """
        num_tries = 10
        while True:
            result = self._job_results.get(job_id)
            if result:
                return result
            if num_tries == 0:
                break
            num_tries -= 1
            time.sleep(100e-3)

    def start(self):
        """
        Start the experiment runner. It will start processing jobs from the job queue.
        """
        pass

    def stop(self):
        """
        Stop the experiment runner. It will stop processing jobs from the job queue.
        """
        pass

    def disconnect(self):
        """
        Disconnect from the setup.
        """
        pass


class ExecutionFactoryFromNearTimeProgram(executor.ExecutionFactory):
    def make(self, near_time_program: NearTimeProgram) -> executor.Statement:
        self._handle_children(near_time_program.children, near_time_program.uid)
        return self._root_sequence

    @staticmethod
    def is_operation(op: NearTimeOperationType):
        return op not in (
            NearTimeOperationType.ACQUIRE_LOOP_NT,
            NearTimeOperationType.ACQUIRE_LOOP_RT,
            NearTimeOperationType.FOR_LOOP,
        )

    @staticmethod
    def convert_loop_type(loop_type: LoopType):
        return {
            LoopType.AVERAGE: executor.LoopType.AVERAGE,
            LoopType.SWEEP: executor.LoopType.SWEEP,
            LoopType.HARDWARE: executor.LoopType.HARDWARE,
        }[loop_type]

    def _handle_children(self, children, parent_uid: str):
        for child in children:
            if child.operation_type is None:
                body = self._sub_scope(self._handle_children, child.children, child.uid)
                sequence = executor.Sequence()
                sequence.append_statement(body)
            if self.is_operation(child.operation_type):
                self._append_statement(
                    self._statement_from_operation(child, parent_uid)
                )
            elif child.operation_type == NearTimeOperationType.FOR_LOOP:
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    executor.ForLoop(
                        child.args["count"],
                        loop_body,
                        self.convert_loop_type(child.args["loop_type"]),
                    )
                )
            elif child.operation_type == NearTimeOperationType.ACQUIRE_LOOP_NT:
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
            else:
                sub_sequence = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(sub_sequence)

    def _handle_sweep(self, sweep: NearTimeOperation):
        for parameter in sweep.args["parameters"]:
            self._append_statement(self._statement_from_param(parameter))
        self._handle_children(sweep.children, sweep.uid)

    def _statement_from_operation(self, operation, parent_uid: str):
        if operation.operation_type == NearTimeOperationType.CALL:
            return executor.ExecUserCall(
                operation.args["func_name"], operation.args["args"]
            )
        if operation.operation_type == NearTimeOperationType.SET:
            return executor.ExecSet(operation.args["path"], operation.args["value"])
        if operation.operation_type == NearTimeOperationType.PLAY_PULSE:
            return executor.Nop()
        if operation.operation_type == NearTimeOperationType.DELAY:
            return executor.Nop()
        if operation.operation_type == NearTimeOperationType.RESERVE:
            return executor.Nop()
        if operation.operation_type == NearTimeOperationType.ACQUIRE:
            return executor.ExecAcquire(operation.handle, operation.signal, parent_uid)

        return executor.Nop()


from laboneq.data.execution_payload import (
    ExecutionPayload,
    LoopType,
    NearTimeOperation,
    NearTimeOperationType,
    NearTimeProgram,
    TargetSetup,
)
from laboneq.data.execution_payload.execution_payload_helper import (
    ExecutionPayloadHelper,
)
from laboneq.executor import executor


def convert_loop_type(loop_type: LoopType):
    return {
        LoopType.AVERAGE: executor.LoopFlags.AVERAGE,
        LoopType.SWEEP: executor.LoopFlags.SWEEP,
        LoopType.HARDWARE: executor.LoopFlags.HARDWARE,
    }[loop_type]


def convert(near_time_program: NearTimeProgram):
    root_marker = "____ROOT___"
    context = {"nodes_by_parent": {}}

    def execution_builder_visitor(operation, context, parent):
        if parent is not None:
            parent_hash = id(parent)
        else:
            parent_hash = root_marker
        current_node_hash = id(operation)
        if parent_hash not in context["nodes_by_parent"]:
            context["nodes_by_parent"][parent_hash] = []

        _logger.debug(
            f"Visiting {operation}, context: {context}, current node hash: {current_node_hash}, parent hash: {parent_hash}"
        )
        if isinstance(operation, NearTimeProgram) or operation.operation_type is None:
            sequence = executor.Sequence()
            num_chidren = 0
            if current_node_hash in context["nodes_by_parent"]:
                num_chidren = len(context["nodes_by_parent"][current_node_hash])
                for child in context["nodes_by_parent"][current_node_hash]:
                    sequence.append_statement(child)

            _logger.debug(f"Appended {num_chidren} statements to sequence")
            context["nodes_by_parent"][parent_hash].append(sequence)

        elif operation.operation_type == NearTimeOperationType.PLAY_PULSE:
            context["nodes_by_parent"][parent_hash].append(executor.Nop())
        elif operation.operation_type == NearTimeOperationType.SET:
            context["nodes_by_parent"][parent_hash].append(
                executor.ExecSet(operation.args["path"], operation.args["value"])
            )

        elif operation.operation_type == NearTimeOperationType.SET_SOFTWARE_PARM:
            param_name = operation.args["parameter_uid"]
            values = operation.args["values"]
            axis_name = operation.args["axis_name"]
            context["nodes_by_parent"][parent_hash].append(
                executor.SetSoftwareParam(param_name, values, axis_name)
            )

        elif operation.operation_type == NearTimeOperationType.FOR_LOOP:
            loop_body = executor.Sequence()
            if current_node_hash in context["nodes_by_parent"]:
                for child in context["nodes_by_parent"][current_node_hash]:
                    loop_body.append_statement(child)
            loop = executor.ForLoop(
                operation.args["count"],
                loop_body,
                convert_loop_type(operation.args["loop_type"]),
            )
            context["nodes_by_parent"][parent_hash].append(loop)
        elif operation.operation_type == NearTimeOperationType.ACQUIRE_LOOP_RT:
            loop_body = executor.Sequence()
            if current_node_hash in context["nodes_by_parent"]:
                for child in context["nodes_by_parent"][current_node_hash]:
                    loop_body.append_statement(child)
            loop = executor.ExecRT(
                operation.args["count"],
                loop_body,
                operation.uid,
                averaging_mode=operation.args["averaging_mode"],
                acquisition_type=operation.args["acquisition_type"],
            )
            context["nodes_by_parent"][parent_hash].append(loop)

    ExecutionPayloadHelper.accept_near_time_program_visitor(
        near_time_program, execution_builder_visitor, context
    )
    _logger.debug(f"Context: {context}")
    return context["nodes_by_parent"][root_marker][0]
