# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Block for workflows."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Callable

from laboneq.workflow.timestamps import utc_now
from laboneq.workflow import variable_tracker
from laboneq.workflow import blocks
from laboneq.workflow.executor import ExecutionStatus, ExecutorState
from laboneq.workflow.opts import WorkflowOptions, get_and_validate_param_type
from laboneq.workflow.reference import Reference, notset
from laboneq.workflow.result import WorkflowResult

if TYPE_CHECKING:
    from laboneq.workflow.opts import TaskOptions


class WorkflowBlock(blocks.Block):
    """Workflow block."""

    def __init__(
        self,
        name: str,
        options_type: type[WorkflowOptions] | None = WorkflowOptions,
        parameters: dict | None = None,
    ) -> None:
        self._name = name
        self._options_type = options_type or WorkflowOptions
        params = {}
        for param, default in (parameters or {}).items():
            if isinstance(default, Reference):
                params[param] = default
            else:
                params[param] = Reference((self, param), default=default)
        super().__init__(parameters=params)
        self._ref = Reference(self)

    @property
    def name(self) -> str:
        """Name of the block."""
        return self._name

    @property
    def options_type(self) -> type[WorkflowOptions]:
        """Type of workflow options."""
        return self._options_type

    def create_options(self) -> WorkflowOptions:
        """Create options for the block.

        The method goes over the sub-blocks and finds which blocks has
            options available.

        Same task option instance is shared across the same named task executions
        per workflow, therefore the same task calls within a single workflow block
        cannot have multiple option definitions.

        Returns:
            Workflow options where `tasks` is populated with the default options
                of sub-blocks.
        """

        def get_options(
            block: blocks.Block, opts: dict
        ) -> WorkflowOptions | TaskOptions | None:
            if isinstance(block, WorkflowBlock):
                return block.create_options()
            if isinstance(block, blocks.TaskBlock) and block.options_type is not None:
                return block.options_type()
            for x in block.body:
                maybe_opts = get_options(x, opts)
                if maybe_opts:
                    opts[x.name] = maybe_opts
            return None

        tasks = {}
        for x in self.body:
            maybe_opts = get_options(x, tasks)
            if maybe_opts:
                tasks[x.name] = maybe_opts
        ret_opt = self.options_type()
        ret_opt._task_options = tasks
        return ret_opt

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    def set_params(self, executor: ExecutorState, **kwargs: object) -> None:
        """Set the initial parameters of the block.

        Arguments:
            executor: Active executor.
            **kwargs: Input parameters of the block.
        """
        inputs = kwargs.copy()
        input_opts = inputs.get("options")  # Options from input arguments
        if isinstance(input_opts, dict):
            input_opts = self.options_type.from_dict(input_opts)
        elif input_opts is None:
            # Options from parent options
            input_opts = executor.get_options(self.name)
        if input_opts is None:
            # Default options
            input_opts = self.options_type()
        inputs["options"] = input_opts
        for k, v in inputs.items():
            if k in self.parameters:
                executor.set_variable(self.parameters[k], v)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        # TODO: Separate executor results and WorkflowResult
        if executor.get_block_status(self) == ExecutionStatus.NOT_STARTED:
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            inputs = executor.resolve_inputs(self)
            self.set_params(executor, **inputs)
            input_opts = (
                executor.get_variable(self.parameters["options"])
                if "options" in self.parameters
                else self.options_type()
            )
            # NOTE: Correct input options are resolved only after 'set_param()'
            # Therefore they need to be overwritten for result
            inputs["options"] = input_opts
            result = WorkflowResult(
                name=self.name, input=inputs, index=executor.get_index()
            )
            result._start_time = utc_now()
            executor.set_variable(self.ref, result)
            executor.add_workflow_result(result)
            executor.recorder.on_start(result)
        elif executor.get_block_status(self) == ExecutionStatus.IN_PROGRESS:
            result = executor.get_variable(self.ref)
            input_opts = (
                executor.get_variable(self.parameters["options"])
                if "options" in self.parameters
                else self.options_type()
            )
        else:
            # Block is finished
            return
        try:
            body_loop_finished = False
            block = None
            with executor.enter_workflow(result, input_opts):
                for block in self.body:
                    if executor.get_block_status(block) in (
                        ExecutionStatus.FINISHED,
                        ExecutionStatus.SKIPPED,
                    ):
                        continue
                    block.execute(executor)
                body_loop_finished = True
                executor.set_block_status(self, ExecutionStatus.FINISHED)
                result._end_time = utc_now()
                executor.recorder.on_end(result)
            # TODO: Better solution if loop cancelled from return statement
            if isinstance(block, blocks.ReturnStatement) and not body_loop_finished:
                executor.set_block_status(self, ExecutionStatus.FINISHED)
                result._end_time = utc_now()
                executor.recorder.on_end(result)
            if executor.settings.run_until == self.name and executor.has_active_context:
                executor.interrupt()
        except Exception as error:
            executor.set_block_status(self, ExecutionStatus.FINISHED)
            result._end_time = utc_now()
            executor.recorder.on_error(result, error)
            executor.recorder.on_end(result)
            raise

    @classmethod
    def from_callable(
        cls, name: str, func: Callable, **kwargs: object
    ) -> WorkflowBlock:
        """Create the block from a callable.

        By default the signature of the function is used to define
        the default parameters of the block.

        Arguments:
            name: Name of the block.
            func: A function defining the workflow
            **kwargs: Default parameter values that overwrite function
                signature
        """
        params = {}
        for k, v in signature(func).parameters.items():
            if k in kwargs:
                value = kwargs[k]
            elif v.default == v.empty:
                value = notset
            else:
                value = v.default
            params[k] = value

        opt_type_hint = None
        if "options" in params:
            opt_type_hint = get_and_validate_param_type(
                func, WorkflowOptions, "options"
            )
        obj = cls(name, opt_type_hint, params)
        with variable_tracker.WorkflowFunctionVariableTrackerContext.scoped(
            variable_tracker.WorkflowFunctionVariableTracker()
        ):
            with obj:
                func(**obj.parameters)
        return obj

    def __str__(self):
        return f"workflow(name={self.name})"
