# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import traceback
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from weakref import ReferenceType


from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.recipe_processor import RecipeData
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import NtStepKey
from laboneq.executor.executor import AsyncExecutorBase, LoopFlags, LoopingMode

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.controller import Controller, ExecutionContext

_logger = logging.getLogger(__name__)


_SessionClass = TypeVar("_SessionClass")


class NearTimeRunner(AsyncExecutorBase, Generic[_SessionClass]):
    def __init__(
        self,
        controller: Controller,
        parent_session_ref: ReferenceType[_SessionClass],
        execution_context: ExecutionContext,
        recipe_data: RecipeData,
    ):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self.controller = controller
        self.parent_session_ref = parent_session_ref
        self.execution_context = execution_context
        self.recipe_data = recipe_data
        self.user_set_nodes = NodeCollector()
        self.nt_loop_indices: list[int] = []
        self.sweep_params_tracker = SweepParamsTracker()
        self.last_nt_step_result_completed: asyncio.Future[None] | None = None

    def nt_step(self) -> NtStepKey:
        return NtStepKey(indices=tuple(self.nt_loop_indices))

    async def set_handler(self, path: str, value):
        self.user_set_nodes.add(path, value, cache=False)

    async def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        if (
            self.last_nt_step_result_completed is not None
            and not self.last_nt_step_result_completed.done()
        ):
            # This is necessary to ensure that the existing logic for partial results
            # is not broken by the new asynchronous execution model.
            await self.last_nt_step_result_completed
        func = self.controller._neartime_callbacks.get(func_name)
        if func is None:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' is not registered."
            )
        parent_session = self.parent_session_ref()
        if parent_session is None:
            raise LabOneQControllerException(
                "Internal error: Originating session has been destroyed."
            )
        protected_session = ProtectedSession(
            wrapped_session=parent_session,
            controller=self.controller,
            recipe_data=self.recipe_data,
            experiment_results=self.execution_context.submission.results,
        )
        try:
            if iscoroutinefunction(func):
                res = await func(protected_session, **args)
            else:
                res = func(protected_session, **args)
        except AbortExecution:
            _logger.warning(f"Execution aborted by near-time callback '{func_name}'")
            raise
        except BaseException as e:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' failed with: {e}"
            ) from e
        neartime_callback_results = self.execution_context.submission.results.neartime_callback_results.setdefault(
            func_name, []
        )
        neartime_callback_results.append(res)

    async def set_sw_param_handler(
        self,
        name: str,
        index: int,
        value: float,
        axis_name: str,
        values: NumPyArray,
    ):
        self.sweep_params_tracker.set_param(name, value)

    async def for_loop_entry_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        self.nt_loop_indices.append(index)

    async def for_loop_exit_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        self.nt_loop_indices.pop()

    async def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        await self.controller._prepare_nt_step(
            recipe_data=self.recipe_data,
            sweep_params_tracker=self.sweep_params_tracker,
            user_set_nodes=self.user_set_nodes,
            nt_step=self.nt_step(),
        )
        self.sweep_params_tracker.clear_for_next_step()
        self.user_set_nodes = NodeCollector()

        try:
            self.last_nt_step_result_completed = (
                await self.controller._execute_one_step(
                    execution_context=self.execution_context,
                    recipe_data=self.recipe_data,
                    nt_step=self.nt_step(),
                )
            )
        except LabOneQControllerException:
            # TODO(2K): introduce "hard" controller exceptions
            self.controller._report_step_error(
                results=self.execution_context.submission.results,
                nt_step=self.nt_step(),
                uid=uid,
                message=traceback.format_exc(),
            )

        await self.controller._after_nt_step()
