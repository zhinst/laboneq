# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.runtime_context_impl import RuntimeContextImpl
from laboneq.controller.toolkit_adapter import ToolkitDevices
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import NtStepKey
from laboneq.executor.executor import AsyncExecutorBase, LoopFlags, LoopingMode

if TYPE_CHECKING:
    from laboneq.controller.controller import Controller, ExecutionContext
    from laboneq.controller.runtime_context_impl import LegacySessionData
    from laboneq.core.types.numpy_support import NumPyArray

_logger = logging.getLogger(__name__)


class NearTimeRunner(AsyncExecutorBase):
    def __init__(
        self,
        controller: Controller,
        execution_context: ExecutionContext,
        do_emulation: bool,
        legacy_session_data: LegacySessionData,
    ):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self.controller = controller
        self.do_emulation = do_emulation
        self.execution_context = execution_context
        self.user_set_nodes = NodeCollector()
        self.nt_loop_indices: list[int] = []
        self.sweep_params_tracker = SweepParamsTracker()
        self.last_nt_step_result_completed: asyncio.Future[None] | None = None
        self._legacy_session_data = legacy_session_data

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
        experiment_results = self.execution_context.submission.results_builder.results
        runtime_context = RuntimeContextImpl(
            controller=self.controller,
            recipe_data=self.execution_context.recipe_data,
            experiment_results=experiment_results,
            devices=ToolkitDevices(
                None if self.do_emulation else self.controller.devices
            ),
            do_emulation=self.do_emulation,
            legacy_session_data=self._legacy_session_data,
        )
        try:
            res_or_coro = func(runtime_context, **args)
            if isawaitable(res_or_coro):
                res = await res_or_coro
            else:
                res = res_or_coro
        except AbortExecution:
            _logger.warning(f"Execution aborted by near-time callback '{func_name}'")
            raise
        except BaseException as e:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' failed with: {e}"
            ) from e
        neartime_callback_results = (
            experiment_results.neartime_callback_results.setdefault(func_name, [])
        )
        neartime_callback_results.append(res)

    async def set_sw_param_handler(
        self,
        name: str,
        index: int,
        value: float,
        axis_name: str,
        values: NumPyArray,
        is_user_registered: bool,
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
        self.last_nt_step_result_completed = await self.controller._execute_one_step(
            execution_context=self.execution_context,
            sweep_params_tracker=self.sweep_params_tracker,
            user_set_nodes=self.user_set_nodes,
            nt_step=self.nt_step(),
        )
