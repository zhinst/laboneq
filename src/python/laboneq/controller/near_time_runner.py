# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import traceback
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any


from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import NtStepKey
from laboneq.executor.executor import AsyncExecutorBase, LoopFlags, LoopingMode

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.controller import Controller

_logger = logging.getLogger(__name__)


class NearTimeRunner(AsyncExecutorBase):
    def __init__(self, controller: Controller, protected_session: ProtectedSession):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self.controller = controller
        self.protected_session = protected_session
        self.user_set_nodes = NodeCollector()
        self.nt_loop_indices: list[int] = []
        self.pipeliner_job: int = 0
        self.sweep_params_tracker = SweepParamsTracker()

    def nt_step(self) -> NtStepKey:
        return NtStepKey(indices=tuple(self.nt_loop_indices))

    async def set_handler(self, path: str, value):
        self.user_set_nodes.add(path, value, cache=False)

    async def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        func = self.controller._neartime_callbacks.get(func_name)
        if func is None:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' is not registered."
            )
        try:
            if iscoroutinefunction(func):
                res = await func(self.protected_session, **args)
            else:
                res = func(self.protected_session, **args)
        except AbortExecution:
            _logger.warning(f"Execution aborted by near-time callback '{func_name}'")
            raise
        neartime_callback_results = (
            self.controller._results.neartime_callback_results.setdefault(func_name, [])
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
        if loop_flags.is_pipeline:
            # Don't add the pipeliner loop index to NT indices
            self.pipeliner_job = index
            return
        self.nt_loop_indices.append(index)

    async def for_loop_exit_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        if loop_flags.is_pipeline:
            return
        self.nt_loop_indices.pop()

    async def rt_entry_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        if self.pipeliner_job > 0:
            # Skip the pipeliner loop iterations, except the first one - iterated by the pipeliner itself
            return

        await self.controller._prepare_nt_step(
            sweep_params_tracker=self.sweep_params_tracker,
            user_set_nodes=self.user_set_nodes,
            nt_step=self.nt_step(),
            rt_section_uid=uid,
        )
        self.sweep_params_tracker.clear_for_next_step()
        self.user_set_nodes = NodeCollector()

        try:
            await self.controller._execute_one_step(
                acquisition_type, rt_section_uid=uid
            )
            await self.controller._read_one_step_results(
                nt_step=self.nt_step(), rt_section_uid=uid
            )
        except LabOneQControllerException:
            # TODO(2K): introduce "hard" controller exceptions
            self.controller._report_step_error(
                nt_step=self.nt_step(),
                rt_section_uid=uid,
                message=traceback.format_exc(),
            )

        await self.controller._after_nt_step()
