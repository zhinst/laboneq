# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import traceback
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from numpy import typing as npt

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeSetAction,
    batch_set,
)
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import NtStepKey
from laboneq.executor.executor import AsyncExecutorBase, LoopFlags, LoopingMode

if TYPE_CHECKING:
    from laboneq.controller.controller import Controller

_logger = logging.getLogger(__name__)


class NearTimeRunner(AsyncExecutorBase):
    def __init__(self, controller: Controller):
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY)
        self.controller = controller
        self.user_set_nodes = []
        self.nt_loop_indices: list[int] = []
        self.pipeline_chunk: int = 0
        self.sweep_params_tracker = SweepParamsTracker()

    def nt_step(self) -> NtStepKey:
        return NtStepKey(indices=tuple(self.nt_loop_indices))

    async def set_handler(self, path: str, value):
        dev = self.controller._devices.find_by_node_path(path)
        self.user_set_nodes.append(
            DaqNodeSetAction(
                dev._daq, path, value, caching_strategy=CachingStrategy.NO_CACHE
            )
        )

    async def nt_callback_handler(self, func_name: str, args: dict[str, Any]):
        func = self.controller._neartime_callbacks.get(func_name)
        if func is None:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' is not registered."
            )
        try:
            if iscoroutinefunction(func):
                res = await func(
                    ProtectedSession(
                        self.controller._session, self.controller._results
                    ),
                    **args,
                )
            else:
                res = func(
                    ProtectedSession(
                        self.controller._session, self.controller._results
                    ),
                    **args,
                )
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
        values: npt.ArrayLike,
    ):
        self.sweep_params_tracker.set_param(name, value)

    async def for_loop_entry_handler(
        self, count: int, index: int, loop_flags: LoopFlags
    ):
        if loop_flags.is_pipeline:
            # Don't add the pipeliner loop index to NT indices
            self.pipeline_chunk = index
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
        if self.pipeline_chunk > 0:
            # Skip the pipeliner loop iterations, except the first one - iterated by the pipeliner itself
            return

        await self.controller._initialize_awgs(
            nt_step=self.nt_step(), rt_section_uid=uid
        )
        await self.controller._configure_triggers()
        nt_sweep_nodes = self.controller._prepare_nt_step(self.sweep_params_tracker)
        step_prepare_nodes = self.controller._prepare_rt_execution(rt_section_uid=uid)

        await batch_set([*self.user_set_nodes, *nt_sweep_nodes, *step_prepare_nodes])
        self.user_set_nodes.clear()
        self.sweep_params_tracker.clear_for_next_step()

        for retry in range(3):  # Up to 3 retries
            if retry > 0:
                _logger.info("Step retry %s of 3...", retry + 1)
                await batch_set(step_prepare_nodes)
            try:
                await self.controller._execute_one_step(
                    acquisition_type, rt_section_uid=uid
                )
                await self.controller._read_one_step_results(
                    nt_step=self.nt_step(), rt_section_uid=uid
                )
                break
            except LabOneQControllerException:
                # TODO(2K): introduce "hard" controller exceptions
                self.controller._report_step_error(
                    nt_step=self.nt_step(),
                    rt_section_uid=uid,
                    message=traceback.format_exc(),
                )
