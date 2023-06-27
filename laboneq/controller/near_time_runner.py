# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import traceback
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from numpy import typing as npt

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
    batch_set,
)
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.recipe_enums import NtStepKey
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.executor.executor import ExecutorBase, LoopFlags, LoopingMode

if TYPE_CHECKING:
    from laboneq.controller.controller import Controller

_logger = logging.getLogger(__name__)


class NearTimeRunner(ExecutorBase):
    def __init__(self, controller: Controller):
        super().__init__(looping_mode=LoopingMode.EXECUTE)
        self.controller = controller
        self.user_set_nodes = []
        self.nt_loop_indices: list[int] = []
        self.sweep_params_tracker = SweepParamsTracker()

    def nt_step(self) -> NtStepKey:
        return NtStepKey(indices=tuple(self.nt_loop_indices))

    def set_handler(self, path: str, value):
        dev = self.controller._devices.find_by_node_path(path)
        self.user_set_nodes.append(
            DaqNodeSetAction(
                dev._daq, path, value, caching_strategy=CachingStrategy.NO_CACHE
            )
        )

    def user_func_handler(self, func_name: str, args: dict[str, Any]):
        func = self.controller._user_functions.get(func_name)
        if func is None:
            raise LabOneQControllerException(
                f"User function '{func_name}' is not registered."
            )
        res = func(ProtectedSession(self.controller._session), **args)
        user_func_results = self.controller._results.user_func_results.setdefault(
            func_name, []
        )
        user_func_results.append(res)

    def set_sw_param_handler(
        self,
        name: str,
        index: int,
        value: float,
        axis_name: str,
        values: npt.ArrayLike,
    ):
        self.sweep_params_tracker.set_param(name, value)

    @contextmanager
    def for_loop_handler(self, count: int, index: int, loop_flags: LoopFlags):
        self.nt_loop_indices.append(index)
        yield
        self.nt_loop_indices.pop()

    @contextmanager
    def rt_handler(
        self,
        count: int,
        uid: str,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        self.controller._initialize_awgs(nt_step=self.nt_step())
        self.controller._configure_triggers()
        attribute_value_tracker = self.controller._recipe_data.attribute_value_tracker
        for param in self.sweep_params_tracker.updated_params():
            attribute_value_tracker.update(
                param, self.sweep_params_tracker.get_param(param)
            )
        self.sweep_params_tracker.clear_for_next_step()

        nt_sweep_nodes: list[DaqNodeAction] = []
        for device_uid, device in self.controller._devices.all:
            nt_sweep_nodes.extend(
                device.collect_prepare_nt_step_nodes(
                    attribute_value_tracker.device_view(device_uid),
                    self.controller._recipe_data,
                )
            )

        step_prepare_nodes = self.controller._prepare_rt_execution(rt_section_uid=uid)

        batch_set([*self.user_set_nodes, *nt_sweep_nodes, *step_prepare_nodes])
        self.user_set_nodes.clear()

        for retry in range(3):  # Up to 3 retries
            if retry > 0:
                _logger.info("Step retry %s of 3...", retry + 1)
                batch_set(step_prepare_nodes)
            try:
                self.controller._execute_one_step(acquisition_type)
                self.controller._read_one_step_results(
                    nt_step=self.nt_step(), rt_section_uid=uid
                )
                break
            except LabOneQControllerException:  # TODO(2K): introduce "hard" controller exceptions
                self.controller._report_step_error(
                    nt_step=self.nt_step(),
                    rt_section_uid=uid,
                    message=traceback.format_exc(),
                )
        yield
