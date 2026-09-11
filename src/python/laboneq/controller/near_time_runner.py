# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.runtime_context_impl import RuntimeContextImpl
from laboneq.controller.toolkit_adapter import ToolkitDevices
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
from laboneq.core.exceptions import AbortExecution
from laboneq.data.nt_step_key import NtStepKey
from laboneq.executor.executor import AsyncExecutorBase, LoopingMode

if TYPE_CHECKING:
    import asyncio

    from laboneq.controller.callback_channel import CallbackChannel
    from laboneq.controller.controller import Controller, ExecutionContext
    from laboneq.controller.runtime_context_impl import LegacySessionData
    from laboneq.core.types.enums.acquisition_type import AcquisitionType
    from laboneq.core.types.enums.averaging_mode import AveragingMode
    from laboneq.core.types.numpy_support import NumPyArray

_logger = logging.getLogger(__name__)


def _json_safe_args(func_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize numpy scalars from sweep resolution and reject non-JSON values.

    Complex values are not supported — decompose into real/imaginary parts.
    """

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"type {type(value).__name__}")

    try:
        # NaN/Infinity pass through deliberately: both ends of the channel use
        # Python's stdlib json, which tolerates the non-standard literals.
        return json.loads(json.dumps(args, default=default))
    except (TypeError, ValueError) as e:
        raise LabOneQControllerException(
            f"Arguments of client near-time callback '{func_name}' must be "
            f"JSON-serializable: {e}"
        ) from e


def _deserialize_replacement(func_name: str, payload: Any) -> Any:
    # Deferred to break the import cycle (laboneq.serializers loads the controller
    # package); the module itself is already loaded at service startup via routes.py.
    from laboneq import serializers

    if not isinstance(payload, dict) or "__serializer__" not in payload:
        raise LabOneQControllerException(
            f"Malformed pulse payload in the reply of near-time callback "
            f"'{func_name}': expected a laboneq.serializers dict."
        )
    try:
        return serializers.from_dict(payload)
    except Exception as e:
        raise LabOneQControllerException(
            f"Malformed pulse payload in the reply of near-time callback "
            f"'{func_name}': {e}"
        ) from e


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

    async def nt_callback_handler(
        self,
        func_name: str,
        args: dict[str, Any],
    ):
        if (
            self.last_nt_step_result_completed is not None
            and not self.last_nt_step_result_completed.done()
        ):
            # This is necessary to ensure that the existing logic for partial results
            # is not broken by the new asynchronous execution model.
            await self.last_nt_step_result_completed
        channel = self.execution_context.submission.callback_channel
        # Callbacks registered on the controller run here; anything else is
        # dispatched to the client over the callback channel when one exists.
        if channel is not None and func_name not in self.controller.neartime_callbacks:
            res = await self._run_client_callback(channel, func_name, args)
        else:
            res = await self._run_local_callback(func_name, args)
        experiment_results = self.execution_context.submission.results_builder.results
        experiment_results.neartime_callback_results.setdefault(func_name, []).append(
            res
        )

    async def _run_local_callback(self, func_name: str, args: dict[str, Any]) -> Any:
        func = self.controller.neartime_callbacks.get(func_name)
        if func is None:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' is not registered."
            )
        runtime_context = RuntimeContextImpl(
            controller=self.controller,
            recipe_data=self.execution_context.recipe_data,
            experiment_results=self.execution_context.submission.results_builder.results,
            devices=ToolkitDevices(
                None if self.do_emulation else self.controller.devices
            ),
            do_emulation=self.do_emulation,
            legacy_session_data=self._legacy_session_data,
        )
        try:
            res_or_coro = func(runtime_context, **args)
            if isawaitable(res_or_coro):
                return await res_or_coro
            return res_or_coro
        except AbortExecution:
            _logger.warning(f"Execution aborted by near-time callback '{func_name}'")
            raise
        except BaseException as e:
            raise LabOneQControllerException(
                f"Near-time callback '{func_name}' failed with: {e}"
            ) from e

    async def _run_client_callback(
        self,
        channel: CallbackChannel,
        func_name: str,
        args: dict[str, Any],
    ) -> Any:
        reply = await channel.invoke(
            func_name, _json_safe_args(func_name, args), self.nt_loop_indices, None
        )
        self._apply_callback_actions(func_name, reply.actions)
        return reply.return_value

    def _apply_callback_actions(
        self, func_name: str, actions: list[dict[str, Any]]
    ) -> None:
        for action in actions:
            kind = action.get("action")
            try:
                if kind == "replace_pulse":
                    self.controller.register_pulse_replacement(
                        pulse_uid=action["pulse_uid"],
                        replacement=_deserialize_replacement(
                            func_name, action["pulse"]
                        ),
                    )
                elif kind == "replace_phase_increment":
                    self.controller.register_phase_increment_replacement(
                        parameter_uid=action["parameter_uid"],
                        new_value=action["value"],
                    )
                elif kind == "abort_execution":
                    _logger.warning(
                        f"Execution aborted by near-time callback '{func_name}'"
                    )
                    raise AbortExecution
                else:
                    raise LabOneQControllerException(
                        f"Unknown action {kind!r} in the reply of near-time "
                        f"callback '{func_name}'."
                    )
            except KeyError as e:
                raise LabOneQControllerException(
                    f"Malformed {kind!r} action in the reply of near-time "
                    f"callback '{func_name}': missing field {e.args[0]!r}."
                ) from e

    async def set_sw_param_handler(
        self,
        name: str,
        index: int,
        value: float,
        axis_name: str,
        values: NumPyArray,
    ):
        self.sweep_params_tracker.set_param(name, value)

    async def for_loop_entry_handler(self, count: int, index: int):
        self.nt_loop_indices.append(index)

    async def for_loop_exit_handler(self, count: int, index: int):
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
