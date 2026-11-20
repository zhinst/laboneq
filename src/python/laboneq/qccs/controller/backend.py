# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from laboneq.controller.constants import CONNECT_CHECK_HOLDOFF
from laboneq.controller.controller_backend import (
    ChunkSubmission,
    ControllerBackend,
    StepSubmission,
)
from laboneq.controller.devices import labone
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.device_base import DeviceBase
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.near_time_replacement import check_replacements
from laboneq.controller.utilities.exception import (
    LabOneQControllerException,
    RecoverableExecutionError,
)
from laboneq.data.artifacts_qccs import ArtifactsCodegen
from laboneq.qccs.controller.near_time_replacement import process_replacements_qccs
from laboneq.qccs.controller.pre_processing import pre_process_compiled_qccs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from laboneq.controller.backend_runtime_state import BackendRuntimeState
    from laboneq.controller.devices.device_utils import NodeCollector
    from laboneq.controller.near_time_replacement import NearTimeReplacements
    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.controller.results import ResultsBuilder
    from laboneq.controller.versioning import SetupCaps
    from laboneq.data.nt_step_key import NtStepKey
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.data.setup_descriptions import SetupDescription
    from laboneq.dsl.device.device_setup import DeviceSetup


_logger = logging.getLogger(__name__)

# Handle of a step that reached the instruments. QCCS reads a step's results from
# its devices, so the handle needs no state.
_EXECUTED = object()

# Errors confined to one near-time step.
_STEP_ERRORS = (TimeoutError, LabOneQControllerException)


class QCCSControllerBackend(ControllerBackend):
    """Drives the instruments of a QCCS setup through their data servers."""

    def __init__(
        self,
        device_setup: DeviceSetup,
        setup_caps: SetupCaps,
        ignore_version_mismatch: bool,
    ):
        self._devices = DeviceCollection(
            device_setup=device_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            setup_caps=setup_caps,
        )
        self._last_connect_check_ts: float | None = None
        # Submitted steps whose results have not been read. Unsubscribing waits for
        # them, because a read in progress needs its subscription.
        self._reads_in_progress = 0
        self._reads_done = asyncio.Event()
        self._reads_done.set()

    def _read_submitted(self):
        self._reads_in_progress += 1
        self._reads_done.clear()

    def _read_finished(self):
        # A read of an abandoned run can finish after the hold was reset.
        self._reads_in_progress = max(self._reads_in_progress - 1, 0)
        if self._reads_in_progress == 0:
            self._reads_done.set()

    @property
    def device_setup_fingerprint(self) -> str:
        return self._devices.device_setup_fingerprint

    @property
    def setup_description(self) -> SetupDescription | None:
        return None

    @property
    def not_ready_reason(self) -> str | None:
        return None

    async def connect(
        self,
        do_emulation: bool,
        reset_devices: bool,
        disable_runtime_checks: bool,
        timeout_s: float | None,
    ):
        self._devices.set_timeout(timeout_s)
        now = time.monotonic()
        if (
            self._last_connect_check_ts is None
            or now - self._last_connect_check_ts > CONNECT_CHECK_HOLDOFF
        ):
            await self._devices.connect(
                do_emulation=do_emulation,
                reset_devices=reset_devices,
                disable_runtime_checks=disable_runtime_checks,
            )
        self._last_connect_check_ts = now
        async with self._devices.capture_logs():
            await self._devices.for_each(DeviceBase.reset_to_idle)

    async def disconnect(self):
        await self._devices.disconnect()
        self._last_connect_check_ts = None

    async def ensure_setup_description(self):
        return

    async def disable_outputs(
        self,
        device_uids: list[str] | None,
        logical_signals: list[str] | None,
        unused_only: bool,
    ):
        await self._devices.disable_outputs(device_uids, logical_signals, unused_only)

    def pre_process(
        self,
        scheduled_experiment: ScheduledExperiment,
        coprocessor_payload_overrides: dict[str, bytes],
    ) -> RecipeData:
        return pre_process_compiled_qccs(
            scheduled_experiment,
            self._devices.devices,
            coprocessor_payload_overrides,
            has_qhub=self._devices.has_qhub,
        )

    def create_backend_runtime_state(
        self, recipe_data: RecipeData
    ) -> BackendRuntimeState | None:
        return None

    @asynccontextmanager
    async def experiment(self, recipe_data: RecipeData) -> AsyncIterator[None]:
        cancelled = False
        async with self._devices.capture_logs():
            try:
                await self._devices.for_each(
                    DeviceBase.on_experiment_begin, recipe_data=recipe_data
                )
                await self._devices.for_each(
                    DeviceBase.initialize_oscillators, recipe_data
                )
                self._devices.reset_reported_device_errors()
                yield
            except LabOneQControllerException:
                # Report device errors, if any, to help diagnose the original exception.
                device_errors = await self._devices.fetch_device_errors()
                if device_errors is not None:
                    _logger.warning(device_errors)
                raise
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                # Ensure that the experiment run time is not included in the idle timeout for the connection check.
                self._last_connect_check_ts = time.monotonic()
                if cancelled:
                    # Reads of the abandoned run may never finish; the next run
                    # must not wait for them.
                    self._reads_in_progress = 0
                    self._reads_done.set()
                else:
                    await self._reads_done.wait()
                await self._devices.for_each(DeviceBase.on_experiment_end)

    async def prepare_nt_step(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        user_set_nodes: NodeCollector,
        replacements: NearTimeReplacements,
    ):
        if check_replacements(replacements, recipe_data):
            process_replacements_qccs(
                recipe_data.get_artifacts(ArtifactsCodegen),
                recipe_data,
                self._devices,
                nt_step,
                replacements,
            )
        try:
            await self._devices.for_each(DeviceBase.configure_trigger, recipe_data)
            await self._devices.for_each(
                DeviceBase.set_nt_step_nodes, recipe_data, user_set_nodes
            )
            await self._devices.for_each(DeviceBase.configure_feedback, recipe_data)
            # AWG / pipeliner upload
            await self._devices.for_each(
                DeviceBase.prepare_artifacts,
                recipe_data=recipe_data,
                nt_step=nt_step,
            )
        except (*_STEP_ERRORS, labone.LabOneCoreError) as e:
            raise RecoverableExecutionError(
                "Preparing the near-time step failed", cause=e
            ) from e

    async def execute_nt_step(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        timeout_s: float,
    ) -> StepSubmission:
        setup_attempted = False
        try:
            await self._devices.for_each(
                DeviceBase.wait_for_channels_ready,
                recipe_data=recipe_data,
                timeout_s=timeout_s,
            )
            setup_attempted = True
            await self._devices.for_each(
                DeviceBase.setup_one_step_execution,
                recipe_data=recipe_data,
                nt_step=nt_step,
                with_pipeliner=recipe_data.rt_execution_info.is_chunked,
            )
        except (*_STEP_ERRORS, labone.LabOneCoreError) as e:
            message = "Near-time step not started on the instruments"
            # A setup that failed on some instruments leaves the others set up.
            if setup_attempted:
                teardown_error = await self._teardown_one_step(recipe_data)
                if teardown_error is not None:
                    message += "\n" + _teardown_failure(teardown_error)
            return StepSubmission.not_submitted(
                nt_step, RecoverableExecutionError(message, cause=e)
            )
        self._read_submitted()
        failures: list[str] = []
        cause: Exception | None = None
        trigger = _StartTrigger()
        try:
            try:
                unmet = [
                    *await self._devices.for_each_collect(
                        DeviceBase.wait_for_execution_ready,
                        recipe_data=recipe_data,
                        timeout_s=timeout_s,
                    ),
                    *await self._wait_execution_to_stop(recipe_data, trigger),
                ]
                if unmet:
                    failures.append(
                        "Execution conditions not met:\n" + "\n".join(unmet)
                    )
            except (*_STEP_ERRORS, labone.LabOneCoreError) as e:
                failures.append(
                    "Timeout during execution of a near-time step"
                    if _is_timeout(e)
                    else str(e)
                )
                cause = e
            teardown_error = await self._teardown_one_step(recipe_data)
        except BaseException:
            # No result item is queued for this step, so no read releases it.
            self._read_finished()
            raise
        if teardown_error is not None:
            failures.append(_teardown_failure(teardown_error))
            cause = cause or teardown_error
        error = (
            RecoverableExecutionError("\n".join(failures), cause=cause)
            if failures
            else None
        )
        if error is not None and not trigger.emitted:
            # The step did not execute, so it has no results. The results of the
            # steps after it are read in order, and reading this step would take
            # the next step's results.
            self._read_finished()
            return StepSubmission(nt_step, [ChunkSubmission(None, error=error)])
        # All chunks run as one execution, so they share its outcome.
        return StepSubmission(
            nt_step, [ChunkSubmission(None, error=error, handle=_EXECUTED)]
        )

    async def _teardown_one_step(self, recipe_data: RecipeData) -> Exception | None:
        """Tears down the step on all instruments; returns the step error it raised."""
        try:
            await self._devices.for_each(
                DeviceBase.teardown_one_step_execution,
                recipe_data=recipe_data,
            )
        except (*_STEP_ERRORS, labone.LabOneCoreError) as e:
            return e
        return None

    async def _wait_execution_to_stop(
        self, recipe_data: RecipeData, trigger: _StartTrigger
    ) -> list[str]:
        """Starts the execution and returns the stop conditions that are not met.

        Marks `trigger` as emitted before emitting it.
        """
        target_devs = [device for _, device in self._devices.all]
        if len(target_devs) == 0:
            return []

        water_results = await _gather(
            *(
                device.make_waiter_for_execution_done(recipe_data=recipe_data)
                for device in target_devs
            )
        )
        wait_times, response_waiters = zip(*water_results)

        min_wait_time = max(wait_times)
        if min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            _logger.info("Estimated RT execution time: %.2f s.", min_wait_time)

        trigger.emitted = True
        await _gather(
            *(
                device.emit_start_trigger(recipe_data=recipe_data)
                for device in target_devs
            )
        )
        unmet = await _gather(
            *(
                device.wait_for_execution_done(
                    response_waiter=response_waiter,
                    min_wait_time=min_wait_time,
                )
                for device, response_waiter in zip(target_devs, response_waiters)
            )
        )
        return [item for device_unmet in unmet for item in device_unmet]

    async def after_nt_step(self):
        await self._devices.for_each(DeviceBase.update_warning_nodes)
        device_errors = await self._devices.fetch_new_device_errors()
        if device_errors is not None:
            raise RecoverableExecutionError(device_errors)

    async def read_results(
        self,
        recipe_data: RecipeData,
        step_submission: StepSubmission,
        results_builder: ResultsBuilder,
    ):
        try:
            await self._devices.for_each(
                DeviceBase.read_results,
                recipe_data=recipe_data,
                nt_step=step_submission.nt_step,
                results_builder=results_builder,
            )
        finally:
            self._read_finished()

    async def process_step_results(
        self,
        recipe_data: RecipeData,
        results_builder: ResultsBuilder,
        backend_runtime_state: BackendRuntimeState | None,
        nt_step: NtStepKey | None,
    ):
        return


@dataclass
class _StartTrigger:
    """Whether the start trigger of a near-time step has been emitted."""

    emitted: bool = False


def _is_timeout(e: Exception) -> bool:
    return isinstance(e, TimeoutError)


def _teardown_failure(e: Exception) -> str:
    reason = "timeout" if _is_timeout(e) else str(e)
    return f"Tearing down the near-time step failed: {reason}"
