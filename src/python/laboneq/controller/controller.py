# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.for_each import for_each_sync
from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
import numpy as np
import zhinst.utils  # type: ignore
from numpy import typing as npt

from laboneq import __version__
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.devices.device_utils import NodeCollector, zhinst_core_version
from laboneq.controller.devices.device_zi import DeviceBase, DeviceZI
from laboneq.controller.near_time_runner import NearTimeRunner
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.recipe_processor import (
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
    pre_process_compiled,
)
from laboneq.controller.results import make_acquired_result
from laboneq.controller.versioning import (
    MINIMUM_SUPPORTED_LABONE_VERSION,
    RECOMMENDED_LABONE_VERSION,
    LabOneVersion,
    SetupCaps,
)
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.utilities.async_helpers import EventLoopMixIn
from laboneq.core.utilities.replace_phase_increment import calc_ct_replacement
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements
from laboneq.data.execution_payload import TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import (
    CodegenWaveform,
    ScheduledExperiment,
)

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


_logger = logging.getLogger(__name__)


# Only recheck for the proper connected state if there was no check since more than
# the below amount of seconds. Important for performance with many small experiments
# executed in a batch.
CONNECT_CHECK_HOLDOFF = 10  # sec


class Controller(EventLoopMixIn):
    def __init__(
        self,
        target_setup: TargetSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable] | None = None,
    ):
        self._ignore_version_mismatch = ignore_version_mismatch
        self._do_emulation = True

        _zhinst_core_version = LabOneVersion.from_version_string(zhinst_core_version())
        self._check_zhinst_core_version_support(_zhinst_core_version)
        self._setup_caps = SetupCaps(client_version=_zhinst_core_version)

        self._devices = DeviceCollection(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            setup_caps=self._setup_caps,
        )

        self._last_connect_check_ts: float | None = None

        # Waves which are uploaded to the devices via pulse replacements
        self._current_waves: dict[str, CodegenWaveform] = {}
        self._neartime_callbacks: dict[str, Callable] = (
            {} if neartime_callbacks is None else neartime_callbacks
        )
        self._recipe_data: RecipeData = None
        self._results = ExperimentResults()

        _logger.debug("Controller created")
        _logger.debug("Controller debug logging is on")

        _logger.info("VERSION: laboneq %s", __version__)

    def _check_zhinst_core_version_support(self, version: LabOneVersion):
        if version < MINIMUM_SUPPORTED_LABONE_VERSION:
            err_msg = f"'zhinst.core' version '{version}' is not supported. We recommend {RECOMMENDED_LABONE_VERSION}."
            raise LabOneQControllerException(err_msg)

    @property
    def setup_caps(self) -> SetupCaps:
        return self._setup_caps

    @property
    def devices(self) -> dict[str, DeviceZI]:
        return self._devices.devices

    async def _prepare_nt_step(
        self,
        sweep_params_tracker: SweepParamsTracker,
        user_set_nodes: NodeCollector,
        nt_step: NtStepKey,
    ):
        # Trigger
        await self._devices.for_each(DeviceBase.configure_trigger, self._recipe_data)

        # NT Sweep parameters
        for param, value in sweep_params_tracker.updated_params():
            self._recipe_data.attribute_value_tracker.update(param, value)
        await self._devices.for_each(
            DeviceBase.set_nt_step_nodes, self._recipe_data, user_set_nodes
        )
        self._recipe_data.attribute_value_tracker.reset_updated()

        # Feedback
        await self._devices.for_each(DeviceBase.configure_feedback, self._recipe_data)

        # AWG / pipeliner upload
        await self._devices.for_each(
            DeviceBase.set_before_awg_upload, self._recipe_data
        )
        await self._devices.for_each(
            DeviceZI.prepare_artifacts,
            recipe_data=self._recipe_data,
            nt_step=nt_step,
        )
        await self._devices.for_each(DeviceBase.set_after_awg_upload, self._recipe_data)

    async def _after_nt_step(self):
        await self._devices.for_each(DeviceBase.update_warning_nodes)
        device_errors = await self._devices.fetch_device_errors()
        if device_errors is not None:
            raise LabOneQControllerException(device_errors)

    async def _wait_execution_to_stop(
        self, acquisition_type: AcquisitionType, rt_execution_info: RtExecutionInfo
    ):
        min_wait_time = self._recipe_data.recipe.max_step_execution_time
        if rt_execution_info.with_pipeliner:
            pipeliner_reload_worst_case = 1500e-6
            min_wait_time = (
                min_wait_time + pipeliner_reload_worst_case
            ) * rt_execution_info.pipeliner_jobs
        if min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            _logger.info("Estimated RT execution time: %.2f s.", min_wait_time)
        guarded_wait_time = round(
            min_wait_time * 1.1 + 1
        )  # +10% and fixed 1sec guard time

        target_devs = [
            device for _, device in self._devices.all if isinstance(device, DeviceBase)
        ]
        response_waiters = await _gather(
            *(
                device.make_waiter_for_execution_done(
                    acquisition_type=acquisition_type,
                    with_pipeliner=rt_execution_info.with_pipeliner,
                    timeout_s=guarded_wait_time,
                )
                for device in target_devs
            )
        )
        await _gather(
            *(
                device.emit_start_trigger(
                    with_pipeliner=rt_execution_info.with_pipeliner
                )
                for device in target_devs
            )
        )
        await _gather(
            *(
                device.wait_for_execution_done(
                    response_waiter=response_waiter,
                    timeout_s=guarded_wait_time,
                    min_wait_time=min_wait_time,
                )
                for device, response_waiter in zip(target_devs, response_waiters)
            )
        )

    # TODO(2K): use timeout passed to connect
    async def _execute_one_step(self, *, timeout_s=5.0):
        rt_execution_info = self._recipe_data.rt_execution_info

        try:
            await self._devices.for_each(
                DeviceBase.wait_for_channels_ready, timeout_s=timeout_s
            )

            await self._devices.for_each(
                DeviceBase.setup_one_step_execution,
                recipe_data=self._recipe_data,
                with_pipeliner=rt_execution_info.with_pipeliner,
            )

            await self._devices.for_each(
                DeviceBase.wait_for_execution_ready,
                with_pipeliner=rt_execution_info.with_pipeliner,
                timeout_s=timeout_s,
            )
            await self._wait_execution_to_stop(
                rt_execution_info.acquisition_type, rt_execution_info=rt_execution_info
            )
            await self._devices.for_each(
                DeviceBase.teardown_one_step_execution,
                with_pipeliner=rt_execution_info.with_pipeliner,
            )
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise LabOneQControllerException(
                "Timeout during execution of a near-time step"
            ) from e

    def connect(
        self,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        # Remember settings for later implicit connect check
        self._do_emulation = do_emulation
        self._devices.set_timeout(timeout_s)
        self._event_loop.run(
            self._connect_async,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
        )

    async def _connect_async(
        self, reset_devices: bool = False, disable_runtime_checks: bool = True
    ):
        now = time.monotonic()
        if (
            self._last_connect_check_ts is None
            or now - self._last_connect_check_ts > CONNECT_CHECK_HOLDOFF
        ):
            await self._devices.connect(
                do_emulation=self._do_emulation,
                reset_devices=reset_devices,
                disable_runtime_checks=disable_runtime_checks,
            )
        self._last_connect_check_ts = now

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        self._event_loop.run(
            self._disable_outputs_async, device_uids, logical_signals, unused_only
        )

    async def _disable_outputs_async(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._devices.disable_outputs(device_uids, logical_signals, unused_only)

    def disconnect(self):
        self._event_loop.run(self._disconnect_async)

    async def _disconnect_async(self):
        _logger.info("Disconnecting from all devices and servers...")
        await self._devices.disconnect()
        self._last_connect_check_ts = None
        _logger.info("Successfully disconnected from all devices and servers.")

    def execute_compiled(
        self,
        scheduled_experiment: ScheduledExperiment,
        protected_session: ProtectedSession | None = None,
    ):
        self._event_loop.run(
            self._execute_compiled_async, scheduled_experiment, protected_session
        )

    async def _execute_compiled_async(
        self,
        scheduled_experiment: ScheduledExperiment,
        protected_session: ProtectedSession | None = None,
    ):
        # Ensure all connect configurations are still valid!
        await self._connect_async()

        self._recipe_data = pre_process_compiled(scheduled_experiment, self._devices)
        self._prepare_result_shapes()

        if protected_session is None:
            protected_session = ProtectedSession(None)
        protected_session._set_experiment_results(self._results)

        async with self._devices.capture_logs():
            try:
                await self._devices.for_each(DeviceBase.reset_to_idle)
                for_each_sync(
                    self._devices.devices.values(),
                    DeviceBase.allocate_resources,
                    self._recipe_data,
                )
                await self._devices.for_each(DeviceBase.on_experiment_begin)
                await self._devices.for_each(DeviceBase.init_warning_nodes)
                await self._devices.for_each(
                    DeviceBase.apply_initialization, self._recipe_data
                )
                await self._devices.for_each(
                    DeviceBase.initialize_oscillators, self._recipe_data
                )

                # Ensure no side effects from the previous execution in the same session
                self._current_waves.clear()

                _logger.info("Starting near-time execution...")
                try:
                    await NearTimeRunner(
                        controller=self,
                        protected_session=protected_session,
                    ).run(self._recipe_data.execution)
                except AbortExecution:
                    # eat the exception
                    pass
                _logger.info("Finished near-time execution.")
            except LabOneQControllerException:
                # Report device errors if any - it maybe useful to diagnose the original exception
                device_errors = await self._devices.fetch_device_errors()
                if device_errors is not None:
                    _logger.warning(device_errors)
                raise
            finally:
                # Ensure that the experiment run time is not included in the idle timeout for the connection check.
                self._last_connect_check_ts = time.monotonic()
                await self._devices.for_each(DeviceBase.on_experiment_end)

    def results(self) -> ExperimentResults:
        return self._results

    def _find_awg(self, seqc_name: str) -> tuple[str | None, int | None]:
        # TODO(2K): Do this in the recipe preprocessor, or even modify the compiled experiment
        #  data model
        for rt_exec_step in self._recipe_data.recipe.realtime_execution_init:
            if rt_exec_step.program_ref == seqc_name:
                assert isinstance(rt_exec_step.awg_id, int)
                return rt_exec_step.device_id, rt_exec_step.awg_id
        return None, None

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        """Replaces specific pulse with the new sample data on the device.

        This is useful when called from the near-time callback, allows fast
        waveform replacement within near-time loop without experiment
        recompilation.

        Args:
            pulse_uid: pulse to replace, can be Pulse object or uid of the pulse
            pulse_or_array: replacement pulse, can be Pulse object or value array
            (see sampled_pulse_* from the pulse library)
        """
        if isinstance(pulse_uid, str):
            for waveform in self._recipe_data.scheduled_experiment.pulse_map[
                pulse_uid
            ].waveforms.values():
                if any([instance.can_compress for instance in waveform.instances]):
                    raise LabOneQControllerException(
                        f"Pulse replacement on pulses that allow compression not "
                        f"allowed. Pulse {pulse_uid}"
                    )

        if getattr(pulse_uid, "can_compress", False):
            raise LabOneQControllerException(
                f"Pulse replacement on pulses that allow compression not allowed. "
                f"Pulse {pulse_uid.uid}"  # type: ignore
            )

        acquisition_type = self._recipe_data.rt_execution_info.acquisition_type
        wave_replacements = calc_wave_replacements(
            self._recipe_data.scheduled_experiment,
            pulse_uid,
            pulse_or_array,
            self._current_waves,
        )
        for repl in wave_replacements:
            awg_indices = next(
                a
                for a in self._recipe_data.scheduled_experiment.wave_indices
                if a["filename"] == repl.awg_id
            )
            awg_wave_map: dict[str, list[int | str]] = awg_indices["value"]
            target_wave_index = awg_wave_map[repl.sig_string][0]
            assert isinstance(target_wave_index, int)
            seqc_name = repl.awg_id
            awg = self._find_awg(seqc_name)
            assert awg[0] is not None and awg[1] is not None
            device = self._devices.find_by_uid(awg[0])

            if repl.replacement_type == ReplacementType.I_Q:
                assert isinstance(repl.samples, list)
                clipped = np.clip(repl.samples, -1.0, 1.0)
                bin_wave = zhinst.utils.convert_awg_waveform(*clipped)
                device.add_waveform_replacement(
                    awg_index=awg[1],
                    wave=WaveformItem(
                        index=target_wave_index,
                        name=repl.sig_string + " (repl)",
                        samples=bin_wave,
                    ),
                    acquisition_type=acquisition_type,
                )
            elif repl.replacement_type == ReplacementType.COMPLEX:
                assert isinstance(repl.samples, np.ndarray)
                np.clip(repl.samples.real, -1.0, 1.0, out=repl.samples.real)
                np.clip(repl.samples.imag, -1.0, 1.0, out=repl.samples.imag)
                device.add_waveform_replacement(
                    awg_index=awg[1],
                    wave=WaveformItem(
                        index=target_wave_index,
                        name=repl.sig_string + " (repl)",
                        samples=repl.samples,
                    ),
                    acquisition_type=acquisition_type,
                )

    def replace_phase_increment(self, parameter_uid: str, new_value: int | float):
        ct_replacements = calc_ct_replacement(
            self._recipe_data.scheduled_experiment, parameter_uid, new_value
        )
        for repl in ct_replacements:
            seqc_name = repl["seqc"]
            device_id, awg_index = self._find_awg(seqc_name)
            assert device_id is not None and awg_index is not None
            device = self._devices.find_by_uid(device_id)
            device.add_command_table_replacement(awg_index, repl["ct"])

    def _prepare_result_shapes(self):
        self._results = ExperimentResults()
        for handle, shape_info in self._recipe_data.result_shapes.items():
            empty_result = make_acquired_result(
                data=np.full(
                    shape=tuple(shape_info.base_shape),
                    fill_value=np.nan,
                    dtype=np.complex128,
                ),
                axis_name=shape_info.base_axis_name,
                axis=shape_info.base_axis,
                handle=handle,
            )
            self._results.acquired_results[handle] = empty_result

    async def _read_one_step_results(self, nt_step: NtStepKey):
        await self._devices.for_each(
            DeviceZI.read_results,
            recipe_data=self._recipe_data,
            nt_step=nt_step,
            results=self._results,
        )

    def _report_step_error(self, nt_step: NtStepKey, uid: str, message: str):
        self._results.execution_errors.append((list(nt_step.indices), uid, message))
