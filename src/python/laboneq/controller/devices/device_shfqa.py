# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Iterator

from laboneq.controller.attribute_value_tracker import (
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.core_base import CoreBase, SHFChannelBase
from laboneq.controller.devices.device_shf_base import (
    OPT_16_INTEGRATORS,
    OPT_LONG_READOUT,
    DeviceSHFBase,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import RawReadoutData
from laboneq.controller.devices.qachannel import QAChannel, SHFQAMixIn
from laboneq.controller.recipe_processor import (
    VIRTUAL_SHFSG_UID_SUFFIX,
    AwgType,
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
)
from laboneq.controller.utilities.for_each import for_each
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.data.recipe import (
    Initialization,
    NtStepKey,
)
from laboneq.data.scheduled_experiment import ScheduledExperiment

_logger = logging.getLogger(__name__)

INTERNAL_TRIGGER_CHANNEL = 8  # PQSC style triggering on the SHFSG/QC
SOFTWARE_TRIGGER_CHANNEL = 1024  # Software triggering on the SHFQA

SAMPLE_FREQUENCY_HZ = 2.0e9


class DeviceSHFQA(SHFQAMixIn, DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFQA4"
        self.dev_opts = []
        self._qachannels: list[QAChannel] = []
        self._channels = 4
        # TODO(2K): This is the number of available integrators.
        # Determine the actual number of integrators in use based on device setup.
        self._integrators = 16
        self._long_readout_available = True
        self._wait_for_awgs = True
        self._emit_trigger = False

    def all_cores(self) -> Iterable[CoreBase]:
        return iter(self._qachannels)

    def allocated_cores(self, recipe_data: RecipeData) -> Iterable[CoreBase]:
        for ch in recipe_data.allocated_awgs(self.uid):
            yield self._qachannels[ch]

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/QA:{self.serial}"
        return f"SHFQA:{self.serial}"

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        self._process_shf_opts()
        if self.dev_type == "SHFQA4":
            self._channels = 4
        elif self.dev_type == "SHFQA2":
            self._channels = 2
        elif self.dev_type == "SHFQC":
            self._channels = 1
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFQA4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
        self._integrators = (
            16
            if OPT_16_INTEGRATORS in self.dev_opts or self.dev_type == "SHFQA4"
            else 8
        )
        self._long_readout_available = OPT_LONG_READOUT in self.dev_opts
        self._qachannels = [
            QAChannel(
                api=self._api,
                subscriber=self._subscriber,
                device_uid=self.uid,
                serial=self.serial,
                core_index=core_index,
                integrators=self._integrators,
                repr_base=self.dev_repr,
                is_plus=self._is_plus,
                long_readout_available=self._long_readout_available,
            )
            for core_index in range(self._channels)
        ]

    def validate_scheduled_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        rt_execution_info: RtExecutionInfo,
    ):
        self._validate_scheduled_experiment_shfqa(
            scheduled_experiment=scheduled_experiment
        )

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)
        yield from self._pre_process_attributes_shfqa(initialization)

    def validate_recipe_data(self, recipe_data: RecipeData):
        self._validate_recipe_data_shfqa(recipe_data=recipe_data)

    def _make_osc_path(self, channel: int, index: int, awg_type: AwgType) -> str:
        return self._qachannels[channel].nodes.osc_freq[index]

    def _busy_nodes(self, recipe_data: RecipeData) -> list[str]:
        if not self._setup_caps.supports_shf_busy:
            return []
        return [
            self._qachannels[ch].nodes.busy
            for ch in recipe_data.allocated_awgs(self.uid)
        ]

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        hw_sync = with_pipeliner and (
            self._has_awg_in_use(recipe_data)
            # TODO(2K): Remove this workaround once SHFQC is correctly modelled in the controller.
            or self.options.is_qc is True
            and self._has_awg_in_use(
                recipe_data, device_uid=self.uid + VIRTUAL_SHFSG_UID_SUFFIX
            )
        )
        await self._set_hw_sync(hw_sync=hw_sync, emit_trigger=self._emit_trigger)

    async def emit_start_trigger(self, recipe_data: RecipeData):
        if self._emit_trigger:
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add(
                (
                    "system/internaltrigger/enable"
                    if self.options.is_qc
                    else "system/swtriggers/0/single"
                ),
                1,
                cache=False,
            )
            await self.set_async(nc)

    async def teardown_one_step_execution(self, recipe_data: RecipeData):
        if not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            await self.set_async(
                NodeCollector.one(f"/{self.serial}/system/synchronization/source", 0)
            )

        await for_each(
            self.allocated_cores(recipe_data=recipe_data),
            SHFChannelBase.teardown_one_step_execution,
            with_pipeliner=recipe_data.rt_execution_info.is_chunked,
        )

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        await for_each(
            self.all_cores(),
            CoreBase.set_nt_step_nodes,
            recipe_data=recipe_data,
            attributes=attributes,
        )

    def prepare_upload_binary_wave(
        self,
        awg_index: int,
        wave: WaveformItem,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        if is_spectroscopy(acquisition_type):
            return self._qachannels[awg_index].upload_spectroscopy_envelope(wave)
        return self._qachannels[awg_index].upload_generator_wave(wave)

    async def configure_trigger(self, recipe_data: RecipeData):
        device_recipe_data = recipe_data.device_settings[self.uid]
        self._wait_for_awgs = True
        self._emit_trigger = False

        if self.is_standalone():
            self._wait_for_awgs = False
            self._emit_trigger = True
            if self.options.is_qc:
                nc = NodeCollector(base=f"/{self.serial}/")
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
                await self._api.set_parallel(nc)

        trig_channel = 0
        if self.is_standalone():
            trig_channel = (
                INTERNAL_TRIGGER_CHANNEL
                if self.options.is_qc
                else SOFTWARE_TRIGGER_CHANNEL
            )

        await for_each(
            self.allocated_cores(recipe_data=recipe_data),
            QAChannel.configure_trigger,
            trig_channel=trig_channel,
        )
        if len(device_recipe_data.allocated_awgs()) == 0:
            # TODO(2K): Purpose of the no-allocated-AWGs case is unclear; kept to maintain original behavior:
            # https://gitlab.zhinst.com/laboneq/laboneq/-/merge_requests/1303/diffs#6ac44cb93a82be7127f076e9108cf97cf6c8f4b5_809_810
            nc = NodeCollector(base=f"/{self.serial}/qachannels/0/")
            nc.add("markers/0/source", 32)
            nc.add("markers/1/source", 36)
            await self._api.set_parallel(nc)

    async def on_experiment_begin(self, recipe_data: RecipeData):
        await _gather(
            super().on_experiment_begin(recipe_data=recipe_data),
            *(
                self._subscriber.subscribe(self._api, path)
                for ch in recipe_data.allocated_awgs(self.uid)
                for path in self._qachannels[ch].subscribe_nodes()
            ),
        )

    async def on_experiment_end(self):
        await _gather(
            super().on_experiment_end(),
            self._api.set_parallel(
                QAChannel.on_experiment_end_nodes(base=f"/{self.serial}/")
            ),
        )

    async def get_measurement_data(
        self,
        *,
        core_index: int,
        recipe_data: RecipeData,
        num_results: int,
        hw_averages: int,
    ) -> list[RawReadoutData]:
        return await self._qachannels[core_index].get_measurement_data(
            recipe_data=recipe_data,
            num_results=num_results,
        )

    def extract_raw_readout(
        self,
        *,
        all_raw_readouts: list[RawReadoutData],
        integrators: list[int],
        rt_execution_info: RtExecutionInfo,
    ) -> RawReadoutData:
        return QAChannel.extract_raw_readout(
            dev_repr=self.dev_repr,
            all_raw_readouts=all_raw_readouts,
            integrators=integrators,
            rt_execution_info=rt_execution_info,
        )

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None, timeout_s: float
    ) -> RawReadoutData:
        return await self._qachannels[channel].get_raw_data(
            acquire_length=acquire_length,
            acquires=acquires,
            timeout_s=timeout_s,
        )

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc_qa = QAChannel.reset_to_idle_nodes(
            base=f"/{self.serial}/",
            is_qc=self.options.is_qc is True,
            long_readout_available=self._long_readout_available,
        )
        await self._api.set_parallel(nc_qa)

    def _collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        for channel in self.all_cores():
            warning_nodes.extend(channel.collect_warning_nodes())
        return warning_nodes
