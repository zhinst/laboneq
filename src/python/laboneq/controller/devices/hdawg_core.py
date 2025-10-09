# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import json
from typing import Any

from laboneq.controller.attribute_value_tracker import (
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
    ResponseWaiterAsync,
    _gather,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.recipe_processor import (
    DeviceRecipeData,
    HDAWGRecipeData,
    HWModulation,
    RecipeData,
    TrigOutSource,
    get_elf,
    prepare_command_table,
    prepare_waves,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import ArtifactsCodegen
import numpy as np


class ModulationMode(IntEnum):
    OFF = 0
    SINE_00 = 1
    SINE_11 = 2
    SINE_01 = 3
    SINE_10 = 4
    ADVANCED = 5
    MIXER_CAL = 6


@dataclass
class HDOutputNodes:
    output_on: str


class HDOutput:
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        repr_base: str,
        has_precompensation: bool,
    ):
        self._api = api
        self._subscriber = subscriber
        self._device_uid = device_uid
        self._serial = serial
        self._channel = channel
        self._unit_repr = f"{repr_base}:ch{channel}"
        self._has_precompensation = has_precompensation
        self._awg_core = channel // 2
        self._rel_channel = channel % 2

    async def configure(self, awg_core_recipe_data: HDAWGRecipeData):
        ch_recipe_data = awg_core_recipe_data.outputs[self._rel_channel]
        nc = NodeCollector(base=f"/{self._serial}/sigouts/{self._channel}/")
        nc_awg = NodeCollector(
            base=f"/{self._serial}/awgs/{self._awg_core}/outputs/{self._rel_channel}/"
        )
        nc_trg = NodeCollector(base=f"/{self._serial}/triggers/out/{self._channel}/")

        if ch_recipe_data.enable is True:
            nc.add("on", 1 if ch_recipe_data.enable else 0)

        ### Range
        if ch_recipe_data.range is not None:
            nc.add("range", ch_recipe_data.range)

        ### Modulation
        modulation_mode: ModulationMode | None = None
        match ch_recipe_data.hw_modulation:
            case HWModulation.IQ:
                modulation_mode = ModulationMode.MIXER_CAL
            case HWModulation.SINE:
                modulation_mode = (
                    ModulationMode.SINE_00
                    if self._rel_channel == 0
                    else ModulationMode.SINE_11
                )
            case HWModulation.OFF:
                modulation_mode = ModulationMode.OFF
        if modulation_mode is not None:
            nc_awg.add("modulation/mode", modulation_mode)

        ### Precompensation
        precomp = ch_recipe_data.precompensation
        precomp_p = "precompensation/"
        try:
            if not precomp:
                raise AttributeError  # @IgnoreException
            if not self._has_precompensation:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Precompensation is not supported."
                )
            nc.add(precomp_p + "enable", 1)
            # Exponentials
            for e in range(8):
                exp_p = precomp_p + f"exponentials/{e}/"
                try:
                    exp = precomp["exponential"][e]
                    nc.add(exp_p + "enable", 1)
                    nc.add(exp_p + "timeconstant", exp["timeconstant"])
                    nc.add(exp_p + "amplitude", exp["amplitude"])
                except (KeyError, IndexError, TypeError):
                    nc.add(exp_p + "enable", 0)
            # Bounce
            bounce_p = precomp_p + "bounces/0/"
            try:
                bounce = precomp["bounce"]
                delay = bounce["delay"]
                amp = bounce["amplitude"]
                nc.add(bounce_p + "enable", 1)
                nc.add(bounce_p + "delay", delay)
                nc.add(bounce_p + "amplitude", amp)
            except (KeyError, TypeError):
                nc.add(bounce_p + "enable", 0)
            # Highpass
            hp_p = precomp_p + "highpass/0/"
            try:
                hp = precomp["high_pass"]
                timeconstant = hp["timeconstant"]
                nc.add(hp_p + "enable", 1)
                nc.add(hp_p + "timeconstant", timeconstant)
                nc.add(hp_p + "clearing/slope", 1)
            except (KeyError, TypeError):
                nc.add(hp_p + "enable", 0)
            # FIR
            fir_p = precomp_p + "fir/"
            try:
                fir = np.array(precomp["FIR"]["coefficients"])
                if len(fir) > 40:
                    raise LabOneQControllerException(
                        "FIR coefficients must be a list of at most 40 doubles"
                    )
                fir = np.concatenate((fir, np.zeros((40 - len(fir)))))
                nc.add(fir_p + "enable", 1)
                nc.add(fir_p + "coefficients", fir)
            except (KeyError, IndexError, TypeError):
                nc.add(fir_p + "enable", 0)
        except (KeyError, TypeError, AttributeError):
            if self._has_precompensation and precomp is None:
                nc.add(precomp_p + "enable", 0)

        ### Trigger output
        if ch_recipe_data.trig_out_source != TrigOutSource.NONE:
            if ch_recipe_data.trig_out_source == TrigOutSource.AWG_TRIGGER:
                nc_trg.add("source", self._rel_channel)
            elif ch_recipe_data.trig_out_source == TrigOutSource.MARKER_1:
                nc_trg.add("source", 4 + 2 * self._rel_channel)
            # set trigger delay to 0
            nc_trg.add("delay", 0.0)

        await self._api.set_parallel(NodeCollector.all((nc, nc_awg, nc_trg)))


@dataclass
class HDAwgCoreNodes:
    output_on: str
    awg_elf_data: str
    awg_elf_progress: str
    awg_enable: str
    awg_ready: str
    awg_command_table: str
    osc_freq: list[str]
    busy: str


class HDAwgCore(ChannelBase):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        repr_base: str,
        is_follower: bool,
        is_leader: bool,
        has_precompensation: bool,
    ):
        super().__init__(api, subscriber, device_uid, serial, channel)
        self._node_base = f"/{serial}/awgs/{channel}"
        self._unit_repr = f"{repr_base}:awg{channel}"
        self._is_follower = is_follower
        self._is_leader = is_leader
        self._pipeliner = AwgPipeliner(self._node_base, f"AWG{channel}")
        self.nodes = HDAwgCoreNodes(
            output_on=f"{self._node_base}/output/on",
            awg_elf_data=f"{self._node_base}/elf/data",
            awg_elf_progress=f"{self._node_base}/elf/progress",
            awg_enable=f"{self._node_base}/enable",
            awg_ready=f"{self._node_base}/ready",
            awg_command_table=f"{self._node_base}/commandtable",
            osc_freq=[f"{self._node_base}/oscs/{i}/freq" for i in range(8)],
            busy=f"{self._node_base}/busy",
        )
        self._outputs: list[HDOutput] = [
            HDOutput(
                api,
                subscriber,
                device_uid,
                serial,
                channel=ch,
                repr_base=self._unit_repr,
                has_precompensation=has_precompensation,
            )
            for ch in range(channel * 2, channel * 2 + 2)
        ]

    @property
    def pipeliner(self) -> AwgPipeliner:
        return self._pipeliner

    def _disable_output(self) -> NodeCollector:
        return NodeCollector()

    def allocate_resources(self):
        self._pipeliner._reload_tracker.reset()

    async def apply_initialization(self, device_recipe_data: DeviceRecipeData):
        awg_core_recipe_data = device_recipe_data.hdawgcores.get(self._channel)
        if awg_core_recipe_data is None:
            if len(device_recipe_data.hdawgcores) == 0 and self._channel == 0:
                # Ensure configuring at least one AWG instance to cover the case that the instrument
                # is only used as a communication proxy. Some of the nodes on the AWG branch are
                # needed to get proper communication between HDAWG and UHFQA.
                await self._configure_awg_core(None)
            return

        await _gather(
            self._outputs[0].configure(awg_core_recipe_data=awg_core_recipe_data),
            self._outputs[1].configure(awg_core_recipe_data=awg_core_recipe_data),
            self._configure_awg_core(awg_core_recipe_data=awg_core_recipe_data),
        )

    async def _configure_awg_core(self, awg_core_recipe_data: HDAWGRecipeData | None):
        # Configure DIO/ZSync at init (previously was after AWG loading).
        # This is a prerequisite for passing AWG checks in FW on the pipeliner commit.
        # Without the pipeliner, these checks are only performed when the AWG is enabled,
        # therefore DIO could be configured after the AWG loading.
        nc_awg = NodeCollector(base=f"/{self._serial}/awgs/{self._channel}/")

        if awg_core_recipe_data is not None:
            nc_awg.add("single", 1)

        if self._is_follower:
            nc_awg.add("dio/strobe/slope", 0)
            nc_awg.add("dio/valid/polarity", 0)

        elif self._is_leader:
            nc_awg.add("auxtriggers/0/slope", 1)
            nc_awg.add("auxtriggers/0/channel", 0)

            nc_awg.add("dio/strobe/slope", 0)
            nc_awg.add("dio/valid/polarity", 2)
            nc_awg.add("dio/valid/index", 0)
            nc_awg.add("dio/mask/value", 0x3FF)
            nc_awg.add("dio/mask/shift", 1)

            # To align execution on all AWG cores on same instrument, enable  HW
            # synchronization even in absence of pipeliner usage.
            if awg_core_recipe_data is not None:
                # TODO(2K): This node is not pipelinable, so it cannot be changed between pipeliner jobs.
                # However, jobs may involve different AWG cores. In the future, enable synchronization
                # across all cores, while loading a dummy program on cores not used in a given job.
                nc_awg.add("synchronization/enable", 1)

        await self._api.set_parallel(nc_awg)

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        pass

    async def load_awg_program(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        # TODO(2K): same code as for SGChannel - consider refactoring to a common function / AWG base class
        artifacts = recipe_data.get_artifacts(ArtifactsCodegen)
        rt_execution_info = recipe_data.rt_execution_info

        elf_nodes = NodeCollector()
        upload_ready_conditions: dict[str, Any] = {}

        if rt_execution_info.with_pipeliner:
            # enable pipeliner
            elf_nodes.extend(self._pipeliner.prepare_for_upload())

        for pipeliner_job in range(rt_execution_info.pipeliner_jobs):
            effective_nt_step = (
                NtStepKey(indices=tuple([*nt_step.indices, pipeliner_job]))
                if rt_execution_info.with_pipeliner
                else nt_step
            )
            rt_exec_step = next(
                (
                    r
                    for r in recipe_data.recipe.realtime_execution_init
                    if r.device_id == self._device_uid
                    and r.awg_index == self._channel
                    and r.nt_step == effective_nt_step
                ),
                None,
            )

            if rt_execution_info.with_pipeliner:
                rt_exec_step = self._pipeliner._reload_tracker.calc_next_step(
                    pipeliner_job=pipeliner_job,
                    rt_exec_step=rt_exec_step,
                )

            if rt_exec_step is None:
                continue

            seqc_elf = get_elf(artifacts, rt_exec_step.program_ref)
            if seqc_elf is not None:
                elf_nodes.add(
                    path=self.nodes.awg_elf_data,
                    value=seqc_elf,
                    cache=False,
                    filename=rt_exec_step.program_ref,
                )

            waves = prepare_waves(artifacts, rt_exec_step.wave_indices_ref)
            if waves is not None:
                for wave in waves:
                    elf_nodes.add(
                        path=f"{self._node_base}/waveform/waves/{wave.index}",
                        value=wave.samples,
                        cache=False,
                        filename=wave.name,
                    )

            command_table = prepare_command_table(
                artifacts, rt_exec_step.wave_indices_ref
            )
            if command_table is not None:
                elf_nodes.add(
                    path=self.nodes.awg_command_table + "/data",
                    value=json.dumps(command_table, sort_keys=True),
                    cache=False,
                )

            if rt_execution_info.with_pipeliner:
                elf_nodes.extend(self._pipeliner.commit())

        if rt_execution_info.with_pipeliner:
            upload_ready_conditions.update(self._pipeliner.ready_conditions())

        rw = ResponseWaiterAsync(api=self._api, dev_repr=self._unit_repr, timeout_s=10)
        rw.add_nodes(upload_ready_conditions)
        await rw.prepare()
        await self._api.set_parallel(elf_nodes)
        await rw.wait()

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        return []

    async def start_execution(self, with_pipeliner: bool):
        pass
