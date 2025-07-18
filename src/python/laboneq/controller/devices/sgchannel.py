# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import delay_to_rounded_samples
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
import numpy as np


SAMPLE_FREQUENCY_HZ = 2.0e9
DELAY_NODE_GRANULARITY_SAMPLES = 1
DELAY_NODE_MAX_SAMPLES = round(124e-9 * SAMPLE_FREQUENCY_HZ)


@dataclass
class SGChannelNodes:
    output_on: str
    awg_elf_data: str
    awg_elf_progress: str
    awg_enable: str
    awg_ready: str
    osc_freq: list[str]
    busy: str


class SGChannel(ChannelBase):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        repr_base: str,
        is_plus: bool,
        has_opt_rtr: bool,
        is_full: bool,
    ):
        super().__init__(api, subscriber, device_uid, serial, channel)
        self._node_base = f"/{serial}/sgchannels/{channel}"
        self._unit_repr = f"{repr_base}:sg{channel}"
        self._is_plus = is_plus
        self._has_opt_rtr = has_opt_rtr
        self._is_full = is_full
        self._pipeliner = AwgPipeliner(self._node_base, f"SG{channel}")
        # TODO(2K): Do all SG channels always have 8 oscillators?
        self.nodes = SGChannelNodes(
            output_on=f"{self._node_base}/output/on",
            awg_elf_data=f"{self._node_base}/awg/elf/data",
            awg_elf_progress=f"{self._node_base}/awg/elf/progress",
            awg_enable=f"{self._node_base}/awg/enable",
            awg_ready=f"{self._node_base}/awg/ready",
            osc_freq=[f"{self._node_base}/oscs/{i}/freq" for i in range(8)],
            busy=f"{self._node_base}/busy",
        )

    @property
    def is_full(self) -> bool:
        return self._is_full

    @property
    def pipeliner(self) -> AwgPipeliner:
        return self._pipeliner

    def _disable_output(self) -> NodeCollector:
        return NodeCollector.one(self.nodes.output_on, 0, cache=False)

    def allocate_resources(self):
        # TODO(2K): Implement channel resources allocation for execution
        pass

    async def load_awg_program(self):
        # TODO(2K): Implement loading of the AWG program.
        return

    def _collect_configure_oscillator_nodes(self, oscillator_idx: int) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        # SHFSG has only one "sine" per channel, therefore "sines" index is hard-coded to 0.
        nc.add("sines/0/oscselect", oscillator_idx)
        nc.add("sines/0/harmonic", 1)
        nc.add("sines/0/phaseshift", 0)
        return nc

    def _collect_output_router_nodes(
        self,
        router_idx: int,
        source: int | None,
        amplitude: float | None,
        phase: float | None,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        if source is not None:
            nc.add(f"outputrouter/routes/{router_idx}/enable", 1)
            nc.add(f"outputrouter/routes/{router_idx}/source", source)
        if amplitude is not None:
            nc.add(
                f"outputrouter/routes/{router_idx}/amplitude",
                amplitude,
            )
        if phase is not None:
            nc.add(
                f"outputrouter/routes/{router_idx}/phase",
                phase * 180 / np.pi,
            )
        return nc

    def _collect_output_router_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        if not self.is_full:
            return nc
        router_config = device_recipe_data.sgchannels[self._channel].router_config
        if router_config is None:
            return nc
        nc.add("outputrouter/enable", 1)
        for router_idx, route in enumerate(router_config):
            nc.extend(
                self._collect_output_router_nodes(
                    router_idx=router_idx,
                    source=route.source,
                    amplitude=route.fixed_amplitude,
                    phase=route.fixed_phase,
                )
            )
        return nc

    async def apply_initialization(self, device_recipe_data: DeviceRecipeData):
        sg_ch_recipe_data = device_recipe_data.sgchannels.get(self._channel)
        if sg_ch_recipe_data is None:
            return

        nc = NodeCollector(base=f"{self._node_base}/")

        if sg_ch_recipe_data.output_enable is not None:
            if self.is_full:
                nc.add("output/on", 1 if sg_ch_recipe_data.output_enable else 0)
                if sg_ch_recipe_data.output_range is not None:
                    nc.add("output/range", sg_ch_recipe_data.output_range)

            nc.add("awg/single", 1)
            nc.add("awg/modulation/enable", 1)

            if not sg_ch_recipe_data.modulation:
                # We still use the output modulation (`awg/modulation/enable`), but we
                # set the oscillator to 0 Hz.
                nc.extend(self._collect_configure_oscillator_nodes(0))
                nc.extend(NodeCollector.one(self.nodes.osc_freq[0], 0.0))

            if self.is_full:
                nc.add(
                    "marker/source",
                    (
                        0  # awg_trigger0
                        if sg_ch_recipe_data.marker_source_trigger
                        else 4  # output0_marker0
                    ),
                )
                nc.add("trigger/delay", 0.0)

                nc.add(
                    "output/rflfpath",
                    (
                        1  # rf
                        if sg_ch_recipe_data.output_rf_path
                        else 0  # lf
                    ),
                )
                if self._is_plus:
                    nc.add(
                        "output/muting/enable",
                        1 if sg_ch_recipe_data.output_mute_enable else 0,
                    )

        nc.extend(self._collect_output_router_initialization_nodes(device_recipe_data))

        await self._api.set_parallel(nc)

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"{self._node_base}/")
        [dig_mixer_cf], dig_mixer_cf_updated = attributes.resolve(
            keys=[(AttributeName.SG_DIG_MIXER_CENTER_FREQ, self._channel)]
        )
        if dig_mixer_cf_updated:
            nc.add("digitalmixer/centerfreq", dig_mixer_cf)
        [scheduler_port_delay, port_delay], updated = attributes.resolve(
            keys=[
                (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, self._channel),
                (AttributeName.OUTPUT_PORT_DELAY, self._channel),
            ]
        )
        if updated and scheduler_port_delay is not None:
            output_delay = scheduler_port_delay + (port_delay or 0.0)
            output_delay_rounded = (
                delay_to_rounded_samples(
                    ch_repr=f"{self._unit_repr}",
                    delay=output_delay,
                    sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                    granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                    max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
                )
                / SAMPLE_FREQUENCY_HZ
            )

            nc.add("output/delay", output_delay_rounded)
        route_amplitude_keys = [
            AttributeName.OUTPUT_ROUTE_1_AMPLITUDE,
            AttributeName.OUTPUT_ROUTE_2_AMPLITUDE,
            AttributeName.OUTPUT_ROUTE_3_AMPLITUDE,
        ]
        route_phase_keys = [
            AttributeName.OUTPUT_ROUTE_1_PHASE,
            AttributeName.OUTPUT_ROUTE_2_PHASE,
            AttributeName.OUTPUT_ROUTE_3_PHASE,
        ]
        for router_idx in range(3):
            amplitude_key = (route_amplitude_keys[router_idx], self._channel)
            phase_key = (route_phase_keys[router_idx], self._channel)
            (
                [route_amplitude, route_phase],
                route_updated,
            ) = attributes.resolve(keys=[amplitude_key, phase_key])
            if route_updated:
                nc.extend(
                    self._collect_output_router_nodes(
                        source=None,
                        router_idx=router_idx,
                        amplitude=route_amplitude,
                        phase=route_phase,
                    )
                )
        await self._api.set_parallel(nc)

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        if self._has_opt_rtr:
            warning_nodes.append(
                (
                    f"{self._node_base}/outputrouter/overflowcount",
                    f"Channel {self._channel} Output Router overflow count",
                )
            )
        warning_nodes.append(
            (
                f"{self._node_base}/output/overrangecount",
                f"Channel {self._channel} Output overrange count",
            )
        )
        return warning_nodes

    async def start_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"{self._node_base}/")
        if with_pipeliner:
            nc.extend(self.pipeliner.collect_execution_nodes())
        else:
            nc.add("awg/enable", 1, cache=False)
        await self._api.set_parallel(nc)
