# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
    ResponseWaiterAsync,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.core_base import SHFBaseProtocol, SHFChannelBase
from laboneq.controller.devices.device_shf_base import (
    OPT_OUTPUT_ROUTER_ADDER,
    check_synth_frequency,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import delay_to_rounded_samples
from laboneq.controller.recipe_processor import (
    DeviceRecipeData,
    RecipeData,
    get_elf,
    get_initialization_by_device_uid,
    prepare_command_table,
    prepare_waves,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.data.recipe import IO, Initialization, NtStepKey
from laboneq.data.scheduled_experiment import ArtifactsCodegen, ScheduledExperiment

_logger = logging.getLogger(__name__)

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
    awg_sequencer_status: str
    awg_command_table: str
    osc_freq: list[str]
    busy: str


class SGChannel(SHFChannelBase):
    def __init__(
        self,
        *,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        core_index: int,
        repr_base: str,
        is_plus: bool,
        has_opt_rtr: bool,
        is_full: bool,
        is_standalone: bool,
    ):
        super().__init__(
            api=api,
            subscriber=subscriber,
            device_uid=device_uid,
            serial=serial,
            core_index=core_index,
        )
        self._node_base = f"/{serial}/sgchannels/{core_index}"
        self._unit_repr = f"{repr_base}:sg{core_index}"
        self._is_plus = is_plus
        self._has_opt_rtr = has_opt_rtr
        self._is_full = is_full
        self._is_standalone = is_standalone
        self._pipeliner = AwgPipeliner(self._node_base, f"SG{core_index}")
        # TODO(2K): Do all SG channels always have 8 oscillators?
        self.nodes = SGChannelNodes(
            output_on=f"{self._node_base}/output/on",
            awg_elf_data=f"{self._node_base}/awg/elf/data",
            awg_elf_progress=f"{self._node_base}/awg/elf/progress",
            awg_enable=f"{self._node_base}/awg/enable",
            awg_ready=f"{self._node_base}/awg/ready",
            awg_sequencer_status=f"{self._node_base}/awg/sequencer/status",
            awg_command_table=f"{self._node_base}/awg/commandtable",
            osc_freq=[f"{self._node_base}/oscs/{i}/freq" for i in range(8)],
            busy=f"{self._node_base}/busy",
        )

    @property
    def is_full(self) -> bool:
        return self._is_full

    @property
    def pipeliner(self) -> AwgPipeliner:
        return self._pipeliner

    async def disable_output(self, outputs: set[int], invert: bool):
        if (self._core_index in outputs) != invert:
            await self._api.set_parallel(
                NodeCollector.one(self.nodes.output_on, 0, cache=False)
            )

    @staticmethod
    def reset_to_idle_nodes(
        *, base: str, is_qc: bool, is_secondary: bool, has_opt_rtr: bool
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{base}/")
        # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
        nc.add("sgchannels/*/pipeliner/reset", 1, cache=False)
        nc.add("sgchannels/*/pipeliner/mode", 0, cache=False)  # off
        nc.add("sgchannels/*/synchronization/enable", 0, cache=False)
        nc.barrier()
        nc.add("sgchannels/*/awg/enable", 0, cache=False)
        if not is_secondary:
            nc.add(
                "system/synchronization/source",
                0,  # internal
                cache=False,
            )
            if is_qc:
                nc.add("system/internaltrigger/synchronization/enable", 0, cache=False)
        if has_opt_rtr:
            # Disable any previously configured output routers to make sure they
            # do not introduce signal delay or unexpected signal paths.
            nc.add("sgchannels/*/outputrouter/enable", 0, cache=False)
            nc.add("sgchannels/*/outputrouter/routes/*/enable", 0, cache=False)
        return nc

    def allocate_resources(self):
        self._pipeliner._reload_tracker.reset()

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
        router_config = device_recipe_data.sgchannels[self._core_index].router_config
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

    async def apply_core_initialization(self, device_recipe_data: DeviceRecipeData):
        sg_ch_recipe_data = device_recipe_data.sgchannels.get(self._core_index)
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
            keys=[(AttributeName.SG_DIG_MIXER_CENTER_FREQ, self._core_index)]
        )
        if dig_mixer_cf_updated:
            nc.add("digitalmixer/centerfreq", dig_mixer_cf)
        [scheduler_port_delay, port_delay], updated = attributes.resolve(
            keys=[
                (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, self._core_index),
                (AttributeName.OUTPUT_PORT_DELAY, self._core_index),
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
            amplitude_key = (route_amplitude_keys[router_idx], self._core_index)
            phase_key = (route_phase_keys[router_idx], self._core_index)
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

    async def load_awg_program(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        artifacts = recipe_data.get_artifacts(ArtifactsCodegen)
        rt_execution_info = recipe_data.rt_execution_info

        elf_nodes = NodeCollector()
        upload_ready_conditions: dict[str, Any] = {}

        if rt_execution_info.is_chunked:
            # enable pipeliner
            elf_nodes.extend(self._pipeliner.prepare_for_upload())

        for pipeliner_job in range(rt_execution_info.chunk_count_or_1):
            effective_nt_step = (
                NtStepKey(indices=tuple([*nt_step.indices, pipeliner_job]))
                if rt_execution_info.is_chunked
                else nt_step
            )
            rt_exec_step = next(
                (
                    r
                    for r in recipe_data.recipe.realtime_execution_init
                    if r.device_id == self._device_uid
                    and r.awg_index == self._core_index
                    and r.nt_step == effective_nt_step
                ),
                None,
            )

            if rt_execution_info.is_chunked:
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
                        path=f"{self._node_base}/awg/waveform/waves/{wave.index}",
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

            if rt_execution_info.is_chunked:
                elf_nodes.extend(self._pipeliner.commit())

        if rt_execution_info.is_chunked:
            upload_ready_conditions.update(self._pipeliner.ready_conditions())

        rw = ResponseWaiterAsync(api=self._api, dev_repr=self._unit_repr, timeout_s=10)
        rw.add_nodes(upload_ready_conditions)
        await rw.prepare()
        await self._api.set_parallel(elf_nodes)
        await rw.wait()

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        if self._has_opt_rtr:
            warning_nodes.append(
                (
                    f"{self._node_base}/outputrouter/overflowcount",
                    f"Channel {self._core_index} Output Router overflow count",
                )
            )
        warning_nodes.append(
            (
                f"{self._node_base}/output/overrangecount",
                f"Channel {self._core_index} Output overrange count",
            )
        )
        return warning_nodes

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if self._is_standalone:
            return {}
        if with_pipeliner:
            return self._pipeliner.conditions_for_execution_ready()
        return {
            self.nodes.awg_sequencer_status: (
                4,
                f"AWG {self._core_index} didn't start.",
            )
        }

    async def start_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"{self._node_base}/")
        if with_pipeliner:
            nc.extend(self.pipeliner.collect_execution_nodes())
        else:
            nc.add("awg/enable", 1, cache=False)
        await self._api.set_parallel(nc)

    def conditions_for_execution_done(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if with_pipeliner:
            return self.pipeliner.conditions_for_execution_done()
        else:
            return {
                self.nodes.awg_enable: (
                    0,
                    f"AWG {self._core_index} didn't stop. Missing start trigger? Check ZSync.",
                )
            }

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"{self._node_base}/")
        if with_pipeliner:
            nc.extend(self.pipeliner.reset_nodes())
        await self._api.set_parallel(nc)


class SHFSGProtocol(SHFBaseProtocol, Protocol):
    @property
    def _outputs(self) -> int: ...

    @property
    def _channels(self) -> int: ...

    @property
    def _sgchannels(self) -> list[SGChannel]: ...

    @property
    def _output_to_synth_map(self) -> list[int]: ...

    @property
    def _has_opt_rtr(self) -> bool: ...

    def _is_internal_channel(self, channel: int) -> bool: ...

    # Methods defined in SHFSGMixIn
    def _validate_range_shfsg(self, io: IO): ...
    def _is_full_channel(self, channel: int) -> bool: ...


class SHFSGMixIn:
    def _is_full_channel(self: SHFSGProtocol, channel: int) -> bool:
        return channel < self._outputs

    def _is_internal_channel(self: SHFSGProtocol, channel: int) -> bool:
        return channel < self._channels

    def _validate_range_shfsg(self: SHFSGProtocol, io: IO):
        if io.range is None:
            return
        range_list = np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=np.float64)
        label = "Output"

        if io.range_unit not in (None, "dBm"):
            raise LabOneQControllerException(
                f"{label} range of device {self.dev_repr} is specified in "
                f"units of {io.range_unit}. Units must be 'dBm'."
            )
        if not any(np.isclose([io.range] * len(range_list), range_list)):
            _logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. "
                "Nearest allowed range will be used.",
                self.dev_repr,
                label,
                io.channel,
                io.range,
                range_list,
            )

    def _validate_scheduled_experiment_shfsg(
        self: SHFSGProtocol,
        scheduled_experiment: ScheduledExperiment,
    ):
        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, self.uid
        )
        if initialization is None:
            return
        for output in initialization.outputs:
            self._validate_range_shfsg(output)
            self._warn_for_unsupported_param(
                output.offset is None or output.offset == 0,
                "voltage_offsets",
                output.channel,
            )
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )
            if output.marker_mode not in (None, "TRIGGER", "MARKER"):
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Marker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel}"
                )
            if output.enable_output_mute and not self._is_plus:
                _logger.warning(
                    f"{self.dev_repr}: Device output muting is enabled, but the device is not"
                    " SHF+ and therefore no muting will happen. It is suggested to disable it."
                )
            if len(output.routed_outputs) > 0:
                if not self._has_opt_rtr:
                    msg = f"{self.dev_repr}: Output router and adder requires '{OPT_OUTPUT_ROUTER_ADDER}' option on SHFSG / SHFQC devices."
                    raise LabOneQControllerException(msg)
                if not self._is_full_channel(output.channel):
                    msg = f"{self.dev_repr}: Outputs can only be routed to device front panel outputs. Invalid channel: {output.channel}"
                    raise LabOneQControllerException(msg)

    def _pre_process_attributes_shfsg(
        self: SHFSGProtocol,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        center_frequencies: dict[int, IO] = {}

        def get_synth_idx(io: IO):
            synth_idx = self._output_to_synth_map[io.channel]
            prev_io = center_frequencies.get(synth_idx)
            if prev_io is None:
                center_frequencies[synth_idx] = io
            elif prev_io.lo_frequency != io.lo_frequency:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator frequency mismatch between outputs "
                    f"{prev_io.channel} and {io.channel} sharing synthesizer {synth_idx}: "
                    f"{prev_io.lo_frequency} != {io.lo_frequency}"
                )
            return synth_idx

        ios = initialization.outputs or []
        for io in ios:
            if not self._is_full_channel(io.channel):
                if not self._is_internal_channel(io.channel):
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Attempt to configure channel {io.channel} on a device "
                        f"with {self._channels} channels. Verify your device setup."
                    )
                continue
            if io.lo_frequency is None:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator for channel {io.channel} is required, "
                    f"but is not provided."
                )
            if io.port_mode is None or io.port_mode == "rf":
                yield DeviceAttribute(
                    name=AttributeName.SG_SYNTH_CENTER_FREQ,
                    index=get_synth_idx(io),
                    value_or_param=io.lo_frequency,
                )
            else:
                yield DeviceAttribute(
                    name=AttributeName.SG_DIG_MIXER_CENTER_FREQ,
                    index=io.channel,
                    value_or_param=io.lo_frequency,
                )
            route_ampl_attrs = [
                AttributeName.OUTPUT_ROUTE_1_AMPLITUDE,
                AttributeName.OUTPUT_ROUTE_2_AMPLITUDE,
                AttributeName.OUTPUT_ROUTE_3_AMPLITUDE,
            ]
            route_phase_attrs = [
                AttributeName.OUTPUT_ROUTE_1_PHASE,
                AttributeName.OUTPUT_ROUTE_2_PHASE,
                AttributeName.OUTPUT_ROUTE_3_PHASE,
            ]
            for idx, router in enumerate(io.routed_outputs):
                yield DeviceAttribute(
                    name=route_ampl_attrs[idx],
                    index=io.channel,
                    value_or_param=router.amplitude,
                )
                yield DeviceAttribute(
                    name=route_phase_attrs[idx],
                    index=io.channel,
                    value_or_param=router.phase,
                )

    async def _teardown_one_step_execution_shfsg(self: SHFSGProtocol):
        nc = NodeCollector(base=f"/{self.serial}/")
        # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
        # wrongly detected gaps when enabled during experiments with feedback.
        # Here we make sure that if they were enabled at `session.connect` we
        # re-enable them in case the previous experiment had feedback.
        nc.add("raw/system/awg/runtimechecks/enable", int(self._enable_runtime_checks))

        await self._api.set_parallel(nc)

    async def _set_nt_step_nodes_shfsg(
        self: SHFSGProtocol, *, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"/{self.serial}/")
        for synth_idx in set(self._output_to_synth_map):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.SG_SYNTH_CENTER_FREQ, synth_idx)]
            )
            if synth_cf_updated:
                check_synth_frequency(synth_cf, self.dev_repr, synth_idx)
                nc.add(f"synthesizers/{synth_idx}/centerfreq", synth_cf)
        await self._api.set_parallel(nc)

    async def _apply_initialization_shfsg(
        self: SHFSGProtocol, device_recipe_data: DeviceRecipeData
    ):
        # If multiple oscillators are assigned to a channel, it indicates oscillator switching
        # via the command table, and the oscselect node is ignored. Therefore it can be set to
        # any oscillator.
        nc = NodeCollector(base=f"/{self.serial}/")
        osc_selects = {
            ch: osc.index
            for osc in device_recipe_data.allocated_oscs
            for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nc.extend(self._sgchannels[ch]._collect_configure_oscillator_nodes(osc_idx))
        await self._api.set_parallel(nc)
