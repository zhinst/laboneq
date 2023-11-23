# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy
from numpy import typing as npt

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_shf_base import DeviceSHFBase
from laboneq.controller.devices.device_zi import (
    SequencerPaths,
    delay_to_rounded_samples,
)
from laboneq.controller.devices.zi_node_monitor import NodeControlBase
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import IO, Initialization, TriggeringMode, ParameterUID


_logger = logging.getLogger(__name__)

SAMPLE_FREQUENCY_HZ = 2.0e9
DELAY_NODE_GRANULARITY_SAMPLES = 1
DELAY_NODE_MAX_SAMPLES = round(124e-9 * SAMPLE_FREQUENCY_HZ)
OPT_OUTPUT_ROUTER_ADDER = "RTR"


class DeviceSHFSG(AwgPipeliner, DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFSG8"
        self.dev_opts = []
        # Available number of full output channels (Front panel outputs).
        self._outputs = 8
        # Available number of output channels (RTR option can extend these with internal channels on certain devices)
        self._channels = self._outputs
        self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        self._wait_for_awgs = True
        self._emit_trigger = False
        self.pipeliner_set_node_base(f"/{self.serial}/sgchannels")
        self._has_opt_rtr = False

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/SG:{self.serial}"
        return f"SHFSG:{self.serial}"

    @property
    def is_secondary(self) -> bool:
        return self.options.qc_with_qa

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        if OPT_OUTPUT_ROUTER_ADDER in self.dev_opts:
            self._has_opt_rtr = True
        if self.dev_type == "SHFSG8":
            self._outputs = 8
            self._channels = self._outputs
            self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        elif self.dev_type == "SHFSG4":
            self._outputs = 4
            self._channels = self._outputs
            self._output_to_synth_map = [0, 1, 2, 3]
            if self._has_opt_rtr:
                self._channels = 8
        elif self.dev_type == "SHFQC":
            # Different numbering on SHFQC - index 0 are QA synths
            if "QC2CH" in self.dev_opts:
                self._outputs = 2
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1]
            elif "QC4CH" in self.dev_opts:
                self._outputs = 4
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1, 2, 2]
            elif "QC6CH" in self.dev_opts:
                self._outputs = 6
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1, 2, 2, 3, 3]
            else:
                _logger.warning(
                    "%s: No valid channel option found, installed options: [%s]. "
                    "Assuming 2ch device.",
                    self.dev_repr,
                    ", ".join(self.dev_opts),
                )
                self._outputs = 2
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1]
            if self._has_opt_rtr:
                self._channels = 6
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFSG4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._outputs = 4
            self._channels = self._outputs
            if self._has_opt_rtr:
                self._channels = 8
            self._output_to_synth_map = [0, 1, 2, 3]

    def _is_full_channel(self, channel: int) -> bool:
        return channel < self._outputs

    def _is_internal_channel(self, channel: int) -> bool:
        return self._outputs < channel < self._channels

    def _get_sequencer_type(self) -> str:
        return "sg"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        return SequencerPaths(
            elf=f"/{self.serial}/sgchannels/{index}/awg/elf/data",
            progress=f"/{self.serial}/sgchannels/{index}/awg/elf/progress",
            enable=f"/{self.serial}/sgchannels/{index}/awg/enable",
            ready=f"/{self.serial}/sgchannels/{index}/awg/ready",
        )

    def _get_num_awgs(self) -> int:
        return self._channels

    def _validate_range(self, io: IO):
        if io.range is None:
            return
        range_list = numpy.array(
            [-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=numpy.float64
        )
        label = "Output"

        if io.range_unit not in (None, "dBm"):
            raise LabOneQControllerException(
                f"{label} range of device {self.dev_repr} is specified in "
                f"units of {io.range_unit}. Units must be 'dBm'."
            )
        if not any(numpy.isclose([io.range] * len(range_list), range_list)):
            _logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. "
                "Nearest allowed range will be used.",
                self.dev_repr,
                label,
                io.channel,
                io.range,
                range_list,
            )

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> int | None:
        if previously_allocated >= 8:
            return None
        return previously_allocated

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/sgchannels/{channel}/oscs/{index}/freq"

    def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        channels_to_disable: list[DaqNodeSetAction] = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{ch}/output/on",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for ch in range(self._outputs)
            if (ch in outputs) != invert
        ]
        return channels_to_disable

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/enable")
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/ready")
            nodes.extend(self.pipeliner_control_nodes(awg))
        return nodes

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        if self.is_secondary:
            return []  # QA will initialize the nodes
        else:
            return super().clock_source_control_nodes()

    def collect_execution_nodes(self, with_pipeliner: bool) -> list[DaqNodeAction]:
        if with_pipeliner:
            return self.pipeliner_collect_execution_nodes()

        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{awg_index}/awg/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeAction]:
        hw_sync = with_pipeliner and has_awg_in_use and not self.is_secondary
        nodes = []
        if hw_sync and self._emit_trigger:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/synchronization/enable",
                    1,  # enable
                )
            )
        if hw_sync and not self._emit_trigger:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    1,  # external
                )
            )
        return nodes

    def collect_internal_start_execution_nodes(self):
        if self._emit_trigger:
            return [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/enable",
                    1,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        return []

    def conditions_for_execution_ready(self, with_pipeliner: bool) -> dict[str, Any]:
        if not self._wait_for_awgs:
            return {}

        if with_pipeliner:
            return self.pipeliner_conditions_for_execution_ready()

        return {
            f"/{self.serial}/sgchannels/{awg_index}/awg/enable": 1
            for awg_index in self._allocated_awgs
        }

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, Any]:
        if with_pipeliner:
            return self.pipeliner_conditions_for_execution_done()

        return {
            f"/{self.serial}/sgchannels/{awg_index}/awg/enable": 0
            for awg_index in self._allocated_awgs
        }

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)

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
                        f"with {self._outputs} channels. Verify your device setup."
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
            route_attrs = [
                AttributeName.OUTPUT_ROUTE_1,
                AttributeName.OUTPUT_ROUTE_2,
                AttributeName.OUTPUT_ROUTE_3,
            ]
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
                    name=route_attrs[idx],
                    index=io.channel,
                    value_or_param=router,
                )
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

    def _collect_output_router_nodes(
        self,
        target: int,
        source: int,
        router_idx: int,
        amplitude: float | None = None,
        phase: float | None = None,
    ) -> list[DaqNodeAction]:
        nodes = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{target}/outputrouter/enable",
                1,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{target}/outputrouter/routes/{router_idx}/enable",
                1,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{target}/outputrouter/routes/{router_idx}/source",
                source,
            ),
        ]
        if amplitude is not None:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{target}/outputrouter/routes/{router_idx}/amplitude",
                    amplitude,
                )
            )
        if phase is not None:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{target}/outputrouter/routes/{router_idx}/phase",
                    phase * 180 / numpy.pi,
                )
            )
        # Turn output router on source channel, device will make sync them internally.
        if self._is_full_channel(source):
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{source}/outputrouter/enable",
                    1,
                )
            )
        return nodes

    def _collect_output_router_initialization_nodes(
        self, outputs: list[IO]
    ) -> list[DaqNodeSetAction]:
        # TODO: Check for overflow warnings: sgchannels/n/outputrouter/overflowcount
        nodes: list[DaqNodeSetAction] = []
        active_output_routers: set[int] = set()
        for output in outputs:
            if output.routed_outputs and self._has_opt_rtr is False:
                msg = f"{self.dev_repr}: Output router and adder requires '{OPT_OUTPUT_ROUTER_ADDER}' option on SHFSG / SHFQC devices."
                raise LabOneQControllerException(msg)
            for idx, route in enumerate(output.routed_outputs):
                if not self._is_full_channel(output.channel):
                    msg = f"{self.dev_repr}: Outputs can only be routed to device front panel outputs. Invalid channel: {output.channel}"
                    raise LabOneQControllerException(msg)
                # We enable the router on both the source and destination channels, so that the delay matches between them.
                active_output_routers.add(output.channel)
                active_output_routers.add(route.from_channel)
                nodes.extend(
                    self._collect_output_router_nodes(
                        target=output.channel,
                        source=route.from_channel,
                        router_idx=idx,
                        amplitude=route.amplitude
                        if not isinstance(route.amplitude, ParameterUID)
                        else None,
                        phase=route.phase
                        if not isinstance(route.phase, ParameterUID)
                        else None,
                    )
                )
            # Disable routes which are not used.
            if output.routed_outputs:
                routes_to_disable = [1, 2]
                [
                    nodes.append(
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/sgchannels/{output.channel}/outputrouter/routes/{route_disable}/enable",
                            0,
                            caching_strategy=CachingStrategy.NO_CACHE,
                        ),
                    )
                    for route_disable in routes_to_disable[
                        len(output.routed_outputs) - 1 :
                    ]
                ]
        # Disable existing, but unconfigured output routers to make sure they
        # do not have signal delay introduces by the setting.
        if self._has_opt_rtr:
            for output in outputs:
                if (
                    output.channel not in active_output_routers
                    and self._is_full_channel(output.channel)
                ):
                    nodes.extend(
                        [
                            DaqNodeSetAction(
                                self._daq,
                                f"/{self.serial}/sgchannels/{output.channel}/outputrouter/enable",
                                0,
                                caching_strategy=CachingStrategy.NO_CACHE,
                            ),
                            DaqNodeSetAction(
                                self._daq,
                                f"/{self.serial}/sgchannels/{output.channel}/outputrouter/routes/*/enable",
                                0,
                                caching_strategy=CachingStrategy.NO_CACHE,
                            ),
                        ]
                    )
        return nodes

    def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nodes_to_initialize_output: list[DaqNodeSetAction] = []

        outputs = initialization.outputs or []
        for output in outputs:
            self._warn_for_unsupported_param(
                output.offset is None or output.offset == 0,
                "voltage_offsets",
                output.channel,
            )
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )

            self._allocated_awgs.add(output.channel)
            if self._is_full_channel(output.channel):
                nodes_to_initialize_output.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{output.channel}/output/on",
                        1 if output.enable else 0,
                    )
                )
                if output.range is not None:
                    self._validate_range(output)
                    nodes_to_initialize_output.append(
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/sgchannels/{output.channel}/output/range",
                            output.range,
                        )
                    )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{output.channel}/awg/single",
                    1,
                )
            )

            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{output.channel}/awg/modulation/enable",
                    1 if output.modulation else 0,
                )
            )

            if self._is_full_channel(output.channel):
                if output.marker_mode is None or output.marker_mode == "TRIGGER":
                    nodes_to_initialize_output.append(
                        DaqNodeSetAction(
                            self.daq,
                            f"/{self.serial}/sgchannels/{output.channel}/marker/source",
                            0,
                        )
                    )
                elif output.marker_mode == "MARKER":
                    nodes_to_initialize_output.append(
                        DaqNodeSetAction(
                            self.daq,
                            f"/{self.serial}/sgchannels/{output.channel}/marker/source",
                            4,
                        )
                    )
                else:
                    raise ValueError(
                        f"Marker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel} on SHFSG {self.serial}"
                    )

                # set trigger delay to 0
                nodes_to_initialize_output.append(
                    DaqNodeSetAction(
                        self.daq,
                        f"/{self.serial}/sgchannels/{output.channel}/trigger/delay",
                        0.0,
                    )
                )

                nodes_to_initialize_output.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{output.channel}/output/rflfpath",
                        1  # RF
                        if output.port_mode is None or output.port_mode == "rf"
                        else 0,  # LF
                    )
                )
        nodes_to_initialize_output.extend(
            self._collect_output_router_initialization_nodes(outputs)
        )
        osc_selects = {
            ch: osc.index for osc in self._allocated_oscs for ch in osc.channels
        }
        # SHFSG has only one "sine" per channel, therefore "sines" index is hard-coded to 0.
        # If multiple oscillators are assigned to a channel, it indicates oscillator switching
        # via the command table, and the oscselect node is ignored. Therefore it can be set to
        # any oscillator.
        for ch, osc_idx in osc_selects.items():
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/0/oscselect",
                    osc_idx,
                )
            )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/0/harmonic",
                    1,
                )
            )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/0/phaseshift",
                    0,
                )
            )
        return nodes_to_initialize_output

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        nodes_to_set = super().collect_prepare_nt_step_nodes(attributes, recipe_data)

        for synth_idx in set(self._output_to_synth_map):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.SG_SYNTH_CENTER_FREQ, synth_idx)]
            )
            if synth_cf_updated:
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/synthesizers/{synth_idx}/centerfreq",
                        synth_cf,
                    )
                )

        for ch in range(self._outputs):
            [dig_mixer_cf], dig_mixer_cf_updated = attributes.resolve(
                keys=[(AttributeName.SG_DIG_MIXER_CENTER_FREQ, ch)]
            )
            if dig_mixer_cf_updated:
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{ch}/digitalmixer/centerfreq",
                        dig_mixer_cf,
                    )
                )
            [scheduler_port_delay, port_delay], updated = attributes.resolve(
                keys=[
                    (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.OUTPUT_PORT_DELAY, ch),
                ]
            )
            if updated and scheduler_port_delay is not None:
                output_delay = scheduler_port_delay + (port_delay or 0.0)
                output_delay_rounded = (
                    delay_to_rounded_samples(
                        channel=ch,
                        dev_repr=self.dev_repr,
                        delay=output_delay,
                        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
                    )
                    / SAMPLE_FREQUENCY_HZ
                )

                nodes_to_set.append(
                    DaqNodeSetAction(
                        daq=self.daq,
                        path=f"/{self.serial}/sgchannels/{ch}/output/delay",
                        value=output_delay_rounded,
                    )
                )
            for route_idx, (route, ampl, phase) in enumerate(
                (
                    [
                        (AttributeName.OUTPUT_ROUTE_1, ch),
                        (AttributeName.OUTPUT_ROUTE_1_AMPLITUDE, ch),
                        (AttributeName.OUTPUT_ROUTE_1_PHASE, ch),
                    ],
                    [
                        (AttributeName.OUTPUT_ROUTE_2, ch),
                        (AttributeName.OUTPUT_ROUTE_2_AMPLITUDE, ch),
                        (AttributeName.OUTPUT_ROUTE_2_PHASE, ch),
                    ],
                    [
                        (AttributeName.OUTPUT_ROUTE_3, ch),
                        (AttributeName.OUTPUT_ROUTE_3_AMPLITUDE, ch),
                        (AttributeName.OUTPUT_ROUTE_3_PHASE, ch),
                    ],
                )
            ):
                (
                    [output_router, route_amplitude, route_phase],
                    output_route_updated,
                ) = attributes.resolve(keys=[route, ampl, phase])
                if (
                    output_route_updated
                    and route_amplitude is not None
                    or route_phase is not None
                ):
                    nodes_to_set.extend(
                        self._collect_output_router_nodes(
                            target=ch,
                            source=output_router.from_channel,
                            router_idx=route_idx,
                            amplitude=route_amplitude,
                            phase=route_phase,
                        )
                    )
        return nodes_to_set

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ):
        return DaqNodeSetAction(
            self._daq,
            f"/{self.serial}/sgchannels/{awg_index}/awg/waveform/waves/{wave_index}",
            waveform,
            filename=filename,
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        ntc = []

        for awg_key, awg_config in recipe_data.awg_configs.items():
            if awg_key.device_uid != initialization.device_uid:
                continue

            if awg_config.source_feedback_register is None:
                continue

            if awg_config.source_feedback_register == "local" and self.is_secondary:
                # local feedback
                ntc.extend(
                    [
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/intfeedback/direct/shift",
                            awg_config.register_selector_shift,
                        ),
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/intfeedback/direct/mask",
                            awg_config.register_selector_bitmask,
                        ),
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/intfeedback/direct/offset",
                            awg_config.command_table_match_offset,
                        ),
                    ]
                )
            else:
                # global feedback
                ntc.extend(
                    [
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/diozsyncswitch",
                            1,  # ZSync Trigger
                        ),
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/zsync/register/shift",
                            awg_config.register_selector_shift,
                        ),
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/zsync/register/mask",
                            awg_config.register_selector_bitmask,
                        ),
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/zsync/register/offset",
                            awg_config.command_table_match_offset,
                        ),
                    ]
                )

        triggering_mode = initialization.config.triggering_mode
        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode in (
            TriggeringMode.DESKTOP_LEADER,
            TriggeringMode.INTERNAL_FOLLOWER,
        ):
            # standalone SHFSG or SHFQC
            self._wait_for_awgs = False
            if not self.is_secondary:
                # otherwise, the QA will initialize the nodes
                self._emit_trigger = True
                ntc += [
                    ("system/internaltrigger/enable", 0),
                    ("system/internaltrigger/repetitions", 1),
                ]
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                ntc += [
                    # Rise
                    (f"sgchannels/{awg_index}/awg/auxtriggers/0/slope", 1),
                    # Internal trigger
                    (f"sgchannels/{awg_index}/awg/auxtriggers/0/channel", 8),
                ]
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFSG."
            )

        nodes_to_configure_triggers = [
            DaqNodeSetAction(self._daq, f"/{self.serial}/{node}", v) for node, v in ntc
        ]
        return nodes_to_configure_triggers

    def add_command_table_header(self, body: dict) -> dict:
        return {
            "$schema": "https://docs.zhinst.com/shfsg/commandtable/v1_1/schema",
            "header": {"version": "1.1.0"},
            "table": body,
        }

    def command_table_path(self, awg_index: int) -> str:
        return f"/{self.serial}/sgchannels/{awg_index}/awg/commandtable/"

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        reset_nodes = super().collect_reset_nodes()
        # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
        reset_nodes.extend(self.pipeliner_reset_nodes())
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/*/awg/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        if not self.is_secondary:
            reset_nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    0,  # internal
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )
            if self.options.is_qc:
                reset_nodes.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/system/internaltrigger/synchronization/enable",
                        0,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                )
        return reset_nodes
