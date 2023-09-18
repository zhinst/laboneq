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
from laboneq.controller.devices.device_shf_base import DeviceSHFBase
from laboneq.controller.devices.device_zi import (
    SequencerPaths,
    delay_to_rounded_samples,
)
from laboneq.controller.devices.zi_node_monitor import NodeControlBase
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import IO, Initialization, TriggeringMode

_logger = logging.getLogger(__name__)

SAMPLE_FREQUENCY_HZ = 2.0e9
DELAY_NODE_GRANULARITY_SAMPLES = 1
DELAY_NODE_MAX_SAMPLES = round(124e-9 * SAMPLE_FREQUENCY_HZ)


class DeviceSHFSG(DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFSG8"
        self.dev_opts = []
        self._channels = 8
        self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        self._wait_for_awgs = True
        self._emit_trigger = False
        self._pipeliner_slot_tracker: list[int] = [0] * 8

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/SG:{self.serial}"
        return f"SHFSG:{self.serial}"

    @property
    def has_pipeliner(self) -> bool:
        return True

    @property
    def is_secondary(self) -> bool:
        return self.options.qc_with_qa

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        if self.dev_type == "SHFSG8":
            self._channels = 8
            self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        elif self.dev_type == "SHFSG4":
            self._channels = 4
            self._output_to_synth_map = [0, 1, 2, 3]
        elif self.dev_type == "SHFQC":
            # Different numbering on SHFQC - index 0 are QA synths
            if "QC2CH" in self.dev_opts:
                self._channels = 2
                self._output_to_synth_map = [1, 1]
            elif "QC4CH" in self.dev_opts:
                self._channels = 4
                self._output_to_synth_map = [1, 1, 2, 2]
            elif "QC6CH" in self.dev_opts:
                self._channels = 6
                self._output_to_synth_map = [1, 1, 2, 2, 3, 3]
            else:
                _logger.warning(
                    "%s: No valid channel option found, installed options: [%s]. "
                    "Assuming 2ch device.",
                    self.dev_repr,
                    ", ".join(self.dev_opts),
                )
                self._channels = 2
                self._output_to_synth_map = [1, 1]
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFSG4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
            self._output_to_synth_map = [0, 1, 2, 3]

    def _get_sequencer_type(self) -> str:
        return "sg"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        return SequencerPaths(
            elf=f"/{self.serial}/sgchannels/{index}/awg/elf/data",
            progress=f"/{self.serial}/sgchannels/{index}/awg/elf/progress",
            enable=f"/{self.serial}/sgchannels/{index}/awg/enable",
            ready=f"/{self.serial}/sgchannels/{index}/awg/ready",
        )

    def pipeliner_prepare_for_upload(self, index: int) -> list[DaqNodeAction]:
        self._pipeliner_slot_tracker = [0] * 8
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{index}/pipeliner/mode",
                1,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{index}/pipeliner/reset",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{index}/synchronization/enable",
                1,
            ),
        ]

    def pipeliner_commit(self, index: int) -> list[DaqNodeAction]:
        self._pipeliner_slot_tracker[index] += 1
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{index}/pipeliner/commit",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
        ]

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        max_slots = 1024  # TODO(2K): read on connect from pipeliner/maxslots
        avail_slots = max_slots - self._pipeliner_slot_tracker[index]
        return {
            f"/{self.serial}/sgchannels/{index}/pipeliner/availableslots": avail_slots
        }

    def _get_num_awgs(self):
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
        channels_to_disable: list[DaqNodeSetAction] = []
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                channels_to_disable.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{ch}/output/on",
                        0,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                )
        return channels_to_disable

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/enable")
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/ready")
            nodes.append(f"/{self.serial}/sgchannels/{awg}/pipeliner/availableslots")
            nodes.append(f"/{self.serial}/sgchannels/{awg}/pipeliner/status")
        return nodes

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        if self.is_secondary:
            return []  # QA will initialize the nodes
        else:
            return super().clock_source_control_nodes()

    def collect_execution_nodes(self, with_pipeliner: bool):
        _logger.debug("Starting execution...")
        control_unit = "pipeliner" if with_pipeliner else "awg"
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{awg_index}/{control_unit}/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_execution_setup_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeAction]:
        nodes = []
        if with_pipeliner:
            if self._emit_trigger:
                nodes.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/system/internaltrigger/synchronization/enable",
                        1,
                    )
                )
            else:
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
        conditions: dict[str, Any] = {}
        if self._wait_for_awgs:
            for awg_index in self._allocated_awgs:
                if with_pipeliner:
                    conditions[
                        f"/{self.serial}/sgchannels/{awg_index}/pipeliner/status"
                    ] = 1  # exec
                else:
                    conditions[f"/{self.serial}/sgchannels/{awg_index}/awg/enable"] = 1
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, Any]:
        conditions: dict[str, Any] = {}
        for awg_index in self._allocated_awgs:
            if with_pipeliner:
                conditions[
                    f"/{self.serial}/sgchannels/{awg_index}/pipeliner/status"
                ] = 3  # done
            else:
                conditions[f"/{self.serial}/sgchannels/{awg_index}/awg/enable"] = 0
        return conditions

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)

        center_frequencies: dict[int, IO] = {}

        def get_synth_idx(io: IO):
            if io.channel >= self._channels:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Attempt to configure channel {io.channel + 1} on a device "
                    f"with {self._channels} channels. Verify your device setup."
                )
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

    def collect_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization
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

        osc_selects = {
            ch: osc.index for osc in self._allocated_oscs for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/{osc_idx}/oscselect",
                    osc_idx,
                )
            )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/{osc_idx}/harmonic",
                    1,
                )
            )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/{ch}/sines/{osc_idx}/phaseshift",
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

        for ch in range(self._channels):
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

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        ntc = []

        for awg_key, awg_config in recipe_data.awg_configs.items():
            if awg_key.device_uid != initialization.device_uid:
                continue

            if awg_config.qa_signal_id is None:
                continue

            if awg_config.source_feedback_register is None and self.is_secondary:
                # local feedback
                ntc.extend(
                    [
                        (
                            f"sgchannels/{awg_key.awg_index}/awg/intfeedback/direct/shift",
                            awg_config.readout_result_index,
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
        reset_nodes.extend(
            [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/*/awg/enable",
                    0,
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/*/pipeliner/mode",
                    0,  # off
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sgchannels/*/synchronization/enable",
                    0,
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    0,  # internal
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/synchronization/enable",
                    0,
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
            ]
        )
        return reset_nodes
