# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy
from numpy import typing as npt

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_1_4_0 import IO, Initialization
from laboneq.controller.recipe_enums import DIOConfigType, ReferenceClockSource
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1
REFERENCE_CLOCK_SOURCE_ZSYNC = 2


class DeviceSHFSG(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFSG8"
        self.dev_opts = []
        self._channels = 8
        self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        self._wait_for_awgs = True
        self._emit_trigger = False

    @property
    def dev_repr(self) -> str:
        if self._get_option("is_qc"):
            return f"SHFQC/SG:{self._get_option('serial')}"
        return f"SHFSG:{self._get_option('serial')}"

    def _process_dev_opts(self):
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
                self._logger.warning(
                    "%s: No valid channel option found, installed options: [%s]. "
                    "Assuming 2ch device.",
                    self.dev_repr,
                    ", ".join(self.dev_opts),
                )
                self._channels = 2
                self._output_to_synth_map = [1, 1]
        else:
            self._logger.warning(
                "%s: Unknown device type '%s', assuming SHFSG4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
            self._output_to_synth_map = [0, 1, 2, 3]

    def _get_sequencer_type(self) -> str:
        return "sg"

    def _get_sequencer_path_patterns(self) -> dict:
        return {
            "elf": "/{serial}/sgchannels/{index}/awg/elf/data",
            "progress": "/{serial}/sgchannels/{index}/awg/elf/progress",
            "enable": "/{serial}/sgchannels/{index}/awg/enable",
            "ready": "/{serial}/sgchannels/{index}/awg/ready",
        }

    def _get_num_awgs(self):
        return self._channels

    def _validate_range(self, io: IO.Data):
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
            self._logger.warning(
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
    ) -> Optional[int]:
        if previously_allocated >= 8:
            return None
        return previously_allocated

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/sgchannels/{channel}/oscs/{index}/freq"

    def _nodes_to_monitor_impl(self) -> List[str]:
        nodes = []
        for awg in range(self._get_num_awgs()):
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/enable")
            nodes.append(f"/{self.serial}/sgchannels/{awg}/awg/ready")
        return nodes

    def collect_execution_nodes(self):
        self._logger.debug("Starting execution...")
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{awg_index}/awg/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_start_execution_nodes(self):
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

    def conditions_for_execution_ready(self) -> Dict[str, Any]:
        conditions: Dict[str, Any] = {}
        if self._wait_for_awgs:
            for awg_index in self._allocated_awgs:
                conditions[f"/{self.serial}/sgchannels/{awg_index}/awg/enable"] = 1
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType
    ) -> Dict[str, Any]:
        conditions: Dict[str, Any] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/sgchannels/{awg_index}/awg/enable"] = 0
        return conditions

    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeSetAction]:
        self._logger.debug("%s: Initializing device...", self.dev_repr)

        nodes_to_initialize_output: List[DaqNodeSetAction] = []

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

            if output.port_delay is not None:
                if output.port_delay != 0:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}'s output does not support port delay"
                    )
                self._logger.info(
                    "%s's output port delay should be set to None, not 0", self.dev_repr
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

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        nodes_to_initialize_measurement = []

        center_frequencies: Dict[int, IO.Data] = {}

        def get_synth_idx(io: IO.Data):
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
            if io.port_mode is None or io.port_mode == "RF":
                nodes_to_initialize_measurement.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{io.channel}/output/rflfpath",
                        1,  # RF
                    )
                )
                nodes_to_initialize_measurement.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/synthesizers/{get_synth_idx(io)}/centerfreq",
                        io.lo_frequency,
                    )
                )
            else:
                nodes_to_initialize_measurement.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{io.channel}/output/rflfpath",
                        0,  # LF
                    )
                )
                nodes_to_initialize_measurement.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/sgchannels/{io.channel}/digitalmixer/centerfreq",
                        io.lo_frequency,
                    )
                )

        return nodes_to_initialize_measurement

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> List[DaqNodeAction]:
        self._logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        nodes_to_configure_triggers = []

        dio_mode = initialization.config.dio_mode

        if dio_mode == DIOConfigType.ZSYNC_DIO:
            pass
        elif dio_mode in (DIOConfigType.HDAWG_LEADER, DIOConfigType.HDAWG):
            # standalone SHFSG or SHFQC
            self._wait_for_awgs = False
            ntc = []
            if not self._get_option("qc_with_qa"):
                # otherwise, the QA will initialize the nodes
                self._emit_trigger = True
                clock_source = initialization.config.reference_clock_source
                ntc += [
                    (
                        "system/clocks/referenceclock/in/source",
                        REFERENCE_CLOCK_SOURCE_INTERNAL
                        if clock_source
                        and clock_source.value == ReferenceClockSource.INTERNAL.value
                        else REFERENCE_CLOCK_SOURCE_EXTERNAL,
                    ),
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
                if initialization.awgs is not None and self._get_option("qc_with_qa"):
                    awg = next(
                        (awg for awg in initialization.awgs if awg.awg == awg_index),
                        None,
                    )
                    if (
                        awg is not None
                        and awg.qa_signal_id is not None
                        and awg.command_table_match_offset is not None
                    ):
                        # Internal feedback requested in the recipe
                        matching_integrator = next(
                            (
                                i
                                for i in recipe_data.recipe.experiment.integrator_allocations
                                if i.signal_id == awg.qa_signal_id
                            ),
                            None,
                        )
                        if (
                            matching_integrator is None
                            or len(matching_integrator.channels) != 1
                        ):
                            raise LabOneQControllerException(
                                f"{self.dev_repr}: Internal error - can't find integrator config "
                                f"for mapped QA signal {awg.qa_signal_id}, that is suitable for "
                                f"the feedback configuration."
                            )

                        shift = matching_integrator.channels[0]
                        offset = awg.command_table_match_offset
                        ntc += [
                            (
                                f"sgchannels/{awg_index}/awg/intfeedback/direct/shift",
                                shift,
                            ),
                            (
                                f"sgchannels/{awg_index}/awg/intfeedback/direct/mask",
                                0b1,
                            ),
                            (
                                f"sgchannels/{awg_index}/awg/intfeedback/direct/offset",
                                offset,
                            ),
                        ]

            nodes_to_configure_triggers = [
                DaqNodeSetAction(self._daq, f"/{self.serial}/{node}", v)
                for node, v in ntc
            ]
        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type SHFSG."
            )

        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            marker_path = f"/{self.serial}/sgchannels/{awg_index}/marker"
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"{marker_path}/source", 0),
            )
        return nodes_to_configure_triggers

    def add_command_table_header(self, body: dict) -> Dict:
        return {
            "$schema": "https://docs.zhinst.com/shfsg/commandtable/v1_1/schema",
            "header": {"version": "1.1.0"},
            "table": body,
        }

    def command_table_path(self, awg_index: int) -> str:
        return f"/{self.serial}/sgchannels/{awg_index}/awg/commandtable/"

    def configure_as_leader(self, initialization: Initialization.Data):
        raise LabOneQControllerException("SHFSG cannot be configured as leader")

    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        if self._get_option("qc_with_qa"):
            return []  # QC follower config is done over it's QA part

        dio_mode = initialization.config.dio_mode
        self._logger.debug("%s: Configuring as a follower...", self.dev_repr)

        nodes_to_configure_as_follower = []

        if dio_mode == DIOConfigType.ZSYNC_DIO:
            self._logger.debug(
                "%s: Configuring reference clock to use ZSYNC as a reference...",
                self.dev_repr,
            )
            self._switch_reference_clock(source=2, expected_freqs=100e6)
        elif dio_mode == DIOConfigType.HDAWG_LEADER:
            # standalone
            pass
        elif dio_mode == DIOConfigType.HDAWG:
            # standalone as part of an SHFQC with active QA part
            pass
        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type SHFSG."
            )

        return nodes_to_configure_as_follower

    def collect_reset_nodes(self) -> List[DaqNodeAction]:
        reset_nodes = super().collect_reset_nodes()
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/*/awg/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        return reset_nodes
