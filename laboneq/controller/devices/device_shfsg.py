# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import numpy
from typing import Dict, List, Optional
from numpy import typing as npt

from laboneq.controller.recipe_enums import DIOConfigType
from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.communication import (
    DaqNodeSetAction,
    DaqNodeWaitAction,
    CachingStrategy,
)
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.recipe_1_4_0 import Initialization, IO
from laboneq.controller.recipe_enums import ReferenceClockSource
from laboneq.controller.devices.device_zi import DeviceZI
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
        self._wait_for_AWGs = True
        self._emit_trigger = False

    def _process_dev_opts(self):
        if self.dev_type == "SHFSG8":
            self._channels = 8
            self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        elif self.dev_type == "SHFSG4":
            self._channels = 4
            self._output_to_synth_map = [0, 1, 2, 3]
        elif self._get_option("is_qc") and self.dev_type == "SHFQC":
            self._channels = 6
            self._output_to_synth_map = [0, 0, 1, 1, 2, 2]
        else:
            self._logger.warning(
                "%s: Unknown device type '%s', assuming 4 channels device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4

    def _get_sequencer_type(self) -> str:
        return "sg"

    def _get_num_AWGs(self):
        return self._channels

    def _validate_range(self, io: IO.Data):
        if io.range is None:
            return
        range_list = numpy.array(
            [-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=numpy.float64
        )
        label = "Output"
        if not any(numpy.isclose([io.range] * len(range_list), range_list)):
            self._logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. Nearest allowed range will be used.",
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

    def collect_conditions_to_close_loop(self, acquisition_units):
        return [
            DaqNodeWaitAction(
                self._daq, f"/{self.serial}/sgchannels/{awg_index}/awg/enable", 0
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_execution_nodes(self):
        self._logger.debug("Starting execution...")
        execution_nodes = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sgchannels/{awg_index}/awg/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]
        if self._emit_trigger:
            execution_nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/enable",
                    1,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )

        return execution_nodes

    def wait_for_execution_ready(self):
        # TODO(2K): hotfix, change to subscription and parallel waiting for all awgs of all followers
        if self._wait_for_AWGs:
            for awg_index in self._allocated_awgs:
                # Wait for awg being enabled. Shall we rather wait for it to disable
                # again/go to zero? It seems that sometimes the value of one was not
                # picked up reliably
                self._wait_for_node(
                    f"/{self.serial}/sgchannels/{awg_index}/awg/enable", 1, timeout=5
                )

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
                    raise Exception(
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
            synth_idx = self._output_to_synth_map[io.channel]
            prev_io = center_frequencies.get(synth_idx)
            if prev_io is None:
                center_frequencies[synth_idx] = io
            elif prev_io.lo_frequency != io.lo_frequency:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator frequency mismatch between outputs {prev_io.channel} and {io.channel} sharing synthesizer {synth_idx}: {prev_io.lo_frequency} != {io.lo_frequency}"
                )
            return synth_idx

        ios = initialization.outputs or []
        for io in ios:
            if io.lo_frequency is None:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator for channel {io.channel} is required, but is not provided."
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

    def collect_trigger_configuration_nodes(self, initialization: Initialization.Data):
        self._logger.debug("Configuring triggers...")
        self._wait_for_AWGs = True
        self._emit_trigger = False

        nodes_to_configure_triggers = []

        dio_mode = initialization.config.dio_mode

        if dio_mode == DIOConfigType.ZSYNC_DIO:
            pass
        elif dio_mode == DIOConfigType.HDAWG_LEADER or dio_mode == DIOConfigType.HDAWG:
            # standalone SHFSG or SHFQC
            self._wait_for_AWGs = False
            ntc = []
            if not self._get_option("qc_with_qa"):
                # otherwise, the QA will initialize the nodes
                self._emit_trigger = True
                clock_source = initialization.config.reference_clock_source
                ntc.append(
                    (
                        "system/clocks/referenceclock/in/source",
                        REFERENCE_CLOCK_SOURCE_INTERNAL
                        if clock_source
                        and clock_source.value == ReferenceClockSource.INTERNAL.value
                        else REFERENCE_CLOCK_SOURCE_EXTERNAL,
                    )
                )
                ntc += [
                    ("system/internaltrigger/enable", 0),
                    ("system/internaltrigger/repetitions", 1),
                ]
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                ntc += [
                    (f"sgchannels/{awg_index}/awg/auxtriggers/0/slope", 1),  # Rise
                    (
                        f"sgchannels/{awg_index}/awg/auxtriggers/0/channel",
                        8,
                    ),  # Internal trigger
                ]

            nodes_to_configure_triggers = [
                DaqNodeSetAction(self._daq, f"/{self.serial}/{node}", v)
                for node, v in ntc
            ]
        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type SHFSG."
            )

        return nodes_to_configure_triggers

    def configure_as_leader(self, initialization):
        raise LabOneQControllerException("SHFSG cannot be configured as leader")

    def collect_follower_configuration_nodes(self, initialization: Initialization.Data):
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

    def collect_reset_nodes(self):
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
