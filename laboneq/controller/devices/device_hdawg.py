# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, Iterator

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    DaqNodeSetAction,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    AllocatedOscillator,
    DeviceZI,
    delay_to_rounded_samples,
)
from laboneq.controller.attribute_value_tracker import DeviceAttribute
from laboneq.controller.devices.zi_node_monitor import (
    Command,
    Condition,
    FloatWithTolerance,
    NodeControlBase,
    Prepare,
    Response,
    Setting,
)
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import (
    Initialization,
    OscillatorParam,
    SignalType,
    TriggeringMode,
)

_logger = logging.getLogger(__name__)

DIG_TRIGGER_1_LEVEL = 0.225

DELAY_NODE_GRANULARITY_SAMPLES = 1
DELAY_NODE_MAX_SAMPLES = 62
DEFAULT_SAMPLE_FREQUENCY_HZ = 2.4e9
GEN2_SAMPLE_FREQUENCY_HZ = 2.0e9


class ReferenceClockSourceHDAWG(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1
    ZSYNC = 2


class ModulationMode(IntEnum):
    OFF = 0
    SINE_00 = 1
    SINE_11 = 2
    SINE_01 = 3
    SINE_10 = 4
    ADVANCED = 5
    MIXER_CAL = 6


class DeviceHDAWG(AwgPipeliner, DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "HDAWG8"
        self.dev_opts = ["MF", "ME", "SKW"]
        self._channels = 8
        self._multi_freq = True
        self._reference_clock_source = ReferenceClockSourceHDAWG.INTERNAL
        self._sampling_rate = (
            GEN2_SAMPLE_FREQUENCY_HZ
            if self.options.gen2
            else DEFAULT_SAMPLE_FREQUENCY_HZ
        )
        self.pipeliner_set_node_base(f"/{self.serial}/awgs", "AWG")

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        if self.dev_type == "HDAWG8":
            self._channels = 8
        elif self.dev_type == "HDAWG4":
            self._channels = 4
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming 4 channels device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4

        self._multi_freq = "MF" in self.dev_opts

    def _get_num_awgs(self) -> int:
        return self._channels // 2

    def _osc_group_by_channel(self, channel: int) -> int:
        # For LabOne Q SW, the AWG oscillator control is always on, in which
        # case every pair of output channels share the same set of oscillators
        return channel // 2

    def _get_next_osc_index(
        self,
        osc_group_oscs: list[AllocatedOscillator],
        osc_param: OscillatorParam,
        recipe_data: RecipeData,
    ) -> int | None:
        osc_group = self._osc_group_by_channel(osc_param.channel)
        previously_allocated = len(osc_group_oscs)
        # With MF option 4 oscillators per channel pair are available,
        # and only 1 oscillator per channel pair without MF option.
        max_per_group = 4 if self._multi_freq else 1
        if previously_allocated >= max_per_group:
            return None
        osc_index_base = osc_group * max_per_group
        return osc_index_base + previously_allocated

    async def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"sigouts/{ch}/on", 0, cache=False)
        return await self.maybe_async(nc)

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.append(f"/{self.serial}/awgs/{awg}/enable")
            nodes.append(f"/{self.serial}/awgs/{awg}/ready")
            nodes.extend(self.pipeliner_control_nodes(awg))
        return nodes

    def update_clock_source(self, force_internal: bool | None):
        if force_internal or force_internal is None and self.is_standalone():
            # Internal specified explicitly or
            # the source is not specified, but HDAWG is a standalone device
            self._reference_clock_source = ReferenceClockSourceHDAWG.INTERNAL
        elif self.is_leader() or self.is_standalone():
            # If HDAWG is a leader or standalone (and not explicitly forced to internal),
            # external is the default (or explicit)
            self._reference_clock_source = ReferenceClockSourceHDAWG.EXTERNAL
        else:
            # If HDAWG is a follower (and not explicitly forced to internal),
            # ZSync is the default (explicit external is also treated as ZSync in this case)
            self._reference_clock_source = ReferenceClockSourceHDAWG.ZSYNC

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        expected_freq = {
            ReferenceClockSourceHDAWG.INTERNAL: None,
            ReferenceClockSourceHDAWG.EXTERNAL: 10e6,
            ReferenceClockSourceHDAWG.ZSYNC: 100e6,
        }[self._reference_clock_source]
        source = self._reference_clock_source.value

        return [
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/freq", expected_freq
            ),
            Setting(f"/{self.serial}/system/clocks/referenceclock/source", source),
            Response(f"/{self.serial}/system/clocks/referenceclock/status", 0),
        ]

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return [
            Command(f"/{self.serial}/system/preset/load", 1),
            Response(f"/{self.serial}/system/preset/busy", 0),
        ]

    def system_freq_control_nodes(self) -> list[NodeControlBase]:
        # If we do not turn all channels off, we get the following error message from
        # the server/device: 'An error happened on device dev8330 during the execution
        # of the experiment. Error message: Reinitialized signal output delay on
        # channel 0 (numbered from 0)'
        # See also https://zhinst.atlassian.net/browse/HBAR-1374?focusedCommentId=41373
        nodes = [
            Prepare(f"/{self.serial}/sigouts/{channel}/on", 0)
            for channel in range(self._channels)
        ]
        nodes.extend(
            [
                Setting(
                    f"/{self.serial}/system/clocks/sampleclock/freq",
                    self._sampling_rate,
                ),
                Response(f"/{self.serial}/system/clocks/sampleclock/status", 0),
            ]
        )
        return nodes

    def rf_offset_control_nodes(self) -> list[NodeControlBase]:
        nodes = []
        for channel, offset in self._rf_offsets.items():
            nodes.extend(
                [
                    Setting(f"/{self.serial}/sigouts/{channel}/on", 1),
                    Setting(
                        f"/{self.serial}/sigouts/{channel}/offset",
                        FloatWithTolerance(offset, 0.0001),
                    ),
                ]
            )
        return nodes

    async def collect_awg_after_upload_nodes(
        self, initialization: Initialization
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")

        for awg in initialization.awgs or []:
            _logger.debug(
                "%s: Configure modulation phase depending on IQ enabling on awg %d.",
                self.dev_repr,
                awg.awg,
            )
            nc.add(
                f"sines/{awg.awg * 2}/phaseshift",
                90 if (awg.signal_type == SignalType.IQ) else 0,
            )
            nc.add(f"sines/{awg.awg * 2 + 1}/phaseshift", 0)

        return await self.maybe_async(nc)

    async def collect_execution_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        if with_pipeliner:
            return await self.maybe_async(self.pipeliner_collect_execution_nodes())

        return await super().collect_execution_nodes(with_pipeliner=with_pipeliner)

    async def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if with_pipeliner:
            conditions = self.pipeliner_conditions_for_execution_ready()
        else:
            conditions = {
                f"/{self.serial}/awgs/{awg_index}/enable": (
                    1,
                    f"{self.dev_repr}: AWG {awg_index + 1} didn't start.",
                )
                for awg_index in self._allocated_awgs
            }
        return conditions  # await self.maybe_async_wait(conditions)

    async def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if with_pipeliner:
            conditions = self.pipeliner_conditions_for_execution_done()
        else:
            conditions = {
                f"/{self.serial}/awgs/{awg_index}/enable": (
                    0,
                    f"{self.dev_repr}: AWG {awg_index + 1} didn't stop. Missing start trigger? Check ZSync.",
                )
                for awg_index in self._allocated_awgs
            }
        return conditions  # await self.maybe_async_wait(conditions)

    async def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        if with_pipeliner and has_awg_in_use:
            nc.add("system/synchronization/source", 1)  # external
        return await self.maybe_async(nc)

    async def collect_execution_teardown_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")

        if not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            nc.add("system/synchronization/source", 0)

        return await self.maybe_async(nc)

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        _logger.debug("%s: Initializing device...", self.dev_repr)
        nc = NodeCollector(base=f"/{self.serial}/")

        outputs = initialization.outputs or []
        for output in outputs:
            awg_idx = output.channel // 2
            self._allocated_awgs.add(awg_idx)

            nc.add(f"sigouts/{output.channel}/on", 1 if output.enable else 0)

            if output.range is not None:
                if output.range_unit not in (None, "volt"):
                    raise LabOneQControllerException(
                        f"The output range of device {self.dev_repr} is specified in "
                        f"units of {output.range_unit}. Units must be 'volt'."
                    )
                if output.range not in (0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0):
                    _logger.warning(
                        "The specified output range %s for device %s is not in the list of "
                        "supported values. It will be rounded to the next higher allowed value.",
                        output.range,
                        self.dev_repr,
                    )
                nc.add(f"sigouts/{output.channel}/range", output.range)
            nc.add(f"awgs/{awg_idx}/single", 1)

            awg_ch = output.channel % 2
            if awg_idx in device_recipe_data.iq_settings:
                modulation_mode = (
                    ModulationMode.MIXER_CAL
                    if output.modulation
                    else ModulationMode.OFF
                )
            else:
                modulation_mode = (
                    (ModulationMode.SINE_00 if awg_ch == 0 else ModulationMode.SINE_11)
                    if output.modulation
                    else ModulationMode.OFF
                )

            nc.add(f"awgs/{awg_idx}/outputs/{awg_ch}/modulation/mode", modulation_mode)
            precomp_p = f"sigouts/{output.channel}/precompensation/"
            has_pc = "PC" in self.dev_opts
            try:
                precomp = output.precompensation
                if not precomp:
                    raise AttributeError
                if not has_pc:
                    raise LabOneQControllerException(
                        f"Precompensation is not supported on device {self.dev_repr}."
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
                if has_pc:
                    nc.add(precomp_p + "enable", 0)
            if output.marker_mode is not None:
                if output.marker_mode == "TRIGGER":
                    nc.add(f"triggers/out/{output.channel}/source", output.channel % 2)
                elif output.marker_mode == "MARKER":
                    nc.add(
                        f"triggers/out/{output.channel}/source",
                        4 + 2 * (output.channel % 2),
                    )
                else:
                    raise ValueError(
                        f"Maker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel} on HDAWG {self.serial}"
                    )
                # set trigger delay to 0
                nc.add(f"triggers/out/{output.channel}/delay", 0.0)

        osc_selects = {
            ch: osc.index for osc in self._allocated_oscs for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nc.add(f"sines/{ch}/oscselect", osc_idx)

        # Configure DIO/ZSync at init (previously was after AWG loading).
        # This is a prerequisite for passing AWG checks in FW on the pipeliner commit.
        # Without the pipeliner, these checks are only performed when the AWG is enabled,
        # therefore DIO could be configured after the AWG loading.
        nc.extend(self._collect_dio_configuration_nodes(initialization, recipe_data))

        return await self.maybe_async(nc)

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)
        for io in initialization.outputs or []:
            yield DeviceAttribute(
                name=AttributeName.OUTPUT_VOLTAGE_OFFSET,
                index=io.channel,
                value_or_param=io.offset,
            )
            yield DeviceAttribute(
                name=AttributeName.OUTPUT_GAIN_DIAGONAL,
                index=io.channel,
                value_or_param=io.gains.diagonal,
            )
            yield DeviceAttribute(
                name=AttributeName.OUTPUT_GAIN_OFF_DIAGONAL,
                index=io.channel,
                value_or_param=io.gains.off_diagonal,
            )

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super().collect_prepare_nt_step_nodes(attributes, recipe_data))

        awg_iq_pair_set: set[int] = set()
        for ch in range(self._channels):
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
                        sample_frequency_hz=self._sampling_rate,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
                    )
                    / self._sampling_rate
                )
                nc.add(f"sigouts/{ch}/delay", output_delay_rounded)
            [output_voltage_offset], updated = attributes.resolve(
                keys=[
                    (AttributeName.OUTPUT_VOLTAGE_OFFSET, ch),
                ]
            )
            if updated and output_voltage_offset is not None:
                nc.add(f"sigouts/{ch}/offset", output_voltage_offset)

            awg_idx = ch // 2
            output_i = output_iq = ch % 2
            output_q = output_i ^ 1
            if (
                awg_idx
                not in recipe_data.device_settings[
                    self.device_qualifier.uid
                ].iq_settings
            ):
                # Fall back to old behavior (suitable for single channel output)
                (
                    [output_gain_i_diagonal, output_gain_i_off_diagonal],
                    updated,
                ) = attributes.resolve(
                    keys=[
                        (AttributeName.OUTPUT_GAIN_DIAGONAL, ch),
                        (AttributeName.OUTPUT_GAIN_OFF_DIAGONAL, ch),
                    ]
                )
                if (
                    updated
                    and output_gain_i_diagonal is not None
                    and output_gain_i_off_diagonal is not None
                ):
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_iq}/gains/{output_i}",
                        output_gain_i_diagonal,
                    )
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_iq}/gains/{output_q}",
                        output_gain_i_off_diagonal,
                    )
            else:
                if awg_idx in awg_iq_pair_set:
                    continue
                # Set both I and Q channels at one loop
                ch_i, ch_q = recipe_data.device_settings[
                    self.device_qualifier.uid
                ].iq_settings[awg_idx]
                (
                    [
                        output_gain_i_diagonal,
                        output_gain_i_off_diagonal,
                        output_gain_q_diagonal,
                        output_gain_q_off_diagnoal,
                    ],
                    updated,
                ) = attributes.resolve(
                    keys=[
                        (AttributeName.OUTPUT_GAIN_DIAGONAL, ch_i),
                        (AttributeName.OUTPUT_GAIN_OFF_DIAGONAL, ch_i),
                        (AttributeName.OUTPUT_GAIN_DIAGONAL, ch_q),
                        (AttributeName.OUTPUT_GAIN_OFF_DIAGONAL, ch_q),
                    ]
                )
                if (
                    updated
                    and output_gain_i_diagonal is not None
                    and output_gain_i_off_diagonal is not None
                    and output_gain_q_diagonal is not None
                    and output_gain_q_off_diagnoal is not None
                ):
                    iq_mixer_calib_mx = np.array(
                        [
                            [output_gain_i_diagonal, output_gain_q_off_diagnoal],
                            [output_gain_i_off_diagonal, output_gain_q_diagonal],
                        ]
                    )
                    # Normalize resulting matrix to its inf-norm, to avoid clamping
                    iq_mixer_calib_normalized = iq_mixer_calib_mx / np.linalg.norm(
                        iq_mixer_calib_mx, np.inf
                    )
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_i}/gains/0",
                        iq_mixer_calib_normalized[0][output_i],
                    )
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_i}/gains/1",
                        iq_mixer_calib_normalized[1][output_i],
                    )
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_q}/gains/0",
                        iq_mixer_calib_normalized[0][output_q],
                    )
                    nc.add(
                        f"awgs/{awg_idx}/outputs/{output_q}/gains/1",
                        iq_mixer_calib_normalized[1][output_q],
                    )
                awg_iq_pair_set.add(awg_idx)
        return nc

    async def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("system/awg/oscillatorcontrol", 1)
        return await self.maybe_async(nc)

    def add_command_table_header(self, body: dict) -> dict:
        return {
            "$schema": "https://docs.zhinst.com/hdawg/commandtable/v1_0/schema",
            "header": {"version": "1.0.0"},
            "table": body,
        }

    def command_table_path(self, awg_index: int) -> str:
        return f"/{self.serial}/awgs/{awg_index}/commandtable/"

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        return []

    def _collect_dio_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> NodeCollector:
        _logger.debug("%s: Configuring trigger configuration nodes.", self.dev_repr)
        nc = NodeCollector(base=f"/{self.serial}/")

        triggering_mode = initialization.config.triggering_mode
        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            _logger.debug(
                "%s: Configuring DIO mode: ZSync pass-through.", self.dev_repr
            )
            _logger.debug("%s: Configuring external clock to ZSync.", self.dev_repr)
            nc.add("dios/0/mode", 3)
            nc.add("dios/0/drive", 0xC)

            # Loop over at least one AWG instance to cover the case that the instrument is only used
            # as a communication proxy. Some of the nodes on the AWG branch are needed to get
            # proper communication between HDAWG and UHFQA.
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                awg_path = f"awgs/{awg_index}"
                nc.add(f"{awg_path}/dio/strobe/slope", 0)
                nc.add(f"{awg_path}/dio/valid/polarity", 0)
                awg_config = next(
                    (
                        awg_config
                        for awg_key, awg_config in recipe_data.awg_configs.items()
                        if (
                            awg_key.device_uid == initialization.device_uid
                            and awg_key.awg_index == awg_index
                            and awg_config.source_feedback_register is not None
                        )
                    ),
                    None,
                )
                if awg_config is not None:
                    nc.add(
                        f"{awg_path}/zsync/register/shift",
                        awg_config.register_selector_shift,
                    )
                    nc.add(
                        f"{awg_path}/zsync/register/mask",
                        awg_config.register_selector_bitmask,
                    )
                    nc.add(
                        f"{awg_path}/zsync/register/offset",
                        awg_config.command_table_match_offset,
                    )
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            nc.add("triggers/in/0/level", DIG_TRIGGER_1_LEVEL)

            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                nc.add(f"awgs/{awg_index}/auxtriggers/0/slope", 1)
                nc.add(f"awgs/{awg_index}/auxtriggers/0/channel", 0)

            nc.add("dios/0/mode", 1)
            nc.add("dios/0/drive", 15)

            # Loop over at least AWG instance to cover the case that the instrument is only used
            # as a communication proxy. Some of the nodes on the AWG branch are needed to get
            # proper communication between HDAWG and UHFQA.
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                nc.add(f"awgs/{awg_index}/dio/strobe/slope", 0)
                nc.add(f"awgs/{awg_index}/dio/valid/polarity", 2)
                nc.add(f"awgs/{awg_index}/dio/valid/index", 0)
                nc.add(f"awgs/{awg_index}/dio/mask/value", 0x3FF)
                nc.add(f"awgs/{awg_index}/dio/mask/shift", 1)

        return nc

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
        nc.extend(self.pipeliner_reset_nodes())
        nc.barrier()
        nc.add("awgs/*/enable", 0, cache=False)
        nc.add("system/synchronization/source", 0, cache=False)  # internal
        reset_nodes = await self.maybe_async(nc)
        # Reset errors must be the last operation, as above sets may cause errors.
        # See https://zhinst.atlassian.net/browse/HULK-1606
        reset_nodes.extend(await super().collect_reset_nodes())
        return reset_nodes
