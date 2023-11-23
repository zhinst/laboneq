# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_zi import DeviceZI, delay_to_rounded_samples
from laboneq.controller.devices.zi_node_monitor import (
    Command,
    Condition,
    FloatWithTolerance,
    NodeControlBase,
    Prepare,
    Response,
)
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import Initialization, SignalType, TriggeringMode

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
        self.pipeliner_set_node_base(f"/{self.serial}/awgs")

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
        self, osc_group: int, previously_allocated: int
    ) -> int | None:
        # With MF option 4 oscillators per channel pair are available,
        # and only 1 oscillator per channel pair without MF option.
        max_per_group = 4 if self._multi_freq else 1
        if previously_allocated >= max_per_group:
            return None
        osc_index_base = osc_group * max_per_group
        return osc_index_base + previously_allocated

    def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        channels_to_disable: list[DaqNodeSetAction] = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/sigouts/{ch}/on",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for ch in range(self._channels)
            if (ch in outputs) != invert
        ]
        return channels_to_disable

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
            Command(f"/{self.serial}/system/clocks/referenceclock/source", source),
            Response(f"/{self.serial}/system/clocks/referenceclock/status", 0),
        ]

    def collect_load_factory_preset_nodes(self):
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/system/preset/load",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
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
                Command(
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
                    Command(f"/{self.serial}/sigouts/{channel}/on", 1),
                    Command(
                        f"/{self.serial}/sigouts/{channel}/offset",
                        FloatWithTolerance(offset, 0.0001),
                    ),
                ]
            )
        return nodes

    def collect_awg_after_upload_nodes(self, initialization: Initialization):
        nodes_to_configure_phase = []

        for awg in initialization.awgs or []:
            _logger.debug(
                "%s: Configure modulation phase depending on IQ enabling on awg %d.",
                self.dev_repr,
                awg.awg,
            )
            nodes_to_configure_phase.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sines/{awg.awg * 2}/phaseshift",
                    90 if (awg.signal_type == SignalType.IQ) else 0,
                )
            )

            nodes_to_configure_phase.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/sines/{awg.awg * 2 + 1}/phaseshift", 0
                )
            )

        return nodes_to_configure_phase

    def collect_execution_nodes(self, with_pipeliner: bool) -> list[DaqNodeAction]:
        if with_pipeliner:
            return self.pipeliner_collect_execution_nodes()

        return super().collect_execution_nodes(with_pipeliner=with_pipeliner)

    def conditions_for_execution_ready(self, with_pipeliner: bool) -> dict[str, Any]:
        if with_pipeliner:
            return self.pipeliner_conditions_for_execution_ready()

        return {
            f"/{self.serial}/awgs/{awg_index}/enable": 1
            for awg_index in self._allocated_awgs
        }

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, Any]:
        if with_pipeliner:
            return self.pipeliner_conditions_for_execution_done()

        return {
            f"/{self.serial}/awgs/{awg_index}/enable": 0
            for awg_index in self._allocated_awgs
        }

    def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeAction]:
        nodes = []
        if with_pipeliner and has_awg_in_use:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    1,  # external
                )
            )
        return nodes

    def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeAction]:
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nodes: list[tuple[str, Any]] = []

        outputs = initialization.outputs or []
        for output in outputs:
            awg_idx = output.channel // 2
            self._allocated_awgs.add(awg_idx)

            nodes.append((f"sigouts/{output.channel}/on", 1 if output.enable else 0))

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
                nodes.append(
                    (
                        f"sigouts/{output.channel}/range",
                        output.range,
                    )
                )
            nodes.append((f"sigouts/{output.channel}/offset", output.offset))
            nodes.append((f"awgs/{awg_idx}/single", 1))

            awg_ch = output.channel % 2
            iq_idx = output.channel // 2
            iq_gains_mx = device_recipe_data.iq_settings.get(iq_idx, None)

            if iq_gains_mx is None:
                # Fall back to old behavior (suitable for single channel output)
                diagonal_channel_index = awg_ch
                off_diagonal_channel_index = int(not diagonal_channel_index)

                modulation_mode = (
                    (ModulationMode.SINE_00 if awg_ch == 0 else ModulationMode.SINE_11)
                    if output.modulation
                    else ModulationMode.OFF
                )
                nodes += [
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/modulation/mode",
                        modulation_mode,
                    ),
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/gains/{diagonal_channel_index}",
                        output.gains.diagonal,
                    ),
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/gains/{off_diagonal_channel_index}",
                        output.gains.off_diagonal,
                    ),
                ]
            else:
                # I/Q output
                modulation_mode = (
                    ModulationMode.MIXER_CAL
                    if output.modulation
                    else ModulationMode.OFF
                )
                nodes += [
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/modulation/mode",
                        modulation_mode,
                    ),
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/gains/0",
                        iq_gains_mx[0][awg_ch],
                    ),
                    (
                        f"awgs/{awg_idx}/outputs/{awg_ch}/gains/1",
                        iq_gains_mx[1][awg_ch],
                    ),
                ]

            precomp_p = f"sigouts/{output.channel}/precompensation/"
            try:
                precomp = output.precompensation
                if not precomp:
                    raise AttributeError
                nodes.append((precomp_p + "enable", 1))
                # Exponentials
                for e in range(8):
                    exp_p = precomp_p + f"exponentials/{e}/"
                    try:
                        exp = precomp["exponential"][e]
                        nodes += [
                            (exp_p + "enable", 1),
                            (exp_p + "timeconstant", exp["timeconstant"]),
                            (exp_p + "amplitude", exp["amplitude"]),
                        ]
                    except (KeyError, IndexError, TypeError):
                        nodes.append((exp_p + "enable", 0))
                # Bounce
                bounce_p = precomp_p + "bounces/0/"
                try:
                    bounce = precomp["bounce"]
                    delay = bounce["delay"]
                    amp = bounce["amplitude"]
                    nodes += [
                        (bounce_p + "enable", 1),
                        (bounce_p + "delay", delay),
                        (bounce_p + "amplitude", amp),
                    ]
                except (KeyError, TypeError):
                    nodes.append((bounce_p + "enable", 0))
                # Highpass
                hp_p = precomp_p + "highpass/0/"
                try:
                    hp = precomp["high_pass"]
                    timeconstant = hp["timeconstant"]
                    nodes += [
                        (hp_p + "enable", 1),
                        (hp_p + "timeconstant", timeconstant),
                        (hp_p + "clearing/slope", 1),
                    ]
                except (KeyError, TypeError):
                    nodes.append((hp_p + "enable", 0))
                # FIR
                fir_p = precomp_p + "fir/"
                try:
                    fir = np.array(precomp["FIR"]["coefficients"])
                    if len(fir) > 40:
                        raise LabOneQControllerException(
                            "FIR coefficients must be a list of at most 40 doubles"
                        )
                    fir = np.concatenate((fir, np.zeros((40 - len(fir)))))
                    nodes += [(fir_p + "enable", 1), (fir_p + "coefficients", fir)]
                except (KeyError, IndexError, TypeError):
                    nodes.append((fir_p + "enable", 0))
            except (KeyError, TypeError, AttributeError):
                nodes.append((precomp_p + "enable", 0))
            if output.marker_mode is not None:
                if output.marker_mode == "TRIGGER":
                    nodes.append(
                        (f"triggers/out/{output.channel}/source", output.channel % 2)
                    )
                elif output.marker_mode == "MARKER":
                    nodes.append(
                        (
                            f"triggers/out/{output.channel}/source",
                            4 + 2 * (output.channel % 2),
                        )
                    )
                else:
                    raise ValueError(
                        f"Maker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel} on HDAWG {self.serial}"
                    )
                # set trigger delay to 0
                nodes.append((f"triggers/out/{output.channel}/delay", 0.0))

        osc_selects = {
            ch: osc.index for osc in self._allocated_oscs for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nodes.append((f"sines/{ch}/oscselect", osc_idx))

        set_actions = [
            DaqNodeSetAction(self._daq, f"/{self.serial}/{k}", v) for k, v in nodes
        ]
        # Configure DIO/ZSync at init (previously was after AWG loading).
        # This is a prerequisite for passing AWG checks in FW on the pipeliner commit.
        # Without the pipeliner, these checks are only performed when the AWG is enabled,
        # therefore DIO could be configured after the AWG loading.
        set_actions.extend(
            self.collect_dio_configuration_nodes(initialization, recipe_data)
        )
        return set_actions

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        nodes_to_set = super().collect_prepare_nt_step_nodes(attributes, recipe_data)

        for ch in range(self._channels):
            [scheduler_port_delay, port_delay], updated = attributes.resolve(
                keys=[
                    (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.OUTPUT_PORT_DELAY, ch),
                ]
            )
            if not updated or scheduler_port_delay is None:
                continue

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

            nodes_to_set.append(
                DaqNodeSetAction(
                    daq=self.daq,
                    path=f"/{self.serial}/sigouts/{ch}/delay",
                    value=output_delay_rounded,
                )
            )

        return nodes_to_set

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ):
        device_specific_initialization_nodes = [
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/system/awg/oscillatorcontrol", 1
            ),
        ]

        return device_specific_initialization_nodes

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
    ) -> list[DaqNodeAction]:
        return []

    def collect_dio_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        _logger.debug("%s: Configuring trigger configuration nodes.", self.dev_repr)
        nodes_to_configure_triggers = []

        triggering_mode = initialization.config.triggering_mode
        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            _logger.debug(
                "%s: Configuring DIO mode: ZSync pass-through.", self.dev_repr
            )
            _logger.debug("%s: Configuring external clock to ZSync.", self.dev_repr)
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/mode", 3)
            )

            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/drive", 0xC)
            )

            # Loop over at least AWG instance to cover the case that the instrument is only used
            # as a communication proxy. Some of the nodes on the AWG branch are needed to get
            # proper communication between HDAWG and UHFQA.
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                awg_path = f"/{self.serial}/awgs/{awg_index}"
                nodes_to_configure_triggers.extend(
                    [
                        DaqNodeSetAction(self._daq, f"{awg_path}/dio/strobe/slope", 0),
                        DaqNodeSetAction(
                            self._daq, f"{awg_path}/dio/valid/polarity", 0
                        ),
                    ]
                )
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
                    nodes_to_configure_triggers.extend(
                        [
                            DaqNodeSetAction(
                                self._daq,
                                f"{awg_path}/zsync/register/shift",
                                awg_config.register_selector_shift,
                            ),
                            DaqNodeSetAction(
                                self._daq,
                                f"{awg_path}/zsync/register/mask",
                                awg_config.register_selector_bitmask,
                            ),
                            DaqNodeSetAction(
                                self._daq,
                                f"{awg_path}/zsync/register/offset",
                                awg_config.command_table_match_offset,
                            ),
                        ]
                    )
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/triggers/in/0/level",
                    DIG_TRIGGER_1_LEVEL,
                )
            )

            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/awgs/{awg_index}/auxtriggers/0/slope",
                        1,
                    )
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/awgs/{awg_index}/auxtriggers/0/channel",
                        0,
                    )
                )

            nodes_to_configure_triggers.extend(
                [
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/triggers/out/0/source", 0
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/triggers/out/1/source", 1
                    ),
                    DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/mode", 1),
                    DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/drive", 15),
                ]
            )

            # Loop over at least AWG instance to cover the case that the instrument is only used
            # as a communication proxy. Some of the nodes on the AWG branch are needed to get
            # proper communication between HDAWG and UHFQA.
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/awgs/{awg_index}/dio/strobe/slope",
                        0,
                    )
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/awgs/{awg_index}/dio/valid/polarity",
                        2,
                    )
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/awgs/{awg_index}/dio/valid/index", 0
                    )
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/awgs/{awg_index}/dio/mask/value",
                        0x3FF,
                    )
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/awgs/{awg_index}/dio/mask/shift", 1
                    )
                )

        return nodes_to_configure_triggers

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        reset_nodes = []
        reset_nodes.extend(
            [
                # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
                *self.pipeliner_reset_nodes(),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/awgs/*/enable",
                    0,
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    0,  # internal
                    caching_strategy=CachingStrategy.NO_CACHE,
                ),
            ]
        )
        # Reset errors must be the last operation, as above sets may cause errors.
        # See https://zhinst.atlassian.net/browse/HULK-1606
        reset_nodes.extend(super().collect_reset_nodes())
        return reset_nodes
