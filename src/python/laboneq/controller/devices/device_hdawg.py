# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any, Iterator

from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.hdawg_core import HDAwgCore
from laboneq.controller.utilities.for_each import for_each
import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.device_utils import FloatWithTolerance, NodeCollector
from laboneq.controller.devices.device_zi import (
    DeviceBase,
    delay_to_rounded_samples,
)
from laboneq.controller.devices.node_control import (
    Command,
    Condition,
    NodeControlBase,
    Prepare,
    Response,
    Setting,
    WaitCondition,
)
from laboneq.controller.recipe_processor import AwgSignalType, RecipeData
from laboneq.data.recipe import (
    Initialization,
    NtStepKey,
)

_logger = logging.getLogger(__name__)

TRIGGER_INPUT_LEVEL_FOR_DESKTOP_SETUP = 0.0
"""For setups without PQSC, we suggest customers to connect UHFQA ref. clock
output to a trigger input on the HDAWG. The ref. clock output of UHFQA is
AC-coupled, so triggering at 0 V is safe, especially when the amplitude is
low (e.g. due to usage of passive splitters for clock distribution)."""

DELAY_NODE_GRANULARITY_SAMPLES = 1
DELAY_NODE_MAX_SAMPLES = 62
DEFAULT_SAMPLE_FREQUENCY_HZ = 2.4e9
GEN2_SAMPLE_FREQUENCY_HZ = 2.0e9


class DIOMode(IntEnum):
    MANUAL = 0
    AWG_SEQUENCER = 1
    DIO_CODEWORD = 2
    QCCS = 3


class ReferenceClockSourceHDAWG(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1
    ZSYNC = 2


class DeviceHDAWG(DeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "HDAWG8"
        self.dev_opts = ["MF", "ME", "SKW"]
        self._hd_awg_cores: list[HDAwgCore] = []
        self._channels = 8
        self._cores = 4
        self._multi_freq = True
        self._reference_clock_source = ReferenceClockSourceHDAWG.INTERNAL
        self._sampling_rate = (
            GEN2_SAMPLE_FREQUENCY_HZ
            if self.options.gen2
            else DEFAULT_SAMPLE_FREQUENCY_HZ
        )

    @property
    def sampling_rate(self) -> float:
        assert self._sampling_rate is not None
        return self._sampling_rate

    @property
    def has_pipeliner(self) -> bool:
        return True

    def all_channels(self) -> Iterator[ChannelBase]:
        """Iterable over all awg cores of the device."""
        return iter(self._hd_awg_cores)

    def allocated_channels(self, recipe_data: RecipeData) -> Iterator[ChannelBase]:
        for ch in recipe_data.allocated_awgs(self.uid):
            yield self._hd_awg_cores[ch]

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        if self.dev_type == "HDAWG8":
            self._channels = 8
            self._cores = 4
        elif self.dev_type == "HDAWG4":
            self._channels = 4
            self._cores = 2
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming 4 channels device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
            self._cores = 2

        self._multi_freq = "MF" in self.dev_opts
        self._hd_awg_cores = [
            HDAwgCore(
                api=self._api,
                subscriber=self._subscriber,
                device_uid=self.uid,
                serial=self.serial,
                channel=ch,
                repr_base=self.dev_repr,
                is_follower=self.is_follower(),
                is_leader=self.is_leader(),
                has_precompensation="PC" in self.dev_opts,
            )
            for ch in range(self._cores)
        ]

    def _busy_nodes(self, recipe_data: RecipeData) -> list[str]:
        busy_nodes = []
        for awg in recipe_data.allocated_awgs(self.uid):
            # We check busy always for both channels even if only
            # one channel is used - overhead is minimal.
            busy_nodes.append(f"/{self.serial}/sigouts/{awg * 2}/busy")
            busy_nodes.append(f"/{self.serial}/sigouts/{awg * 2 + 1}/busy")
        return busy_nodes

    async def disable_outputs(self, outputs: set[int], invert: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"sigouts/{ch}/on", 0, cache=False)
        await self.set_async(nc)

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

    def _sigout_from_port(self, ports: list[str]) -> int | None:
        if len(ports) != 1:
            return None
        port_parts = ports[0].upper().split("/")
        if len(port_parts) != 2 or port_parts[0] != "SIGOUTS":
            return None
        return int(port_parts[1])

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
            # TODO(2K): Remove once https://zhinst.atlassian.net/browse/HULK-1800 is resolved
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/source", 0),
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/status", 0),
        ]

    def runtime_check_control_nodes(self) -> list[NodeControlBase]:
        # Enable AWG runtime checks which includes the gap detector.
        return [
            Setting(
                f"/{self.serial}/raw/system/awg/runtimechecks/enable",
                int(self._enable_runtime_checks),
            )
        ]

    def system_freq_control_nodes(self) -> list[NodeControlBase]:
        # If we do not turn all channels off, we get the following error message from
        # the server/device: 'An error happened on device dev8330 during the execution
        # of the experiment. Error message: Reinitialized signal output delay on
        # channel 0 (numbered from 0)'
        # See also https://zhinst.atlassian.net/browse/HBAR-1374?focusedCommentId=41373
        nodes: list[NodeControlBase] = [
            Prepare(f"/{self.serial}/sigouts/{channel}/on", 0)
            for channel in range(self._channels)
        ]
        nodes.extend(
            [
                Setting(
                    f"/{self.serial}/system/clocks/sampleclock/freq",
                    self.sampling_rate,
                ),
                Response(f"/{self.serial}/system/clocks/sampleclock/status", 0),
            ]
        )
        return nodes

    def rf_offset_control_nodes(self) -> list[NodeControlBase]:
        nodes: list[NodeControlBase] = []
        for channel, offset in self._voltage_offsets.items():
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

    async def set_after_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")

        device_recipe_data = recipe_data.device_settings[self.uid]
        for awg_index, awg_config in device_recipe_data.awg_configs.items():
            _logger.debug(
                "%s: Configure modulation phase depending on IQ enabling on awg %d.",
                self.dev_repr,
                awg_index,
            )
            nc.add(
                f"sines/{awg_index * 2}/phaseshift",
                90 if (awg_config.signal_type == AwgSignalType.IQ) else 0,
            )
            nc.add(f"sines/{awg_index * 2 + 1}/phaseshift", 0)

        await self.set_async(nc)

    async def _do_start_execution(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        for awg_index in recipe_data.allocated_awgs(self.uid):
            if recipe_data.rt_execution_info.with_pipeliner:
                nc.extend(
                    self._hd_awg_cores[awg_index]._pipeliner.collect_execution_nodes()
                )
            else:
                nc.add(f"awgs/{awg_index}/enable", 1, cache=False)
        await self.set_async(nc)

    async def start_execution(self, recipe_data: RecipeData):
        # For standalone HDAWG start the execution at the emit_start_trigger stage
        if not self.is_standalone():
            await self._do_start_execution(recipe_data=recipe_data)

    async def emit_start_trigger(self, recipe_data: RecipeData):
        if self.is_leader() or self.is_standalone():
            await self._do_start_execution(recipe_data=recipe_data)

    def conditions_for_execution_ready(
        self, recipe_data: RecipeData
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in recipe_data.allocated_awgs(self.uid):
            if (
                recipe_data.rt_execution_info.with_pipeliner
                and not self.is_standalone()
            ):
                conditions.update(
                    self._hd_awg_cores[
                        awg_index
                    ]._pipeliner.conditions_for_execution_ready()
                )
            elif not self.is_standalone():
                conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                    1,
                    f"AWG {awg_index} didn't start.",
                )
        return conditions

    def conditions_for_execution_done(
        self, recipe_data: RecipeData
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in recipe_data.allocated_awgs(self.uid):
            if recipe_data.rt_execution_info.with_pipeliner:
                conditions.update(
                    self._hd_awg_cores[
                        awg_index
                    ]._pipeliner.conditions_for_execution_done(
                        with_execution_start=self.is_standalone()
                    )
                )
            elif self.is_standalone():
                conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                    [1, 0],
                    f"AWG {awg_index} failed to transition to exec and back to stop.",
                )
            else:
                conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                    0,
                    f"AWG {awg_index} didn't stop. Missing start trigger? Check ZSync.",
                )
        return conditions

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        nc = NodeCollector(base=f"/{self.serial}/")
        if (
            with_pipeliner
            and self._has_awg_in_use(recipe_data)
            and not self.is_standalone()
        ):
            nc.add("system/synchronization/source", 1)  # external
        await self.set_async(nc)

    async def teardown_one_step_execution(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")

        if recipe_data.rt_execution_info.with_pipeliner and not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            nc.add("system/synchronization/source", 0)

        if recipe_data.rt_execution_info.with_pipeliner:
            for awg_index in recipe_data.allocated_awgs(self.uid):
                nc.extend(self._hd_awg_cores[awg_index]._pipeliner.reset_nodes())

        # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
        # wrongly detected gaps when enabled during experiments with feedback.
        # Here we make sure that if they were enabled at `session.connect` we
        # re-enable them in case the previous experiment had feedback.
        nc.add("raw/system/awg/runtimechecks/enable", int(self._enable_runtime_checks))

        await self.set_async(nc)

    async def apply_initialization(
        self,
        recipe_data: RecipeData,
    ):
        device_recipe_data = recipe_data.device_settings[self.uid]
        if device_recipe_data is None:
            return

        await for_each(
            self.all_channels(),
            HDAwgCore.apply_initialization,
            device_recipe_data=device_recipe_data,
        )

        nc = NodeCollector(base=f"/{self.serial}/")

        # Configure DIO/ZSync at init (previously was after AWG loading).
        # This is a prerequisite for passing AWG checks in FW on the pipeliner commit.
        # Without the pipeliner, these checks are only performed when the AWG is enabled,
        # therefore DIO could be configured after the AWG loading.
        if self.is_follower():
            nc.add("dios/0/mode", DIOMode.QCCS)
            nc.add("dios/0/drive", 0b1100)

        elif self.is_leader():
            # For desktop setups (setup with HDAWG and UHFQA only) we recommend
            # users to connect UHFQA reference clock output to the trigger
            # input of the first channel on the HDAWG.
            nc.add("triggers/in/0/level", TRIGGER_INPUT_LEVEL_FOR_DESKTOP_SETUP)
            # The reference clock output of UHFQA is 50 Ohm. If we don't match
            # it here, this 'may' cause issue for the user if they are providing the
            # clock by splitting it and the cable lengths involved are large.
            nc.add("triggers/in/0/imp50", 1)

            # DIO_CODEWORD is mandatory for HDAWG+UHFQA systems to prevent jitter.
            nc.add("dios/0/mode", DIOMode.DIO_CODEWORD)
            nc.add("dios/0/drive", 0b1111)

        osc_selects = {
            ch: osc.index
            for osc in device_recipe_data.allocated_oscs
            for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nc.add(f"sines/{ch}/oscselect", osc_idx)

        await self._api.set_parallel(nc)

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

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"/{self.serial}/")

        device_recipe_data = recipe_data.device_settings[self.uid]
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
                        ch_repr=f"{self.dev_repr}:ch{ch}",
                        delay=output_delay,
                        sample_frequency_hz=self.sampling_rate,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
                    )
                    / self.sampling_rate
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
            if awg_idx not in device_recipe_data.iq_settings:
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
                ch_i, ch_q = device_recipe_data.iq_settings[awg_idx]
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
        await self.set_async(nc)

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("system/awg/oscillatorcontrol", 1)
        await self.set_async(nc)

    def command_table_path(self, awg_index: int) -> str:
        return self._hd_awg_cores[awg_index].nodes.awg_command_table + "/"

    async def configure_trigger(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")

        for awg_key, awg_config in recipe_data.awg_configs.items():
            has_no_feedback = awg_config.source_feedback_register is None
            if (awg_key.device_uid != self.uid) or has_no_feedback:
                continue

            # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
            # wrongly detected gaps when enabled during experiments with feedback.
            # Here we ensure that the gap detector is disabled if we are
            # configuring feedback.
            nc.add("raw/system/awg/runtimechecks/enable", 0)
            break

        await self.set_async(nc)

    async def reset_to_idle(self):
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
        nc.add("awgs/*/pipeliner/reset", 1, cache=False)
        nc.add("awgs/*/pipeliner/mode", 0, cache=False)  # off
        nc.add("awgs/*/synchronization/enable", 0, cache=False)
        nc.barrier()
        nc.add("awgs/*/enable", 0, cache=False)
        nc.add("system/synchronization/source", 0, cache=False)  # internal
        await self.set_async(nc)
        # Reset errors must be the last operation, as above sets may cause errors.
        # See https://zhinst.atlassian.net/browse/HULK-1606
        await super().reset_to_idle()
