# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
import os
import os.path
import re
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

import numpy as np
import zhinst.core
import zhinst.utils
from numpy import typing as npt
from zhinst.core.errors import CoreError as LabOneCoreError  # pylint: disable=E0401

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeGetAction,
    DaqNodeSetAction,
    DaqWrapper,
)
from laboneq.controller.devices.zi_node_monitor import NodeControlBase
from laboneq.controller.recipe_1_4_0 import (
    Initialization,
    IntegratorAllocation,
    OscillatorParam,
)
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
)
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment


_logger = logging.getLogger(__name__)

seqc_osc_match = re.compile(
    r'(\s*string\s+osc_node_)(\w+)(\s*=\s*"oscs/)[0-9]+(/freq"\s*;\s*)', re.ASCII
)


class AwgCompilerStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    WARNING = 2


@dataclass
class AllocatedOscillator:
    group: int
    channels: set[int]
    index: int
    id: str
    frequency: float
    param: str


@dataclass
class DeviceOptions:
    serial: str
    interface: str
    dev_type: str | None = None
    is_qc: bool = False
    qc_with_qa: bool = False
    gen2: bool = False


@dataclass
class DeviceQualifier:
    dry_run: bool = True
    driver: str = None
    server: DaqWrapper = None
    options: DeviceOptions = None


def delay_to_rounded_samples(
    channel: int,
    dev_repr: str,
    delay: float,
    sample_frequency_hz,
    granularity_samples,
    max_node_delay_samples,
) -> int:
    if delay < 0:
        raise LabOneQControllerException(
            f"Negative node delay for device {dev_repr} and channel {channel} specified."
        )

    delay_samples = delay * sample_frequency_hz
    # Quantize to granularity and round ties towards zero
    delay_rounded = (
        math.ceil(delay_samples / granularity_samples + 0.5) - 1
    ) * granularity_samples

    if delay_rounded > max_node_delay_samples:
        raise LabOneQControllerException(
            f"Maximum delay via {dev_repr}'s node is "
            + f"{max_node_delay_samples / sample_frequency_hz * 1e9:.2f} ns - for larger "
            + "values, use the delay_signal property."
        )
    if abs(delay_samples - delay_rounded) > 1:
        _logger.debug(
            "Node delay %.2f ns of %s, channel %d will be rounded to "
            "%.2f ns, a multiple of %.0f samples.",
            delay_samples / sample_frequency_hz * 1e9,
            dev_repr,
            channel,
            delay_rounded / sample_frequency_hz * 1e9,
            granularity_samples,
        )

    return delay_rounded


class DeviceZI(ABC):
    def __init__(self, device_qualifier: DeviceQualifier):
        self._device_qualifier: DeviceQualifier = device_qualifier
        self._downlinks: dict[str, tuple[str, ReferenceType[DeviceZI]]] = {}
        self._uplinks: dict[str, ReferenceType[DeviceZI]] = {}
        self._rf_offsets: dict[int, float] = []

        self._daq: DaqWrapper = device_qualifier.server
        self.dev_type: str = None
        self.dev_opts: list[str] = []
        self._connected = False
        self._allocated_oscs: list[AllocatedOscillator] = []
        self._allocated_awgs: set[int] = set()
        self._nodes_to_monitor = None
        self._sampling_rate = None

        if self._daq is None:
            raise LabOneQControllerException("ZI devices need daq")

        if self.serial is None:
            raise LabOneQControllerException(
                "ZI device must be provided with serial number via options"
            )

        if self.interface is None or self.interface == "":
            raise LabOneQControllerException(
                "ZI device must be provided with interface via options"
            )

    @property
    def device_qualifier(self):
        return self._device_qualifier

    @property
    def dry_run(self):
        return self._device_qualifier.dry_run

    @property
    def dev_repr(self) -> str:
        return f"{self._device_qualifier.driver.upper()}:{self.serial}"

    @property
    def has_awg(self) -> bool:
        return self._get_num_awgs() > 0

    @property
    def options(self) -> DeviceOptions:
        return self._device_qualifier.options

    @property
    def serial(self):
        return self.options.serial.lower()

    @property
    def interface(self):
        return self.options.interface.lower()

    @property
    def daq(self):
        return self._daq

    def add_command_table_header(self, body: dict) -> dict:
        # Stub, implement in sub-class
        _logger.debug("Command table unavailable on device %s", self.dev_repr)
        return {}

    def command_table_path(self, awg_index: int) -> str:
        # Stub, implement in sub-class
        _logger.debug("No command table available for device %s", self.dev_repr)
        return ""

    def _warn_for_unsupported_param(self, param_assert, param_name, channel):
        if not param_assert:
            channel_clause = (
                "" if channel is None else f" specified for the channel {channel}"
            )
            _logger.warning(
                "%s: parameter '%s'%s is not supported on this device type.",
                self.dev_repr,
                param_name,
                channel_clause,
            )

    def _process_dev_opts(self):
        pass

    def _get_sequencer_type(self) -> str:
        return "auto-detect"

    def _get_sequencer_path_patterns(self) -> dict[str, str]:
        return {
            "elf": "/{serial}/awgs/{index}/elf/data",
            "progress": "/{serial}/awgs/{index}/elf/progress",
            "enable": "/{serial}/awgs/{index}/enable",
            "ready": "/{serial}/awgs/{index}/ready",
        }

    def get_sequencer_paths(self, index: int) -> dict[str, str]:
        props = {
            "serial": self.serial,
            "index": index,
        }
        patterns = self._get_sequencer_path_patterns()
        return {k: v.format(**props) for k, v in patterns.items()}

    def add_downlink(self, port: str, linked_device_uid: str, linked_device: DeviceZI):
        self._downlinks[port] = (linked_device_uid, ref(linked_device))

    def add_uplink(self, port: str, linked_device: DeviceZI):
        self._uplinks[port] = ref(linked_device)

    def remove_all_links(self):
        self._downlinks.clear()
        self._uplinks.clear()

    def is_leader(self):
        # Check also downlinks, to exclude standalone devices
        return len(self._uplinks) == 0 and len(self._downlinks) > 0

    def is_follower(self):
        # Treat standalone devices as followers
        return len(self._uplinks) > 0 or self.is_standalone()

    def is_standalone(self):
        return len(self._uplinks) == 0 and len(self._downlinks) == 0

    def collect_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> list[DaqNodeAction]:
        return []

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        return []

    def configure_as_leader(self, initialization: Initialization.Data):
        pass

    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> list[DaqNodeAction]:
        return []

    def _connect_to_data_server(self):
        if self._connected:
            return

        _logger.debug("%s: Connecting to %s interface.", self.dev_repr, self.interface)
        try:
            self._daq.connectDevice(self.serial, self.interface)
        except RuntimeError as exc:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Connecting failed"
            ) from exc

        _logger.debug("%s: Connected to %s interface.", self.dev_repr, self.interface)

        dev_type_path = f"/{self.serial}/features/devtype"
        dev_opts_path = f"/{self.serial}/features/options"
        dev_traits = self._daq.batch_get(
            [
                DaqNodeGetAction(self._daq, dev_type_path),
                DaqNodeGetAction(self._daq, dev_opts_path),
            ]
        )
        dev_type = dev_traits.get(dev_type_path)
        dev_opts = dev_traits.get(dev_opts_path)
        if isinstance(dev_type, str):
            self.dev_type = dev_type
        if isinstance(dev_opts, str):
            self.dev_opts = dev_opts.split("\n")
        self._process_dev_opts()

        self._connected = True

    def connect(self):
        self._connect_to_data_server()
        self._daq.node_monitor.add_nodes(self.nodes_to_monitor())

    def disconnect(self):
        if not self._connected:
            return

        self._daq.disconnectDevice(self.serial)
        self._connected = False

    def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        """Returns actions to disable the specified outputs for the device.

        outputs: set(int)
            - When 'invert' is False: set of outputs to disable.
            - When 'invert' is True: set of used outputs to be skipped, remaining
            outputs will be disabled.

        invert: bool
            Controls how 'outputs' argument is interpreted, see above. Special case: set
            to True along with empty 'outputs' to disable all outputs.
        """
        return []

    def shut_down(self):
        _logger.debug(
            "%s: Turning off signal output (stub, not implemented).", self.dev_repr
        )

    def free_allocations(self):
        self._allocated_oscs.clear()
        self._allocated_awgs.clear()

    def _nodes_to_monitor_impl(self):
        nodes = []
        nodes.extend([node.path for node in self.clock_source_control_nodes()])
        nodes.extend([node.path for node in self.system_freq_control_nodes()])
        nodes.extend([node.path for node in self.rf_offset_control_nodes()])
        return nodes

    def update_clock_source(self, force_internal: bool | None):
        pass

    def update_rf_offsets(self, rf_offsets: dict[int, float]):
        self._rf_offsets = rf_offsets

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        return []

    def system_freq_control_nodes(self) -> list[NodeControlBase]:
        return []

    def rf_offset_control_nodes(self) -> list[NodeControlBase]:
        return []

    def nodes_to_monitor(self) -> list[str]:
        if self._nodes_to_monitor is None:
            self._nodes_to_monitor = self._nodes_to_monitor_impl()
        return self._nodes_to_monitor

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> int | None:
        return None

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/oscs/{index}/freq"

    def allocate_osc(self, osc_param: OscillatorParam.Data):
        osc_group = self._osc_group_by_channel(osc_param.channel)
        osc_group_oscs = [o for o in self._allocated_oscs if o.group == osc_group]
        same_id_osc = next((o for o in osc_group_oscs if o.id == osc_param.id), None)
        if same_id_osc is None:
            # pylint: disable=E1128
            new_index = self._get_next_osc_index(osc_group, len(osc_group_oscs))
            if new_index is None:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: exceeded the number of available oscillators for "
                    f"channel {osc_param.channel}"
                )
            self._allocated_oscs.append(
                AllocatedOscillator(
                    group=osc_group,
                    channels={osc_param.channel},
                    index=new_index,
                    id=osc_param.id,
                    frequency=osc_param.frequency,
                    param=osc_param.param,
                )
            )
        else:
            if same_id_osc.frequency != osc_param.frequency:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: ambiguous frequency in recipe for oscillator "
                    f"'{osc_param.id}': {same_id_osc.frequency} != {osc_param.frequency}"
                )
            same_id_osc.channels.add(osc_param.channel)

    def allocate_params(self, initialization: Initialization.Data):
        pass

    def configure_feedback(self, recipe_data: RecipeData) -> list[DaqNodeAction]:
        return []

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> list[DaqNodeAction]:
        return []

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ):
        return None  # default -> no results available from the device

    def get_input_monitor_data(self, channel: int, num_results: int):
        return None  # default -> no results available from the device

    def wait_for_conditions_to_start(self):
        pass

    def conditions_for_execution_ready(self) -> dict[str, Any]:
        return {}

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType
    ) -> dict[str, Any]:
        return {}

    def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        pass

    def _wait_for_node(
        self, path: str, expected: Any, timeout: float, guard_time: float = 0
    ):
        retries = 0
        start_time = time.time()
        guard_start = None
        last_report = start_time
        last_val = None

        while True:
            if retries > 0:
                now = time.time()
                elapsed = floor(now - start_time)
                if now - start_time > timeout:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Node '{path}' didn't switch to '{expected}' "
                        f"within {timeout}s. Last value: {last_val}"
                    )
                if now - last_report > 5:
                    _logger.debug(
                        "Waiting for node '%s' switching to '%s', %f s remaining "
                        "until %f s timeout...",
                        path,
                        expected,
                        timeout - elapsed,
                        timeout,
                    )
                    last_report = now
                time.sleep(0.1)
            retries += 1

            daq_reply = self._daq.batch_get(
                [
                    DaqNodeGetAction(
                        self._daq, path, caching_strategy=CachingStrategy.NO_CACHE
                    )
                ]
            )
            last_val = daq_reply[path.lower()]

            if self.dry_run:
                break

            if last_val != expected:
                guard_start = None  # Start over the guard time waiting
                continue

            if guard_start is None:
                guard_start = time.time()
            if time.time() - guard_start >= guard_time:
                break

    def _switch_reference_clock(self, source, expected_freqs):
        if expected_freqs is not None and not isinstance(expected_freqs, list):
            expected_freqs = [expected_freqs]

        source_path = f"/{self.serial}/system/clocks/referenceclock/in/source"
        status_path = f"/{self.serial}/system/clocks/referenceclock/in/status"
        sourceactual_path = (
            f"/{self.serial}/system/clocks/referenceclock/in/sourceactual"
        )
        freq_path = f"/{self.serial}/system/clocks/referenceclock/in/freq"

        self._daq.batch_set(
            [
                DaqNodeSetAction(
                    self._daq,
                    source_path,
                    source,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        )

        retries = 0
        timeout = 60  # s
        start_time = time.time()
        last_report = start_time
        sourceactual = None
        status = None
        freq = None

        while True:
            if retries > 0:
                now = time.time()
                elapsed = floor(now - start_time)
                if now - start_time > timeout:
                    raise LabOneQControllerException(
                        f"Unable to switch reference clock within {timeout}s. "
                        f"Requested source: {source}, actual: {sourceactual}, status: {status}, "
                        f"expected frequencies: {expected_freqs}, actual: {freq}"
                    )
                if now - last_report > 5:
                    _logger.debug(
                        "Waiting for reference clock switching, %f s remaining "
                        "until %f s timeout...",
                        timeout - elapsed,
                        timeout,
                    )
                    last_report = now
                time.sleep(0.1)
            retries += 1

            daq_reply = self._daq.batch_get(
                [
                    DaqNodeGetAction(
                        self._daq,
                        sourceactual_path,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                ]
            )
            sourceactual = daq_reply[sourceactual_path]
            if sourceactual != source and not self.dry_run:
                continue

            daq_reply = self._daq.batch_get(
                [
                    DaqNodeGetAction(
                        self._daq,
                        status_path,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                ]
            )
            status = daq_reply[status_path]
            if not self.dry_run:
                if status == 2:  # still locking
                    continue
                if status == 1:  # error while locking
                    raise LabOneQControllerException(
                        f"Unable to switch reference clock, device returned error "
                        f"after {elapsed}s. Requested source: {source}, actual: {sourceactual}, "
                        f"status: {status}, expected frequency: {expected_freqs}, actual: {freq}"
                    )

            if expected_freqs is None:
                break
            daq_reply = self._daq.batch_get(
                [
                    DaqNodeGetAction(
                        self._daq, freq_path, caching_strategy=CachingStrategy.NO_CACHE
                    )
                ]
            )
            freq = daq_reply[freq_path]
            if freq in expected_freqs or self.dry_run:
                break
            else:
                raise LabOneQControllerException(
                    f"Unexpected frequency after switching the reference clock. "
                    f"Requested source: {source}, actual: {sourceactual}, status: {status}, "
                    f"expected frequency: {expected_freqs}, actual: {freq}"
                )

    def _adjust_frequency(self, freq):
        return freq

    def collect_prepare_sweep_step_nodes_for_param(
        self, sweep_params_tracker: SweepParamsTracker
    ) -> list[DaqNodeAction]:
        nodes_to_set: list[DaqNodeAction] = []
        for osc in self._allocated_oscs:
            if osc.param in sweep_params_tracker.updated_params():
                freq_value = self._adjust_frequency(
                    sweep_params_tracker.get_param(osc.param)
                )
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        self._make_osc_path(next(iter(osc.channels)), osc.index),
                        freq_value,
                    )
                )
        return nodes_to_set

    @staticmethod
    def _contains_only_zero_or_one(a):
        if a is None:
            return True
        return not np.any(a * (1 - a))

    def _prepare_wave_iq(self, waves, sig: str) -> tuple[str, npt.ArrayLike]:
        wave_i = next((w for w in waves if w["filename"] == f"{sig}_i.wave"), None)
        if not wave_i:
            raise LabOneQControllerException(
                f"I wave not found, IQ wave signature '{sig}'"
            )

        wave_q = next((w for w in waves if w["filename"] == f"{sig}_q.wave"), None)
        if not wave_q:
            raise LabOneQControllerException(
                f"Q wave not found, IQ wave signature '{sig}'"
            )

        marker1_samples = None
        marker2_samples = None
        try:
            marker1_samples = next(
                (w for w in waves if w["filename"] == f"{sig}_marker1.wave")
            )["samples"]
        except StopIteration:
            pass

        try:
            marker2_samples = next(
                (w for w in waves if w["filename"] == f"{sig}_marker2.wave")
            )["samples"]
        except StopIteration:
            pass

        marker_samples = None
        if marker1_samples is not None:
            if not self._contains_only_zero_or_one(marker1_samples):
                raise LabOneQControllerException(
                    "Marker samples must only contain ones and zeros"
                )
            marker_samples = np.array(marker1_samples)
        if marker2_samples is not None:
            if marker_samples is None:
                marker_samples = np.zeros(len(marker2_samples), dtype=np.int32)
            elif len(marker1_samples) != len(marker2_samples):
                raise LabOneQControllerException(
                    "Samples for marker1 and marker2 must have the same length"
                )
            if not self._contains_only_zero_or_one(marker2_samples):
                raise LabOneQControllerException(
                    "Marker samples must only contain ones and zeros"
                )
            marker2_samples = np.array(marker2_samples)
            # we want marker 2 to be played on output 2, marker 1
            # bits 0/1 = marker 1/2 of output 1, bit 2/4 = marker 1/2 output 2
            # bit 2 is factor 4
            factor = 4
            marker_samples += factor * marker2_samples

        return (
            sig,
            zhinst.utils.convert_awg_waveform(
                np.clip(wave_i["samples"], -1, 1),
                np.clip(wave_q["samples"], -1, 1),
                markers=marker_samples,
            ),
        )

    def _prepare_wave_single(self, waves, sig: str) -> tuple[str, npt.ArrayLike]:
        wave = next((w for w in waves if w["filename"] == f"{sig}.wave"), None)
        marker_samples = None
        try:
            marker_samples = next(
                (w for w in waves if w["filename"] == f"{sig}_marker1.wave")
            )["samples"]
        except StopIteration:
            pass

        if not self._contains_only_zero_or_one(marker_samples):
            raise LabOneQControllerException(
                "Marker samples must only contain ones and zeros"
            )

        if not wave:
            raise LabOneQControllerException(f"Wave not found, signature '{sig}'")

        return sig, zhinst.utils.convert_awg_waveform(
            np.clip(wave["samples"], -1, 1), markers=marker_samples
        )

    def _prepare_wave_complex(self, waves, sig: str) -> tuple[str, npt.ArrayLike]:
        filename_to_find = f"{sig}.wave"
        wave = next((w for w in waves if w["filename"] == filename_to_find), None)

        if not wave:
            raise LabOneQControllerException(
                f"Wave not found, signature '{sig}' filename '{filename_to_find}'"
            )

        return sig, np.array(wave["samples"], dtype=np.complex128)

    def _prepare_waves(
        self, compiled: CompiledExperiment, seqc_filename: str
    ) -> list[tuple[str, npt.ArrayLike]]:
        wave_indices_filename = os.path.splitext(seqc_filename)[0] + "_waveindices.csv"
        wave_indices: dict[str, list[int | str]] = next(
            (
                i
                for i in compiled.wave_indices
                if i["filename"] == wave_indices_filename
            ),
            {"value": {}},
        )["value"]

        waves_by_index = {}
        waves = compiled.waves or []
        for sig, [idx, sig_type] in wave_indices.items():
            if sig_type in ("iq", "double", "multi"):
                waves_by_index[idx] = self._prepare_wave_iq(waves, sig)
            elif sig_type == "single":
                waves_by_index[idx] = self._prepare_wave_single(waves, sig)
            elif sig_type == "complex":
                waves_by_index[idx] = self._prepare_wave_complex(waves, sig)
            else:
                raise LabOneQControllerException(
                    f"Unexpected signal type for binary wave for '{sig}' in '{seqc_filename}' - "
                    f"'{sig_type}', should be one of [iq, double, multi, single, complex]"
                )

        bin_waves: list[tuple[str, npt.ArrayLike]] = []
        idx = 0
        while idx in waves_by_index:
            bin_waves.append(waves_by_index[idx])
            idx += 1
        return bin_waves

    def _prepare_command_table(
        self, compiled: CompiledExperiment, seqc_filename: str
    ) -> dict | None:
        command_table_body = next(
            (ct["ct"] for ct in compiled.command_tables if ct["seqc"] == seqc_filename),
            None,
        )

        if command_table_body is None:
            return None

        oscillator_map = {osc.id: osc.index for osc in self._allocated_oscs}
        command_table_body = deepcopy(command_table_body)
        for entry in command_table_body:
            if "oscillatorSelect" not in entry:
                continue
            oscillator_uid = entry["oscillatorSelect"]["value"]["$ref"]
            entry["oscillatorSelect"]["value"] = oscillator_map[oscillator_uid]

        return self.add_command_table_header(command_table_body)

    def prepare_seqc(
        self, seqc_filename: str, compiled: CompiledExperiment
    ) -> tuple[str, list[tuple[str, npt.ArrayLike]], dict[Any]]:
        """
        `compiled` expected to have the following members:
         - `src`   -> list[dict[str, str]]
                        `filename` -> `<seqc_filename>`
                        `text`     -> `<seqc_content>`
         - `waves` -> list[dict[str, str]]
                        `filename` -> `<wave_filename_csv>`
                        `text`     -> `<wave_content_csv>`

        Returns a tuple of
         1. str: seqc text to pass to the awg compiler
         2. list[(str, array)]: waves(id, samples) to upload to the instrument (ordered by index)
         3. dict: command table
        """
        seqc = next((s for s in compiled.src if s["filename"] == seqc_filename), None)
        if seqc is None:
            raise LabOneQControllerException(
                f"SeqC program '{seqc_filename}' not found"
            )

        # Substitute oscillator nodes by actual assignment
        seqc_lines = seqc["text"].split("\n")
        for i, seqc_line in enumerate(seqc_lines):
            m = seqc_osc_match.match(seqc_line)
            if m is not None:
                param = m.group(2)
                for osc in self._allocated_oscs:
                    if osc.param == param:
                        seqc_lines[
                            i
                        ] = f"{m.group(1)}{m.group(2)}{m.group(3)}{osc.index}{m.group(4)}"
        seqc_text = "\n".join(seqc_lines)

        bin_waves = self._prepare_waves(compiled, seqc_filename)
        command_table = self._prepare_command_table(compiled, seqc_filename)

        return seqc_text, bin_waves, command_table

    def prepare_upload_elf(self, elf: bytes, awg_index: int, filename: str):
        sequencer_paths = self.get_sequencer_paths(awg_index)
        return DaqNodeSetAction(
            self._daq,
            sequencer_paths["elf"],
            elf,
            filename=filename,
            caching_strategy=CachingStrategy.NO_CACHE,
        )

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
            f"/{self.serial}/awgs/{awg_index}/waveform/waves/{wave_index}",
            waveform,
            filename=filename,
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    def prepare_upload_all_binary_waves(
        self,
        awg_index,
        waves: list[tuple[str, npt.ArrayLike]],
        acquisition_type: AcquisitionType,
    ):
        # Default implementation for "old" devices, override for newer devices
        return [
            self.prepare_upload_binary_wave(
                filename=filename,
                waveform=waveform,
                awg_index=awg_index,
                wave_index=wave_index,
                acquisition_type=acquisition_type,
            )
            for wave_index, [filename, waveform] in enumerate(waves)
        ]

    def _upload_all_binary_waves(
        self,
        awg_index,
        waves: list[tuple[str, npt.ArrayLike]],
        acquisition_type: AcquisitionType,
    ):
        waves_upload = self.prepare_upload_all_binary_waves(
            awg_index, waves, acquisition_type
        )
        self._daq.batch_set(waves_upload)

    def prepare_upload_command_table(self, awg_index, command_table: dict):
        command_table_path = self.command_table_path(awg_index)
        return DaqNodeSetAction(
            self._daq,
            command_table_path + "data",
            json.dumps(command_table, sort_keys=True),
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    def upload_command_table(self, awg_index, command_table: dict):
        command_table_path = self.command_table_path(awg_index)
        self._daq.batch_set(
            [self.prepare_upload_command_table(awg_index, command_table)]
        )

        status_path = command_table_path + "status"

        status = int(
            self._daq.batch_get(
                [
                    DaqNodeGetAction(
                        self._daq,
                        status_path,
                    )
                ]
            )[status_path]
        )

        if status & 0b1000 != 0:
            raise LabOneQControllerException("Failed to parse command table JSON")
        if not self.dry_run:
            if status & 0b0001 == 0:
                raise LabOneQControllerException("Failed to upload command table")

    def compile_seqc(self, code: str, awg_index: int, filename_hint: str = None):
        _logger.debug(
            "%s: Compiling sequence for AWG #%d...",
            self.dev_repr,
            awg_index,
        )
        sequencer = self._get_sequencer_type()
        sequencer = "auto" if sequencer == "auto-detect" else sequencer

        try:
            elf, extra = zhinst.core.compile_seqc(
                code,
                self.dev_type,
                options=self.dev_opts,
                index=awg_index,
                sequencer=sequencer,
                filename=filename_hint,
                samplerate=self._sampling_rate,
            )
        except LabOneCoreError as exc:
            raise LabOneQControllerException(  # pylint: disable=W0707
                f"{self.dev_repr}: AWG compilation failed.\n{str(exc)}"
            )

        compiler_warnings = extra["messages"]
        if compiler_warnings:
            raise LabOneQControllerException(
                f"{self.dev_repr}: AWG compilation succeeded, but there are warnings:\n"
                f"{compiler_warnings}"
            )

        _logger.debug(
            "%s: Compilation successful on AWG #%d with no warnings.",
            self.dev_repr,
            awg_index,
        )

        return elf

    def _get_num_awgs(self):
        return 0

    def collect_osc_initialization_nodes(self) -> list[DaqNodeAction]:
        nodes_to_initialize_oscs = []
        osc_inits = {
            self._make_osc_path(ch, osc.index): osc.frequency
            for osc in self._allocated_oscs
            for ch in osc.channels
        }
        for path, freq in osc_inits.items():
            nodes_to_initialize_oscs.append(
                DaqNodeSetAction(
                    self._daq, path, 0 if freq is None else self._adjust_frequency(freq)
                )
            )
        return nodes_to_initialize_oscs

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        return []

    def collect_awg_after_upload_nodes(self, initialization: Initialization.Data):
        return []

    def collect_execution_nodes(self):
        nodes_to_execute = []
        _logger.debug("%s: Executing AWGS...", self.dev_repr)

        if self._daq is not None:
            for awg_index in self._allocated_awgs:
                _logger.debug(
                    "%s: Starting AWG #%d sequencer", self.dev_repr, awg_index
                )
                path = f"/{self.serial}/awgs/{awg_index}/enable"
                nodes_to_execute.append(
                    DaqNodeSetAction(
                        self._daq, path, 1, caching_strategy=CachingStrategy.NO_CACHE
                    )
                )

        return nodes_to_execute

    def collect_start_execution_nodes(self):
        return []

    def check_errors(self):
        error_node = f"/{self.serial}/raw/error/json/errors"
        all_errors = self._daq.get_raw(error_node)
        if not self.dry_run:
            # for proper testing of the logic we have to mock the data server better.
            # Currently is returns 0 for all nodes...
            check_errors(all_errors[error_node], self.dev_repr)

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        return [DaqNodeSetAction(self._daq, f"/{self.serial}/raw/error/clear", 1)]


class ErrorLevels(Enum):
    info = 0
    warning = 1
    error = 2


def check_errors(errors, serial):
    collected_messages = []
    for error in errors:
        # the key in error["vector"] looks like a dict, but it's a string. so we have to use
        # json.loads to convert it into a dict.
        error_vector = json.loads(error["vector"])
        for message in error_vector["messages"]:
            if message["code"] == "AWGRUNTIMEERROR" and message["params"][0] == 1:
                awg_core = int(message["attribs"][0])
                program_counter = int(message["params"][1])
                collected_messages.append(
                    f"Gap detected on AWG core {awg_core}, program counter {program_counter}"
                )
            if message["severity"] >= ErrorLevels.error.value:
                collected_messages.append(message["message"])
    if len(collected_messages) > 0:
        all_messages = "\n".join(collected_messages)
        raise LabOneQControllerException(
            f"An error happened on device {serial} during the execution of the experiment. "
            f"Error messages:\n{all_messages}"
        )
        # should we return the warnings in the log?
