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
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from weakref import ReferenceType, ref

import numpy as np
import zhinst.core
import zhinst.utils
from numpy import typing as npt
from zhinst.core.errors import CoreError as LabOneCoreError  # pylint: disable=E0401

from laboneq.controller.communication import (
    AwgModuleWrapper,
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
    RtExecutionInfo,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment

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
    channels: Set[int]
    index: int
    id: str
    frequency: float
    param: str


@dataclass
class DeviceQualifier:
    dry_run: bool = True
    driver: str = None
    server: DaqWrapper = None
    options: Dict[str, str] = None


class DeviceZI(ABC):
    def __init__(self, device_qualifier: DeviceQualifier):
        self._device_qualifier: DeviceQualifier = device_qualifier
        self._logger = logging.getLogger(__name__)
        self._downlinks: Dict[str, ReferenceType[DeviceZI]] = {}
        self._uplinks: Dict[str, ReferenceType[DeviceZI]] = {}

        self._daq: DaqWrapper = device_qualifier.server
        self.dev_type = None
        self.dev_opts = []
        self._connected = False
        self._allocated_oscs: List[AllocatedOscillator] = []
        self._allocated_awgs: Set[int] = set()
        self._nodes_to_monitor = None
        self._sampling_rate = None

        self._is_using_standalone_compiler = device_qualifier.options.get(
            "standalone_awg", False
        )

        if self._daq is None:
            raise LabOneQControllerException("ZI devices need daq")

        self._awg_modules: List[AwgModuleWrapper] = []

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
        return f"{self._device_qualifier.driver.upper()}:{self._get_option('serial')}"

    @property
    def has_awg(self) -> bool:
        return self._get_num_awgs() > 0

    @property
    def serial(self):
        return self._get_option("serial").lower()

    @property
    def interface(self):
        return self._get_option("interface").lower()

    @property
    def daq(self):
        return self._daq

    def add_command_table_header(self, body: Dict) -> Dict:
        # Stub, implement in sub-class
        self._logger.debug("Command table unavailable on device %s", self.dev_repr)
        return {}

    def command_table_path(self, awg_index: int) -> str:
        # Stub, implement in sub-class
        self._logger.debug("No command table available for device %s", self.dev_repr)
        return ""

    def _get_option(self, key):
        return self._device_qualifier.options.get(key)

    def _warn_for_unsupported_param(self, param_assert, param_name, channel):
        if not param_assert:
            channel_clause = (
                "" if channel is None else f" specified for the channel {channel}"
            )
            self._logger.warning(
                "%s: parameter '%s'%s is not supported on this device type.",
                self.dev_repr,
                param_name,
                channel_clause,
            )

    def _process_dev_opts(self):
        pass

    def _get_sequencer_type(self) -> str:
        return "auto-detect"

    def _get_sequencer_path_patterns(self) -> Dict[str, str]:
        return {
            "elf": "/{serial}/awgs/{index}/elf/data",
            "progress": "/{serial}/awgs/{index}/elf/progress",
            "enable": "/{serial}/awgs/{index}/enable",
            "ready": "/{serial}/awgs/{index}/ready",
        }

    def get_sequencer_paths(self, index: int) -> Dict[str, str]:
        props = {
            "serial": self.serial,
            "index": index,
        }
        patterns = self._get_sequencer_path_patterns()
        return {k: v.format(**props) for k, v in patterns.items()}

    def add_downlink(self, port: str, linked_device: DeviceZI):
        self._downlinks[port] = ref(linked_device)

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

    @abstractmethod
    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        ...

    @abstractmethod
    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> List[DaqNodeAction]:
        ...

    @abstractmethod
    def configure_as_leader(self, initialization: Initialization.Data):
        ...

    @abstractmethod
    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        ...

    def connect(self):
        if self._connected:
            return

        self._logger.debug(
            "%s: Connecting to %s interface.", self.dev_repr, self.interface
        )
        try:
            self._daq.connectDevice(self.serial, self.interface)
        except RuntimeError as exc:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Connecting failed"
            ) from exc

        self._logger.debug(
            "%s: Connected to %s interface.", self.dev_repr, self.interface
        )

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

        if not self._is_using_standalone_compiler:
            for i in range(self._get_num_awgs()):
                awg_module = self._daq.create_awg_module(
                    f"{self.serial}:awg_module{str(i)}"
                )
                self._awg_modules.append(awg_module)
                awg_config = [
                    DaqNodeSetAction(awg_module, "/index", i),
                    DaqNodeSetAction(awg_module, "/device", self.serial),
                ]
                if self._get_option("is_qc"):
                    awg_config.append(
                        DaqNodeSetAction(
                            awg_module, "/sequencertype", self._get_sequencer_type()
                        )
                    )
                awg_module.batch_set(awg_config)
                awg_module.execute()
                self._logger.debug("%s: Creating AWG Module #%d", self.dev_repr, i)

        self._daq.node_monitor.add_nodes(self.nodes_to_monitor())

        self._connected = True

    def free_allocations(self):
        self._allocated_oscs.clear()
        self._allocated_awgs.clear()

    def _nodes_to_monitor_impl(self):
        return []

    def update_clock_source(self, force_internal: Optional[bool]):
        pass

    def clock_source_control_nodes(self) -> List[NodeControlBase]:
        return []

    def nodes_to_monitor(self) -> List[str]:
        if self._nodes_to_monitor is None:
            self._nodes_to_monitor = self._nodes_to_monitor_impl()
        return self._nodes_to_monitor

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> Optional[int]:
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

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> List[DaqNodeAction]:
        return []

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: List[int],
        num_results: int,
        hw_averages: int,
    ):
        return None  # default -> no results available from the device

    def get_input_monitor_data(self, channel: int, num_results: int):
        return None  # default -> no results available from the device

    def wait_for_conditions_to_start(self):
        pass

    def conditions_for_execution_ready(self) -> Dict[str, Any]:
        return {}

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType
    ) -> Dict[str, Any]:
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
                    self._logger.debug(
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
                    self._logger.debug(
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

    def collect_prepare_sweep_step_nodes_for_param(self, param: str, value: float):
        nodes_to_set: List[DaqNodeAction] = []
        for osc in self._allocated_oscs:
            if osc.param == param:
                freq_value = self._adjust_frequency(value)
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        self._make_osc_path(next(iter(osc.channels)), osc.index),
                        freq_value,
                    )
                )
        return nodes_to_set

    def _wait_for_elf_upload(self, awg_index):
        awg_module = self._awg_modules[awg_index]

        # this cannot use batch_get because these two nodes are 'special'
        while awg_module.progress < 1.0 and (awg_module.elf_status != 1):
            pass

        if awg_module.elf_status != 0:
            status_string = awg_module.get(
                DaqNodeGetAction(
                    awg_module,
                    "/compiler/statusstring",
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )
            self._logger.error("ELF upload error:\n%s", status_string)
            raise LabOneQControllerException(
                "ELF file upload to the instrument failed."
            )

    def _check_awg_compiler_status(self, awg_index):
        self._logger.debug(
            "%s: Checking the status of compilation for AWG #%d...",
            self.dev_repr,
            awg_index,
        )
        while True:
            awg_module = self._awg_modules[awg_index]
            compiler_status = awg_module.get(
                DaqNodeGetAction(
                    awg_module,
                    "/compiler/status",
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )

            if compiler_status == AwgCompilerStatus.SUCCESS.value:
                self._logger.debug(
                    "%s: Compilation successful on AWG #%d with no warnings, will upload the "
                    "program to the instrument.",
                    self.dev_repr,
                    awg_index,
                )
                break

            status_string = awg_module.get(
                DaqNodeGetAction(
                    awg_module,
                    "/compiler/statusstring",
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )

            if compiler_status in (
                AwgCompilerStatus.ERROR.value,
                AwgCompilerStatus.WARNING.value,
            ):
                raise LabOneQControllerException(
                    f"{self.dev_repr}: AWG compilation failed, compiler output:\n{status_string}"
                )
            time.sleep(0.1)

    @staticmethod
    def _contains_only_zero_or_one(a):
        if a is None:
            return True
        return not np.any(a * (1 - a))

    def _prepare_wave_iq(self, waves, sig: str) -> Tuple[str, npt.ArrayLike]:
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

    def _prepare_wave_single(self, waves, sig: str) -> Tuple[str, npt.ArrayLike]:
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

    def _prepare_wave_complex(self, waves, sig: str) -> Tuple[str, npt.ArrayLike]:
        filename_to_find = f"{sig}.wave"
        wave = next((w for w in waves if w["filename"] == filename_to_find), None)

        if not wave:
            raise LabOneQControllerException(
                f"Wave not found, signature '{sig}' filename '{filename_to_find}'"
            )

        return sig, np.array(wave["samples"], dtype=np.complex128)

    def _prepare_waves(
        self, compiled: CompiledExperiment, seqc_filename: str
    ) -> List[Tuple[str, npt.ArrayLike]]:
        wave_indices_filename = os.path.splitext(seqc_filename)[0] + "_waveindices.csv"
        wave_indices: Dict[str, List[Union[int, str]]] = next(
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

        bin_waves: List[Tuple[str, npt.ArrayLike]] = []
        idx = 0
        while idx in waves_by_index:
            bin_waves.append(waves_by_index[idx])
            idx += 1
        return bin_waves

    def _prepare_command_table(
        self, compiled: CompiledExperiment, seqc_filename: str
    ) -> Optional[Dict]:
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
    ) -> Tuple[str, List[Tuple[str, npt.ArrayLike]], Dict[Any]]:
        """
        `compiled` expected to have the following members:
         - `src`   -> List[Dict[str, str]]
                        `filename` -> `<seqc_filename>`
                        `text`     -> `<seqc_content>`
         - `waves` -> List[Dict[str, str]]
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
        waves: List[Tuple[str, npt.ArrayLike]],
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
        waves: List[Tuple[str, npt.ArrayLike]],
        acquisition_type: AcquisitionType,
    ):
        waves_upload = self.prepare_upload_all_binary_waves(
            awg_index, waves, acquisition_type
        )
        self._daq.batch_set(waves_upload)

    def prepare_upload_command_table(self, awg_index, command_table: Dict):
        command_table_path = self.command_table_path(awg_index)
        return DaqNodeSetAction(
            self._daq,
            command_table_path + "data",
            json.dumps(command_table),
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    def upload_command_table(self, awg_index, command_table: Dict):
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

    def _compile_and_upload_seqc(
        self, code: str, awg_index: int, filename_hint: str = None
    ):
        try:
            self._logger.debug(
                "%s: Running AWG compiler on AWG #%d...",
                self.dev_repr,
                awg_index,
            )
            awg_module = self._awg_modules[awg_index]
            awg_module.batch_set(
                [
                    DaqNodeSetAction(
                        awg_module,
                        "/compiler/sourcestring",
                        code,
                        filename=filename_hint,
                        caching_strategy=CachingStrategy.NO_CACHE,  # if only external waves changed
                    )
                ]
            )
        except LabOneQControllerException as exc:
            raise LabOneQControllerException(
                f"Exception raised while uploading program from file {filename_hint} "
                f"to AWG #{awg_index}\nSeqC code:\n{code}"
            ) from exc

        # TODO(2K): handle timeout, emit:
        # f"{str(exp)}\nAWG compiler timed out while trying to compile:\n{data}\n"
        self._check_awg_compiler_status(awg_index)
        self._wait_for_elf_upload(awg_index)

    def compile_seqc(self, code: str, awg_index: int, filename_hint: str = None):
        self._logger.debug(
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

        self._logger.debug(
            "%s: Compilation successful on AWG #%d with no warnings.",
            self.dev_repr,
            awg_index,
        )

        return elf

    def upload_awg_program(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        assert not self._is_using_standalone_compiler

        if initialization.awgs is None:
            return

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for awg_obj in initialization.awgs:
            awg_index = awg_obj.awg

            self._logger.debug(
                "%s: Starting to compile and upload AWG program '%s' to AWG #%d",
                self.dev_repr,
                awg_obj.seqc,
                awg_index,
            )

            data, waves, command_table = self.prepare_seqc(
                awg_obj.seqc, recipe_data.compiled
            )

            self._compile_and_upload_seqc(data, awg_index, filename_hint=awg_obj.seqc)

            self._upload_all_binary_waves(awg_index, waves, acquisition_type)
            if command_table is not None:
                self.upload_command_table(awg_index, command_table)

    def _get_num_awgs(self):
        return 0

    def collect_osc_initialization_nodes(self) -> List[DaqNodeAction]:
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
        self._logger.debug("%s: Executing AWGS...", self.dev_repr)

        if self._daq is not None:
            for awg_index in self._allocated_awgs:
                self._logger.debug(
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

    def shut_down(self):
        for awg_module in self._awg_modules:
            if awg_module is not None:
                self._logger.debug(
                    "%s: Stopping AWG sequencer (stub, not implemented).", self.dev_repr
                )

        if self._daq is not None:
            self._logger.debug(
                "%s: Turning off signal output (stub, not implemented).", self.dev_repr
            )

    def disconnect(self):
        if self._daq is not None:
            self._daq.disconnectDevice(self.serial)

    def check_errors(self):
        error_node = f"/{self.serial}/raw/error/json/errors"
        all_errors = self._daq.get_raw(error_node)
        if not self.dry_run:
            # for proper testing of the logic we have to mock the data server better.
            # Currently is returns 0 for all nodes...
            check_errors(all_errors[error_node], self.dev_repr)

    def collect_reset_nodes(self) -> List[DaqNodeAction]:
        return [DaqNodeSetAction(self._daq, f"/{self.serial}/raw/error/clear", 1)]

    def _get_total_rounded_delay_samples(
        self,
        port,
        sample_frequency_hz,
        granularity_samples,
        max_node_delay_samples,
        measurement_delay_samples=0,
    ):
        channel = 0
        if port is not None:
            if port.port_delay is not None:
                measurement_delay_samples += port.port_delay * sample_frequency_hz
                channel = port.channel
        else:
            self._logger.debug(
                "Port argument of %s is None, please check whether port delays are as specified.",
                self.dev_repr,
            )

        if measurement_delay_samples < 0:
            raise LabOneQControllerException(
                f"Negative node delay for device {self.dev_repr} and channel {channel} specified."
            )
        # Quantize to granularity and round ties towards zero
        measurement_delay_rounded = (
            math.ceil(measurement_delay_samples / granularity_samples + 0.5) - 1
        ) * granularity_samples
        if measurement_delay_rounded > max_node_delay_samples:
            raise LabOneQControllerException(
                f"Maximum delay via {self.dev_repr}'s node is "
                + f"{max_node_delay_samples / sample_frequency_hz * 1e9:.2f} ns - for larger "
                + "values, use the delay_signal property."
            )
        if abs(measurement_delay_samples - measurement_delay_rounded) > 1:
            self._logger.debug(
                "Node delay %.2f ns of %s, channel %d will be rounded to "
                "%.2f ns, a multiple of %.0f samples.",
                measurement_delay_samples / sample_frequency_hz * 1e9,
                self.dev_repr,
                channel,
                measurement_delay_rounded / sample_frequency_hz * 1e9,
                granularity_samples,
            )
        return measurement_delay_rounded


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
