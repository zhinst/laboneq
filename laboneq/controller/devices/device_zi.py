# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import floor
import os
import os.path
import time
import json
import math

from typing import List, Optional, Tuple, Any, Set, TYPE_CHECKING
from laboneq.controller.recipe_1_4_0 import Initialization, OscillatorParam
from laboneq.controller.recipe_processor import RecipeData, RtExecutionInfo

from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.communication import (
    AwgModuleWrapper,
    DaqNodeAction,
    DaqNodeSetAction,
    DaqNodeGetAction,
    DaqNodeWaitAction,
    DaqWrapper,
    CachingStrategy,
)
from laboneq.core.types.enums.acquisition_type import AcquisitionType

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment

from laboneq.controller.devices.device_base import DeviceBase, DeviceQualifier
import zhinst.utils
import numpy as np
from numpy import typing as npt


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


class DeviceZI(DeviceBase):
    def __init__(self, device_qualifier: DeviceQualifier):
        super().__init__(device_qualifier)

        self._daq: DaqWrapper = device_qualifier.server
        self.dev_type = None
        self.dev_opts = []
        self._connected = False
        self._allocated_oscs: List[AllocatedOscillator] = []
        self._allocated_awgs: Set[int] = set()

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

    @property  # Overrides one from DeviceBase
    def dev_repr(self) -> str:
        return f"{self._device_qualifier.driver}:{self._get_option('serial')}"

    @property  # Overrides one from DeviceBase
    def has_awg(self) -> bool:
        return len(self._awg_modules) > 0

    @property
    def serial(self):
        return self._get_option("serial").lower()

    @property
    def interface(self):
        return self._get_option("interface").lower()

    @property
    def daq(self):
        return self._daq

    def _process_dev_opts(self):
        pass

    def _get_sequencer_type(self) -> str:
        return "auto-detect"

    def connect(self):
        if self._connected:
            return

        self._logger.debug(
            "%s: Connecting to %s interface.", self.dev_repr, self.interface
        )
        try:
            self._daq.connectDevice(self.serial, self.interface)
        except RuntimeError as exp:
            raise LabOneQControllerException(f"{self.dev_repr}: {str(exp)}")

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

        for i in range(self._get_num_AWGs()):
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

        self._connected = True

    def free_allocations(self):
        self._allocated_oscs.clear()
        self._allocated_awgs.clear()

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
                    f"{self.dev_repr}: exceeded the number of available oscillators for channel {osc_param.channel}"
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
                    f"{self.dev_repr}: ambiguous frequency in recipe for oscillator '{osc_param.id}': {same_id_osc.frequency} != {osc_param.frequency}"
                )
            same_id_osc.channels.add(osc_param.channel)

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
                        f"{self.dev_repr}: Node '{path}' didn't switch to '{expected}' within {timeout}s. Last value: {last_val}"
                    )
                if now - last_report > 5:
                    self._logger.debug(
                        "Waiting for node '%s' switching to '%s', %f s remaining until %f s timeout...",
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
                        f"Unable to switch reference clock within {timeout}s. Requested source: {source},"
                        f" actual: {sourceactual}, status: {status}, expected frequencies: {expected_freqs}, actual: {freq}"
                    )
                if now - last_report > 5:
                    self._logger.debug(
                        "Waiting for reference clock switching, %f s remaining until %f s timeout...",
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
                        f"Unable to switch reference clock, device returned error after {elapsed}s. Requested source: {source}, actual: {sourceactual}, status: {status}, expected frequency: {expected_freqs}, actual: {freq}"
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
                    f"Unexpected frequency after switching the reference clock. Requested source: {source}, actual: {sourceactual}, status: {status}, expected frequency: {expected_freqs}, actual: {freq}"
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
                    "%s: Compilation successful on AWG #%d with no warnings, will upload the program to the instrument.",
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

            if (
                compiler_status == AwgCompilerStatus.ERROR.value
                or compiler_status == AwgCompilerStatus.WARNING.value
            ):
                raise LabOneQControllerException(
                    f"{self.dev_repr}: AWG compilation failed, compiler output:\n{status_string}"
                )
            time.sleep(0.1)

    def _prepare_wave_iq(self, waves, sig):
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

        return (
            sig,
            zhinst.utils.convert_awg_waveform(
                np.clip(wave_i["samples"], -1, 1), np.clip(wave_q["samples"], -1, 1)
            ),
        )

    def _prepare_wave_single(self, waves, sig):
        wave = next((w for w in waves if w["filename"] == f"{sig}.wave"), None)
        if not wave:
            raise LabOneQControllerException(f"Wave not found, signature '{sig}'")

        return sig, zhinst.utils.convert_awg_waveform(np.clip(wave["samples"], -1, 1))

    def _prepare_wave_complex(self, waves, sig):
        filename_to_find = f"{sig}.wave"
        wave = next((w for w in waves if w["filename"] == filename_to_find), None)
        if not wave:
            raise LabOneQControllerException(
                f"Wave not found, signature '{sig}' filename '{filename_to_find}'"
            )

        return sig, np.array(wave["samples"], dtype=np.complex128)

    def _prepare_waves(self, compiled: CompiledExperiment, seqc_filename: str):
        wave_indices_filename = os.path.splitext(seqc_filename)[0] + "_waveindices.csv"
        wave_indices = next(
            (
                i
                for i in compiled.wave_indices
                if i["filename"] == wave_indices_filename
            ),
            {"value": {}},
        )["value"]

        waves_by_index = {}
        waves = compiled.waves or []
        for sig, (idx, sigtype) in wave_indices.items():
            if sigtype in ["iq", "double", "multi"]:
                waves_by_index[idx] = self._prepare_wave_iq(waves, sig)
            elif sigtype == "single":
                waves_by_index[idx] = self._prepare_wave_single(waves, sig)
            elif sigtype == "complex":
                waves_by_index[idx] = self._prepare_wave_complex(waves, sig)
            else:
                raise LabOneQControllerException(
                    f"Unexpected signal type for binary wave for '{sig}' in '{seqc_filename}' - '{sigtype}', should be one of [iq, double, multi, single, complex]"
                )

        bin_waves = []
        idx = 0
        while idx in waves_by_index:
            bin_waves.append(waves_by_index[idx])
            idx += 1
        return bin_waves

    def _prepare_seqc(
        self, seqc_filename: str, compiled: CompiledExperiment
    ) -> Tuple[str, str, List[Any]]:
        # 'compiled' expected to have the following members:
        #  - 'src'   -> List[Dict[str, str]]
        #                 'filename' -> '<seqc_filename>'
        #                 'text'     -> '<seqc_content>'
        #  - 'waves' -> List[Dict[str, str]]
        #                 'filename' -> '<wave_filename_csv>'
        #                 'text'     -> '<wave_content_csv>'
        #
        # returns:
        #  - 1) str: seqc text to pass to the awg compiler
        #  - 2) list[array]: waves to upload to the instrument (ordered by index)
        seqc = next((s for s in compiled.src if s["filename"] == seqc_filename), None)
        if seqc is None:
            raise LabOneQControllerException(
                f"SeqC program '{seqc_filename}' not found"
            )

        bin_waves = self._prepare_waves(compiled, seqc_filename)
        return seqc["text"], bin_waves

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

    def _upload_all_binary_waves(
        self, awg_index, waves, acquisition_type: AcquisitionType
    ):
        # Default implementation for "old" devices, override for newer devices
        self._daq.batch_set(
            [
                self.prepare_upload_binary_wave(
                    filename=w[0],
                    waveform=w[1],
                    awg_index=awg_index,
                    wave_index=i,
                    acquisition_type=acquisition_type,
                )
                for i, w in enumerate(waves)
            ]
        )

    def upload_awg_program(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        if initialization.awgs is not None:
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

                data, waves = self._prepare_seqc(awg_obj.seqc, recipe_data.compiled)

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
                                data,
                                filename=awg_obj.seqc,
                                caching_strategy=CachingStrategy.NO_CACHE,  # if only external waves changed
                            ),
                        ]
                    )
                except LabOneQControllerException as exp:
                    raise LabOneQControllerException(
                        f"Exception raised while uploading program from file {awg_obj.seqc} to AWG #{awg_index}\n"
                        f"details:\n{str(exp)}\n{data}\n"
                    )
                # TODO(2K): handle timeout, emit:
                # f"{str(exp)}\nAWG compiler timed out while trying to compile:\n{data}\n"
                self._check_awg_compiler_status(awg_index)
                self._wait_for_elf_upload(awg_index)
                self._upload_all_binary_waves(awg_index, waves, acquisition_type)

    def _get_num_AWGs(self):
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

    def collect_awg_after_upload_nodes(self, initialization: Initialization.Data):
        return []

    def collect_conditions_to_close_loop(self, acquisition_units):
        return [
            DaqNodeWaitAction(self._daq, f"/{self.serial}/awgs/{awg_index}/enable", 0)
            for awg_index in self._allocated_awgs
        ]

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

    def collect_reset_nodes(self):
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
            raise Exception(
                f"Negative node delay for device {self.dev_repr} and channel {channel} specified."
            )
        # Quantize to granularity and round ties towards zero
        measurement_delay_rounded = (
            math.ceil(measurement_delay_samples / granularity_samples + 0.5) - 1
        ) * granularity_samples
        if measurement_delay_rounded > max_node_delay_samples:
            raise Exception(
                f"Maximum delay via {self.dev_repr}'s node is "
                + f"{max_node_delay_samples / sample_frequency_hz * 1e9:.2f} ns - for larger "
                + "values, use the delay_signal property."
            )
        if abs(measurement_delay_samples - measurement_delay_rounded) > 1:
            self._logger.debug(
                "Node delay %.2f ns of %s, channel %d will be rounded to "
                + "%.2f ns, a multiple of %.0f samples.",
                (measurement_delay_samples / sample_frequency_hz * 1e9),
                self.dev_repr,
                channel,
                (measurement_delay_rounded / sample_frequency_hz * 1e9),
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
        # the key in error["vector"] looks like a dict, but it's a string. so we have to use json.loads to convert it into a dict.
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
            f"An error happened on device {serial} during the execution of the experiment. Error messages:\n{all_messages}"
        )
        # should we return the warnings in the log?
