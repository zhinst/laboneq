# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import logging

from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import RawReadoutData
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    RecipeData,
    WaveformItem,
    Waveforms,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IntegratorAllocation
from laboneq.data.scheduled_experiment import ArtifactsCodegen
import numpy as np


_logger = logging.getLogger(__name__)


# value reported by /system/properties/timebase
TIME_STAMP_TIMEBASE = 0.25e-9

SAMPLE_FREQUENCY_HZ = 2.0e9

MAX_AVERAGES_SCOPE = 1 << 16
MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_RESULT_VECTOR_LENGTH = 1 << 19

MAX_WAVEFORM_LENGTH_INTEGRATION = 4096
MAX_WAVEFORM_LENGTH_SPECTROSCOPY = 65536


def ch_uses_lrt(device_uid: str, channel: int, recipe_data: RecipeData) -> bool:
    artifacts = recipe_data.get_artifacts(ArtifactsCodegen)
    osc_signals = set(
        o.signal_id
        for o in recipe_data.recipe.oscillator_params
        if o.device_id == device_uid and o.channel == channel
    )
    return (
        len(
            osc_signals.intersection(
                artifacts.requires_long_readout.get(device_uid, [])
            )
        )
        > 0
    )


@dataclass
class QAChannelNodes:
    output_on: str
    generator_elf_data: str
    generator_elf_progress: str
    generator_enable: str
    generator_ready: str
    osc_freq: list[str]
    readout_result_wave: list[str]
    spectroscopy_result_wave: str
    busy: str


class QAChannel(ChannelBase):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        integrators: int,
        repr_base: str,
    ):
        self._api = api
        self._subscriber = subscriber
        self._device_uid = device_uid
        self._serial = serial
        self._channel = channel
        self._node_base = f"/{serial}/qachannels/{channel}"
        self._unit_repr = f"{repr_base}:qa{channel}"
        self._pipeliner = AwgPipeliner(self._node_base, f"QA{channel}")
        # TODO(2K): Use actual device config to determine number of oscs.
        # Currently the max possible number is hardcoded
        self.nodes = QAChannelNodes(
            output_on=f"{self._node_base}/output/on",
            generator_elf_data=f"{self._node_base}/generator/elf/data",
            generator_elf_progress=f"{self._node_base}/generator/elf/progress",
            generator_enable=f"{self._node_base}/generator/enable",
            generator_ready=f"{self._node_base}/generator/ready",
            osc_freq=[f"{self._node_base}/oscs/{i}/freq" for i in range(6)],
            readout_result_wave=[
                f"{self._node_base}/readout/result/data/{i}/wave"
                for i in range(integrators)
            ],
            spectroscopy_result_wave=f"{self._node_base}/spectroscopy/result/data/wave",
            busy=f"{self._node_base}/busy",
        )

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def pipeliner(self) -> AwgPipeliner:
        return self._pipeliner

    def allocate_resources(self):
        # TODO(2K): Implement channel resources allocation for execution
        pass

    async def load_awg_program(self):
        # TODO(2K): Implement loading of the AWG program.
        return

    def configure_acquisition(
        self,
        recipe_data: RecipeData,
        pipeliner_job: int,
    ) -> NodeCollector:
        rt_execution_info = recipe_data.rt_execution_info
        awg_config = recipe_data.awg_configs[AwgKey(self._device_uid, self._channel)]
        nc = NodeCollector()
        nc.extend(
            self._configure_readout(
                rt_execution_info.acquisition_type,
                awg_config,
                recipe_data.recipe.integrator_allocations,
                rt_execution_info.effective_averages,
                rt_execution_info.effective_averaging_mode,
                recipe_data,
                pipeliner_job,
            )
        )
        nc.extend(
            self._configure_spectroscopy(
                rt_execution_info.acquisition_type,
                awg_config.result_length,
                rt_execution_info.effective_averages,
                rt_execution_info.effective_averaging_mode,
                pipeliner_job,
            )
        )
        if pipeliner_job == 0:
            # The scope is not pipelined, so any job beyond the first cannot reconfigure
            # it.
            nc.extend(
                self._configure_scope(
                    enable=rt_execution_info.is_raw_acquisition,
                    averages=rt_execution_info.effective_averages,
                    acquire_length=awg_config.raw_acquire_length,
                    acquires=awg_config.result_length,
                )
            )
        return nc

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        recipe_data: RecipeData,
        pipeliner_job: int,
    ) -> NodeCollector:
        enable = acquisition_type in [
            AcquisitionType.INTEGRATION,
            AcquisitionType.DISCRIMINATION,
        ]
        nc = NodeCollector(base=f"{self._node_base}/")
        if enable:
            nc.add("readout/result/enable", 0, cache=False)
        if enable and pipeliner_job == 0:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            result_length = awg_config.result_length
            if result_length is None:
                # this AWG core does not acquire results
                return nc
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts {result_length}"
                    f" exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )

            nc.add(
                "readout/result/source",
                # 1 - result_of_integration
                # 3 - result_of_discrimination
                3 if acquisition_type == AcquisitionType.DISCRIMINATION else 1,
            )
            nc.add("readout/result/length", result_length)
            nc.add("readout/result/averages", averages)

            nc.add(
                "readout/result/mode",
                # 0 - cyclic
                # 1 - sequential
                0 if averaging_mode == AveragingMode.CYCLIC else 1,
            )
            if acquisition_type in [
                AcquisitionType.INTEGRATION,
                AcquisitionType.DISCRIMINATION,
            ]:
                uses_lrt = ch_uses_lrt(self._device_uid, self._channel, recipe_data)
                for integrator in integrator_allocations:
                    if (
                        integrator.device_id != self._device_uid
                        or integrator.signal_id not in awg_config.acquire_signals
                    ):
                        continue
                    assert len(integrator.channels) == 1
                    integrator_idx = integrator.channels[0]
                    if uses_lrt:
                        if len(integrator.thresholds) > 1:
                            raise LabOneQControllerException(
                                f"{self._unit_repr}: Multistate discrimination cannot be used with a long readout."
                            )
                        threshold = (
                            0.0
                            if len(integrator.thresholds) == 0
                            else integrator.thresholds[0]
                        )
                        nc.add(
                            f"readout/discriminators/{integrator_idx}/threshold",
                            threshold,
                        )
                        continue
                    for state_i, threshold in enumerate(integrator.thresholds):
                        nc.add(
                            f"readout/multistate/qudits/{integrator_idx}/thresholds/{state_i}/value",
                            threshold or 0.0,
                        )
        nc.barrier()
        nc.add(
            "readout/result/enable",
            1 if enable else 0,
            cache=False,
        )
        return nc

    def _configure_spectroscopy(
        self,
        acq_type: AcquisitionType,
        result_length: int | None,
        averages: int,
        averaging_mode: AveragingMode,
        pipeliner_job: int,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        if result_length is None:
            return nc  # this AWG does not acquire results
        if is_spectroscopy(acq_type) and pipeliner_job == 0:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            if result_length is None:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts is not defined for spectroscopy."
                )
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts {result_length}"
                    f" exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )
            nc.add("spectroscopy/result/length", result_length)
            nc.add("spectroscopy/result/averages", averages)
            nc.add(
                "spectroscopy/result/mode",
                # 0 - "cyclic"
                # 1 - "sequential"
                0 if averaging_mode == AveragingMode.CYCLIC else 1,
            )
            nc.add("spectroscopy/psd/enable", 0)
        if is_spectroscopy(acq_type):
            nc.add("spectroscopy/result/enable", 0)
        if acq_type == AcquisitionType.SPECTROSCOPY_PSD:
            nc.add("spectroscopy/psd/enable", 1)
        nc.barrier()
        nc.add(
            "spectroscopy/result/enable",
            1 if is_spectroscopy(acq_type) else 0,
        )
        return nc

    def _configure_scope(
        self,
        enable: bool,
        averages: int,
        acquire_length: int | None,
        acquires: int | None,
    ) -> NodeCollector:
        # TODO(2K): multiple acquire events
        nc = NodeCollector(base=f"/{self._serial}/scopes/0/")
        if enable:
            if acquire_length is None:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Raw acquire length is not defined."
                )
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            nc.add("time", 0)  # 0 -> 2 GSa/s
            nc.add("averaging/enable", 1)
            nc.add("averaging/count", averages)
            nc.add(f"channels/{self._channel}/enable", 1)
            nc.add(
                f"channels/{self._channel}/inputselect", self._channel
            )  # channelN_signal_input
            if acquires is not None and acquires > 1:
                nc.add("segments/enable", 1)
                nc.add("segments/count", acquires)
            else:
                nc.add("segments/enable", 0)
            nc.barrier()
            # Length has to be set after the segments are configured, as the maximum length
            # (of a segment) is determined by the number of segments.
            # Scope length has a granularity of 16.
            scope_length = (acquire_length + 0xF) & (~0xF)
            nc.add("length", scope_length)
            # TODO(2K): only one trigger is possible for all channels. Which one to use?
            nc.add("trigger/channel", 64 + self._channel)  # channelN_sequencer_monitor0
            nc.add("trigger/enable", 1)
            nc.add("enable", 0)  # todo: barrier needed?
            nc.add("single", 1, cache=False)
        nc.add("enable", 1 if enable else 0)
        return nc

    def validate_spectroscopy_waves(self, waves: Waveforms) -> WaveformItem | None:
        if len(waves) > 1:
            raise LabOneQControllerException(
                f"{self._unit_repr}: Only one envelope waveform per physical channel is "
                "possible in spectroscopy mode. Check play commands for the channel."
            )
        max_len = MAX_WAVEFORM_LENGTH_SPECTROSCOPY
        for wave in waves:
            wave_len = len(wave.samples)
            if wave_len > max_len:
                max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Length {wave_len} of the spectroscopy envelope waveform "
                    f"'{wave.name}' exceeds maximum of {max_len} samples. Ensure measure pulse doesn't "
                    f"exceed {max_pulse_len * 1e6:.3f} us."
                )
            return wave
        return None

    def validate_generator_waves(self, waves: Waveforms):
        max_len = MAX_WAVEFORM_LENGTH_INTEGRATION
        for wave in waves:
            wave_len = len(wave.samples)
            if wave_len > max_len:
                max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                raise LabOneQControllerException(
                    f"{self._unit_repr}:slot{wave.index} Length {wave_len} of the generator waveform "
                    f"'{wave.name}' exceeds maximum of {max_len} samples. Ensure measure pulse doesn't "
                    f"exceed {max_pulse_len * 1e6:.3f} us."
                )

    def upload_spectroscopy_envelope(
        self,
        wave: WaveformItem,
    ) -> NodeCollector:
        return NodeCollector.one(
            path=f"{self._node_base}/spectroscopy/envelope/wave",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )

    def upload_generator_wave(
        self,
        wave: WaveformItem,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add(
            path=f"generator/waveforms/{wave.index}/wave",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )
        if wave.hold_start is not None:
            assert wave.hold_length is not None
            hold_base = f"generator/waveforms/{wave.index}/hold"
            nc.add(f"{hold_base}/enable", 1, cache=False)
            nc.add(f"{hold_base}/samples/startindex", wave.hold_start, cache=False)
            nc.add(f"{hold_base}/samples/length", wave.hold_length, cache=False)
        return nc

    def prepare_upload_all_binary_waves(
        self,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        has_spectroscopy_envelope = False
        if is_spectroscopy(acquisition_type):
            wave = self.validate_spectroscopy_waves(waves)
            if wave is not None:
                has_spectroscopy_envelope = True
                nc.extend(self.upload_spectroscopy_envelope(wave))
        else:
            self.validate_generator_waves(waves)
            nc.add("generator/clearwave", 1, cache=False)
            for wave in waves:
                nc.extend(self.upload_generator_wave(wave))
        nc.add("spectroscopy/envelope/enable", 1 if has_spectroscopy_envelope else 0)
        return nc

    def disable_output(self) -> NodeCollector:
        return NodeCollector.one(self.nodes.output_on, 0, cache=False)

    def subscribe_nodes(self) -> NodeCollector:
        nc = NodeCollector()
        for path in self.nodes.readout_result_wave:
            nc.add_path(path)
        nc.add_path(self.nodes.spectroscopy_result_wave)
        return nc

    async def _read_all_jobs_result(
        self,
        result_path: str,
        ch_repr: str,
        pipeliner_jobs: int,
        num_results: int,
        timeout_s: float,
    ) -> RawReadoutData:
        rt_result = RawReadoutData(
            np.full(pipeliner_jobs * num_results, np.nan, dtype=np.complex128)
        )
        jobs_processed: set[int] = set()

        try:
            while True:
                if len(jobs_processed) == pipeliner_jobs:
                    break
                job_result = await self._subscriber.get_result(
                    result_path, timeout_s=timeout_s
                )
                properties = job_result.properties

                job_id = properties["jobid"]
                if job_id in jobs_processed:
                    _logger.error(
                        f"{ch_repr}: Ignoring duplicate job id {job_id} in the results."
                    )
                    continue
                if job_id >= pipeliner_jobs:
                    _logger.error(
                        f"{ch_repr}: Ignoring job id {job_id} in the results, as it "
                        f"falls outside the defined range of {pipeliner_jobs} jobs."
                    )
                    continue
                jobs_processed.add(job_id)

                num_samples = properties.get("numsamples", num_results)

                if num_samples != num_results:
                    _logger.error(
                        f"{ch_repr}: The number of measurements acquired ({num_samples}) "
                        f"does not match the number of measurements defined ({num_results}). "
                        "Possibly the time between measurements within a loop is too short, "
                        "or the measurement was not started."
                    )

                valid_samples = min(num_results, num_samples)
                np.put(
                    rt_result.vector,
                    range(job_id * num_results, job_id * num_results + valid_samples),
                    job_result.vector[:valid_samples],
                    mode="clip",
                )

                timestamp_clock_cycles = properties["firstSampleTimestamp"]
                rt_result.metadata.setdefault(job_id, {})["timestamp"] = (
                    timestamp_clock_cycles * TIME_STAMP_TIMEBASE
                )
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{ch_repr}: Failed to receive all results within {timeout_s} s, timing out."
            )

        missing_jobs = set(range(pipeliner_jobs)) - jobs_processed
        if len(missing_jobs) > 0:
            _logger.error(
                f"{ch_repr}: Results for job id(s) {missing_jobs} are missing."
            )

        return rt_result

    async def get_readout_data(
        self,
        pipeliner_jobs: int,
        num_results: int,
        timeout_s: float,
        integrator: int,
    ) -> RawReadoutData:
        return await self._read_all_jobs_result(
            result_path=self.nodes.readout_result_wave[integrator],
            ch_repr=f"{self._unit_repr}:readout{integrator}",
            pipeliner_jobs=pipeliner_jobs,
            num_results=num_results,
            timeout_s=timeout_s,
        )

    async def get_spectroscopy_data(
        self,
        pipeliner_jobs: int,
        num_results: int,
        timeout_s: float,
    ) -> RawReadoutData:
        return await self._read_all_jobs_result(
            result_path=self.nodes.spectroscopy_result_wave,
            ch_repr=f"{self._unit_repr}:spectroscopy",
            pipeliner_jobs=pipeliner_jobs,
            num_results=num_results,
            timeout_s=timeout_s,
        )
