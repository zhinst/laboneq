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
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import RawReadoutData
import numpy as np


_logger = logging.getLogger(__name__)


# value reported by /system/properties/timebase
TIME_STAMP_TIMEBASE = 0.25e-9


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


class QAChannel:
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        serial: str,
        channel: int,
        integrators: int,
        repr_base: str,
    ):
        self._api = api
        self._subscriber = subscriber
        self._channel = channel
        self._node_base = f"/{serial}/qachannels/{channel}"
        self._unit_repr = f"{repr_base}:qa{channel}"
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
