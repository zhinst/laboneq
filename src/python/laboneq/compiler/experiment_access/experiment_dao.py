# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.data.compilation_job import (
    ChunkingInfo,
    DeviceInfo,
    ExperimentInfo,
    SignalInfo,
)

if TYPE_CHECKING:
    from laboneq.dsl.experiment import Experiment


class ExperimentDAO:
    def __init__(self, experiment: ExperimentInfo):
        self.source_experiment: Experiment | None = None
        self.source_experiment = experiment.src
        self.dsl_parameters = experiment.dsl_parameters
        self._uid = experiment.uid
        self._devices: dict[str, DeviceInfo] = {}
        self._chunking: ChunkingInfo | None = None
        self._signals: dict[str, SignalInfo] = {}

        for dev in experiment.devices:
            assert dev.uid not in self._devices
            self._devices[dev.uid] = dev

        for s in experiment.signals:
            self._signals[s.uid] = s
            assert self._devices[s.device.uid] == s.device
            if s.device.uid not in self._devices:
                self._devices[s.device.uid] = s.device

    def signals(self) -> list[str]:
        return sorted([s.uid for s in self._signals.values()])

    def device_info(self, device_id) -> DeviceInfo:
        return self._devices[device_id]

    def device_infos(self) -> list[DeviceInfo]:
        return list(self._devices.values())

    def signal_info(self, signal_id: str) -> SignalInfo:
        return self._signals[signal_id]
