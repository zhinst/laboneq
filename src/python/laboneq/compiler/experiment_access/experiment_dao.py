# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.core.types.enums import AcquisitionType
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
    ChunkingInfo,
    DeviceInfo,
    DeviceInfoType,
    ExperimentInfo,
    ParameterInfo,
    SignalInfo,
    SignalInfoType,
    SignalRange,
)

if TYPE_CHECKING:
    from laboneq.data.experiment_description import Experiment
    from laboneq.data.parameter import Parameter


class ExperimentDAO:
    def __init__(self, experiment: ExperimentInfo):
        self.source_experiment: Experiment | None = None
        self.dsl_parameters: list[Parameter] = []
        self.source_experiment = experiment.src
        self.dsl_parameters = experiment.dsl_parameters
        self._uid = experiment.uid
        self._acquisition_type: AcquisitionType = (
            experiment.acquisition_type or AcquisitionType.INTEGRATION
        )
        self._devices: dict[str, DeviceInfo] = {}
        self._chunking: ChunkingInfo | None = None
        self._signals: dict[str, SignalInfo] = {}
        self._global_leader_device_id: str | None = None

        self._global_leader_device_id = (
            experiment.global_leader_device.uid
            if experiment.global_leader_device is not None
            else None
        )

        for dev in experiment.devices:
            assert dev.uid not in self._devices
            self._devices[dev.uid] = dev

        for s in experiment.signals:
            self._signals[s.uid] = s
            assert self._devices[s.device.uid] == s.device
            if s.device.uid not in self._devices:
                self._devices[s.device.uid] = s.device

    def add_signal(self, device_id, channels, signal_id, signal_type):
        assert signal_id not in self._signals

        self._signals[signal_id] = SignalInfo(
            uid=signal_id,
            type=SignalInfoType(signal_type),
            device=self.device_info(device_id),
            channels=channels,
        )

    @property
    def acquisition_type(self) -> AcquisitionType:
        return self._acquisition_type

    def signals(self) -> list[str]:
        return sorted([s.uid for s in self._signals.values()])

    def global_leader_device(self) -> str:
        return self._global_leader_device_id

    def device_info(self, device_id) -> DeviceInfo:
        return self._devices[device_id]

    def device_infos(self) -> list[DeviceInfo]:
        return list(self._devices.values())

    def signal_info(self, signal_id: str) -> SignalInfo:
        return self._signals[signal_id]

    def pqscs(self) -> list[str]:
        return [
            d.uid
            for d in self.device_infos()
            if d.device_type in [DeviceInfoType.PQSC, DeviceInfoType.QHUB]
        ]

    def pqsc_followers(self, pqsc_device_uid: str) -> list[str]:
        assert pqsc_device_uid in self.pqscs()
        return [
            d.uid
            for d in self.device_infos()
            if d.device_type
            in [DeviceInfoType.HDAWG, DeviceInfoType.SHFQA, DeviceInfoType.SHFSG]
        ]

    def dio_followers(self) -> list[str]:
        return [
            follower_uid
            for leader in self.device_infos()
            for follower_uid in leader.followers
            if leader.device_type not in [DeviceInfoType.PQSC, DeviceInfoType.QHUB]
        ]

    def signal_oscillator(self, signal_id):
        return self._signals[signal_id].oscillator

    def voltage_offset(self, signal_id) -> float | ParameterInfo:
        return self._signals[signal_id].voltage_offset

    def mixer_calibration(self, signal_id):
        return self._signals[signal_id].mixer_calibration

    def precompensation(self, signal_id):
        return self._signals[signal_id].precompensation

    def lo_frequency(self, signal_id) -> float | ParameterInfo:
        return self._signals[signal_id].lo_frequency

    def signal_range(self, signal_id) -> SignalRange:
        return self._signals[signal_id].signal_range

    def port_delay(self, signal_id) -> float | ParameterInfo | None:
        return self._signals[signal_id].port_delay

    def port_mode(self, signal_id):
        return self._signals[signal_id].port_mode

    def threshold(self, signal_id):
        return self._signals[signal_id].threshold

    def amplitude(self, signal_id) -> float | ParameterInfo | None:
        return self._signals[signal_id].amplitude

    def amplifier_pump(self, signal_id) -> AmplifierPumpInfo | None:
        return self._signals[signal_id].amplifier_pump
