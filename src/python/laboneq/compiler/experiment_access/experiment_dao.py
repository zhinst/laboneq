# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from laboneq._utils import cached_method
from laboneq.compiler.experiment_access.experiment_info_loader import (
    ExperimentInfoLoader,
)
from laboneq.core.types.enums import AcquisitionType
from laboneq.core.validators import dicts_equal
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
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
        self._loader = self._load_experiment_info(experiment)
        self._uid = experiment.uid
        self._data = self._loader.data()
        self._acquisition_type: AcquisitionType = self._loader.acquisition_type

    def to_experiment_info(self):
        return ExperimentInfo(
            uid=self._uid,
            device_setup_fingerprint="",  # Not used in this context
            devices=list(self._data.devices.values()),
            signals=list(self._data.signals.values()),
            sections=list(self._data.sections.values()),
            chunking=self._data.chunking,
            global_leader_device=self._data.devices[self.global_leader_device()]
            if self.global_leader_device() is not None
            else None,
        )

    def __eq__(self, other):
        if not isinstance(other, ExperimentDAO):
            return False

        return self._acquisition_type == other._acquisition_type and dicts_equal(
            asdict(self._data), asdict(other._data)
        )

    def add_signal(self, device_id, channels, signal_id, signal_type):
        assert signal_id not in self._data.signals

        self._data.signals[signal_id] = SignalInfo(
            uid=signal_id,
            type=SignalInfoType(signal_type),
            device=self.device_info(device_id),
            channels=channels,
        )

    def _load_experiment_info(self, experiment: ExperimentInfo) -> ExperimentInfoLoader:
        loader = ExperimentInfoLoader()
        loader.load(experiment)
        return loader

    @property
    def acquisition_type(self) -> AcquisitionType:
        return self._acquisition_type

    def signals(self) -> list[str]:
        return sorted([s.uid for s in self._data.signals.values()])

    def devices(self) -> list[str]:
        return [d.uid for d in self._data.devices.values()]

    def global_leader_device(self) -> str:
        return self._data.global_leader_device_id

    def device_info(self, device_id) -> DeviceInfo:
        return self._data.devices[device_id]

    def device_infos(self) -> list[DeviceInfo]:
        return list(self._data.devices.values())

    def device_from_signal(self, signal_id) -> DeviceInfo:
        return self.device_info(self.signal_info(signal_id).device.uid)

    @cached_method()
    def signal_info(self, signal_id: str) -> SignalInfo:
        return self._data.signals[signal_id]

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

    def dio_leader(self, device_id) -> str | None:
        for d in self.device_infos():
            if d.device_type in [DeviceInfoType.PQSC, DeviceInfoType.QHUB]:
                continue
            if device_id in d.followers:
                return d.uid

        return None

    def signal_oscillator(self, signal_id):
        return self._data.signals[signal_id].oscillator

    def voltage_offset(self, signal_id) -> float | ParameterInfo:
        return self._data.signals[signal_id].voltage_offset

    def mixer_calibration(self, signal_id):
        return self._data.signals[signal_id].mixer_calibration

    def precompensation(self, signal_id):
        return self._data.signals[signal_id].precompensation

    def lo_frequency(self, signal_id) -> float | ParameterInfo:
        return self._data.signals[signal_id].lo_frequency

    def signal_range(self, signal_id) -> SignalRange:
        return self._data.signals[signal_id].signal_range

    def port_delay(self, signal_id) -> float | ParameterInfo | None:
        return self._data.signals[signal_id].port_delay

    def port_mode(self, signal_id):
        return self._data.signals[signal_id].port_mode

    def threshold(self, signal_id):
        return self._data.signals[signal_id].threshold

    def amplitude(self, signal_id) -> float | ParameterInfo | None:
        return self._data.signals[signal_id].amplitude

    def amplifier_pump(self, signal_id) -> AmplifierPumpInfo | None:
        return self._data.signals[signal_id].amplifier_pump

    def markers_on_signal(self, signal_id: str) -> set[str] | None:
        return self._data.signal_markers.get(signal_id)

    def triggers_on_signal(self, signal_id: str) -> int | None:
        return self._data.signal_trigger.get(signal_id)

    def parameter_map(self) -> dict[str, ParameterInfo]:
        return {
            param.uid: param
            for params in self._data.section_parameters.values()
            for param in params
        }
