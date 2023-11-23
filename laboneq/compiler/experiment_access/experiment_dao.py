# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import deque
from typing import Any, List

from jsonschema import ValidationError

from laboneq._utils import cached_method
from laboneq.compiler import DeviceType
from laboneq.compiler.experiment_access import json_dumper
from laboneq.compiler.experiment_access.experiment_info_loader import (
    ExperimentInfoLoader,
)
from laboneq.compiler.experiment_access.json_loader import JsonLoader
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType, ExecutionType
from laboneq.core.validators import dicts_equal
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
    DeviceInfo,
    DeviceInfoType,
    ExperimentInfo,
    OscillatorInfo,
    ParameterInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
    SignalRange,
)

_logger = logging.getLogger(__name__)


class ExperimentDAO:
    def __init__(self, experiment, loader=None):
        self._data: dict[str, Any] = {}
        self._acquisition_type: AcquisitionType = None  # type: ignore

        if loader is not None:
            assert experiment is None, "Cannot pass both experiment and inject a loader"
            self._loader = loader
        elif isinstance(experiment, ExperimentInfo):
            self._loader = self._load_experiment_info(experiment)
            self._uid = experiment.uid
        else:
            self._loader = self._load_experiment(experiment)
            self._uid = "exp_from_json"
        self._data = self._loader.data()
        self._acquisition_type = self._loader.acquisition_type

        self.validate_experiment()

    def to_experiment_info(self):
        return ExperimentInfo(
            uid=self._uid,
            devices=list(self._data["devices"].values()),
            signals=list(self._data["signals"].values()),
            sections=list(self._data["sections"].values()),
            global_leader_device=self._data["devices"][self.global_leader_device()]
            if self.global_leader_device() is not None
            else None,
            pulse_defs=list(self._data["pulses"].values()),
        )

    def __eq__(self, other):
        if not isinstance(other, ExperimentDAO):
            return False

        return self._acquisition_type == other._acquisition_type and dicts_equal(
            self._data, other._data
        )

    def add_signal(self, device_id, channels, signal_id, signal_type):
        assert signal_id not in self._data["signals"]

        self._data["signals"][signal_id] = SignalInfo(
            uid=signal_id,
            type=SignalInfoType(signal_type),
            device=self.device_info(device_id),
            channels=channels,
        )

    def _load_experiment_info(self, experiment: ExperimentInfo) -> ExperimentInfoLoader:
        loader = ExperimentInfoLoader()
        loader.load(experiment)
        return loader

    def _load_experiment(self, experiment) -> JsonLoader:
        loader = JsonLoader()
        try:
            validator = loader.schema_validator()
            validator.validate(experiment)
        except ValidationError as exception:
            _logger.warning("Failed to validate input:")
            for line in str(exception).splitlines():
                _logger.warning("validation error: %s", line)
        loader.load(experiment)
        return loader

    @staticmethod
    def dump(experiment_dao: "ExperimentDAO"):
        return json_dumper.dump(experiment_dao)

    @property
    def acquisition_type(self) -> AcquisitionType:
        return self._acquisition_type

    def signals(self) -> list[str]:
        return sorted([s.uid for s in self._data["signals"].values()])

    def devices(self) -> List[str]:
        return [d.uid for d in self._data["devices"].values()]

    def global_leader_device(self) -> str:
        return self._data["global_leader_device_id"]

    def device_info(self, device_id) -> DeviceInfo | None:
        return self._data["devices"].get(device_id)

    def device_infos(self) -> List[DeviceInfo]:
        return list(self._data["devices"].values())

    def device_reference_clock(self, device_id):
        return self._data["devices"][device_id].reference_clock

    def device_from_signal(self, signal_id):
        return self.device_info(self.signal_info(signal_id).device.uid)

    @cached_method()
    def signal_info(self, signal_id) -> SignalInfo | None:
        return self._data["signals"].get(signal_id)

    def sections(self) -> List[str]:
        return list(self._data["sections"].keys())

    def section_info(self, section_id) -> SectionInfo:
        retval = self._data["sections"][section_id]
        return retval

    def root_sections(self):
        return self._data["root_sections"]

    @cached_method()
    def _has_near_time_child(self, section_id) -> str | None:
        children = self.direct_section_children(section_id)
        for child in children:
            child_info = self.section_info(child)
            if child_info.execution_type == "controller":
                return child
            child_contains_nt = self._has_near_time_child(child)
            if child_contains_nt:
                return child_contains_nt
        return None

    @cached_method()
    def root_rt_sections(self):
        retval = []
        queue = deque(self.root_sections())
        while len(queue):
            candidate = queue.popleft()
            info = self.section_info(candidate)
            nt_subsection = self._has_near_time_child(candidate)
            if info.execution_type in (None, ExecutionType.REAL_TIME):
                if nt_subsection is not None:
                    raise LabOneQException(
                        f"Real-time section {candidate} has near-time sub-section "
                        f"{nt_subsection}."
                    )
                retval.append(candidate)
            else:
                queue.extend(self.direct_section_children(candidate))
        return tuple(retval)  # tuple is immutable, so no one can break memoization

    @cached_method()
    def direct_section_children(self, section_id) -> List[str]:
        return [child.uid for child in self.section_info(section_id).children]

    @cached_method()
    def all_section_children(self, section_id):
        retval = []

        direct_children = self.direct_section_children(section_id)
        retval = retval + direct_children
        for child in direct_children:
            retval = retval + list(self.all_section_children(child))

        return set(retval)

    @cached_method()
    def section_parent(self, section_id) -> str | None:
        for parent_id in self.sections():
            parent = self.section_info(parent_id)
            if any(child.uid == section_id for child in parent.children):
                return parent.uid
        return None

    def pqscs(self) -> list[str]:
        return [
            d.uid for d in self.device_infos() if d.device_type == DeviceInfoType.PQSC
        ]

    def pqsc_ports(self, pqsc_device_uid: str):
        assert pqsc_device_uid in self.pqscs()
        leader = self.device_info(pqsc_device_uid)
        return [{"device": p.device.uid, "port": p.port} for p in leader.followers]

    def dio_followers(self) -> list[str]:
        return [
            follower.device.uid
            for leader in self.device_infos()
            for follower in leader.followers
            if leader.device_type != DeviceInfoType.PQSC
        ]

    def dio_leader(self, device_id) -> str | None:
        for d in self.device_infos():
            if d.device_type == DeviceInfoType.PQSC:
                continue
            for f in d.followers:
                if f.device.uid == device_id:
                    return d.uid

        return None

    def dio_connections(self) -> list[tuple[str, str]]:
        return [
            (leader.uid, follower.device.uid)
            for leader in self.device_infos()
            for follower in leader.followers
            if leader.device_type != DeviceInfoType.PQSC
        ]

    def section_signals(self, section_id):
        return {s.uid for s in self.section_info(section_id).signals}

    @cached_method()
    def section_signals_with_children(self, section_id):
        retval = set()
        section_with_children = self.all_section_children(section_id)
        section_with_children.add(section_id)
        for child in section_with_children:
            retval |= self.section_signals(child)
        return retval

    def pulses(self) -> list[str]:
        return list(self._data["pulses"].keys())

    def pulse(self, pulse_id) -> PulseDef:
        return self._data["pulses"].get(pulse_id)

    def oscillator_info(self, oscillator_id) -> OscillatorInfo | None:
        return self._data["oscillators"].get(oscillator_id)

    def hardware_oscillators(self) -> List[OscillatorInfo]:
        oscillator_infos: List[OscillatorInfo] = []
        for device in self.devices():
            device_oscillators = self.device_oscillators(device)
            for oscillator_id in device_oscillators:
                info = self.oscillator_info(oscillator_id)
                if info is not None and info.is_hardware:
                    info.device_id = device
                    oscillator_infos.append(info)

        return sorted(oscillator_infos, key=lambda x: x.uid)

    def device_oscillators(self, device_id) -> list[OscillatorInfo]:
        return [
            do["oscillator_id"]
            for do in self._data["device_oscillators"].get(device_id, [])
        ]

    def oscillators(self):
        return list(self._data["oscillators"].keys())

    def signal_oscillator(self, signal_id):
        return self._data["signals"][signal_id].oscillator

    def voltage_offset(self, signal_id) -> float | ParameterInfo:
        return self._data["signals"][signal_id].voltage_offset

    def mixer_calibration(self, signal_id):
        return self._data["signals"][signal_id].mixer_calibration

    def precompensation(self, signal_id):
        return self._data["signals"][signal_id].precompensation

    def lo_frequency(self, signal_id) -> float | ParameterInfo:
        return self._data["signals"][signal_id].lo_frequency

    def signal_range(self, signal_id) -> SignalRange:
        return self._data["signals"][signal_id].signal_range

    def port_delay(self, signal_id) -> float | ParameterInfo | None:
        return self._data["signals"][signal_id].port_delay

    def port_mode(self, signal_id):
        return self._data["signals"][signal_id].port_mode

    def threshold(self, signal_id):
        return self._data["signals"][signal_id].threshold

    def amplitude(self, signal_id) -> float | ParameterInfo | None:
        return self._data["signals"][signal_id].amplitude

    def amplifier_pump(self, signal_id) -> AmplifierPumpInfo | None:
        return self._data["signals"][signal_id].amplifier_pump

    def section_pulses(self, section_id, signal_id) -> list[SectionSignalPulse]:
        return (
            self._data["section_signal_pulses"].get(section_id, {}).get(signal_id, [])
        )

    def markers_on_signal(self, signal_id: str):
        return self._data["signal_markers"].get(signal_id)

    def triggers_on_signal(self, signal_id: str):
        return self._data["signal_trigger"].get(signal_id)

    def section_parameters(self, section_id) -> list[ParameterInfo]:
        return self._data["section_parameters"].get(section_id, [])

    def validate_experiment(self):
        all_parameters = set()
        for section_id in self.sections():
            for parameter in self.section_parameters(section_id):
                all_parameters.add(parameter.uid)

        for section_id in self.sections():
            for signal_id in self.section_signals(section_id):
                for section_pulse in self.section_pulses(section_id, signal_id):
                    if section_pulse.pulse is None:
                        continue
                    pulse_id = section_pulse.pulse.uid

                    if (
                        DeviceType.from_device_info_type(
                            section_pulse.signal.device.device_type
                        ).is_qa_device
                    ) and not len(section_pulse.markers) == 0:
                        raise RuntimeError(
                            f"Pulse {pulse_id} referenced in section {section_id}"
                            f" has markers but is to be played on a QA device. QA"
                            f" devices do not support markers."
                        )

    def acquisition_signal(self, handle: str) -> str | None:
        return self._data["handle_acquires"][handle]
