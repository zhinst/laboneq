# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict
import logging
from collections import deque

from jsonschema import ValidationError

from laboneq._utils import cached_method
from laboneq.compiler.experiment_access import json_dumper, validators
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
        if loader is not None:
            assert experiment is None, "Cannot pass both experiment and inject a loader"
            self._loader = loader
            self._uid = "exp_from_injected_loader"
        elif isinstance(experiment, ExperimentInfo):
            self._loader = self._load_experiment_info(experiment)
            self._uid = experiment.uid
        else:
            self._loader = self._load_experiment(experiment)
            self._uid = "exp_from_json"
        self._data = self._loader.data()
        self._acquisition_type: AcquisitionType = self._loader.acquisition_type

        self.validate_experiment()

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
            pulse_defs=list(self._data.pulses.values()),
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

    def sections(self) -> list[str]:
        return list(self._data.sections.keys())

    def section_info(self, section_id: str) -> SectionInfo:
        retval = self._data.sections[section_id]
        return retval

    def root_sections(self) -> list[str]:
        return self._data.root_sections

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
    def direct_section_children(self, section_id) -> list[str]:
        return [child.uid for child in self.section_info(section_id).children]

    @cached_method()
    def all_section_children(self, section_id: str) -> set[str]:
        """Returns UID of all children of an section."""
        retval = set()
        for child in self.direct_section_children(section_id):
            retval.add(child)
            retval.update(self.all_section_children(child))
        return retval

    @cached_method()
    def section_parent(self, section_id) -> str | None:
        for parent_id in self.sections():
            parent = self.section_info(parent_id)
            if any(child.uid == section_id for child in parent.children):
                return parent.uid
        return None

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

    def dio_connections(self) -> list[tuple[str, str]]:
        return [
            (leader.uid, follower_uid)
            for leader in self.device_infos()
            for follower_uid in leader.followers
            if leader.device_type not in [DeviceInfoType.PQSC, DeviceInfoType.QHUB]
        ]

    def section_signals(self, section_id: str) -> set[str]:
        return {s.uid for s in self.section_info(section_id).signals}

    @cached_method()
    def section_signals_with_children(self, section_id: str) -> set[str]:
        """Returns UIDs of the signals in the section and its' children."""
        section = self.section_info(section_id)
        signals = self.section_signals(section.uid)
        for child in section.children:
            signals.update(self.section_signals_with_children(child.uid))
        return signals

    def pulses(self) -> list[str]:
        return list(self._data.pulses.keys())

    def pulse(self, pulse_id) -> PulseDef:
        return self._data.pulses.get(pulse_id)

    def oscillator_info(self, oscillator_id) -> OscillatorInfo:
        return self._data.oscillators[oscillator_id]

    def device_oscillators(self, device_id) -> list[str]:
        return list(self._data.device_oscillators.get(device_id, set()))

    def oscillators(self):
        return list(self._data.oscillators.keys())

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

    def section_pulses(self, section_id, signal_id) -> list[SectionSignalPulse]:
        return self._data.section_signal_pulses.get(section_id, {}).get(signal_id, [])

    def markers_on_signal(self, signal_id: str) -> set[str] | None:
        return self._data.signal_markers.get(signal_id)

    def triggers_on_signal(self, signal_id: str) -> int | None:
        return self._data.signal_trigger.get(signal_id)

    def section_parameters(self, section_id) -> list[ParameterInfo]:
        return self._data.section_parameters.get(section_id, [])

    def validate_experiment(self):
        validators.shfqa_unique_measure_pulse(self)
        validators.check_triggers_and_markers(self)
        validators.missing_sweep_parameter_for_play(self)
        validators.check_ppc_sweeper(self)
        validators.check_lo_frequency(self)
        validators.freq_sweep_on_acquire_line_requires_spectroscopy_mode(self)
        validators.check_phase_on_rf_signal_support(self)
        validators.check_phase_increments_support(self)
        validators.check_acquire_only_on_acquire_line(self)
        validators.check_no_play_on_acquire_line(self)
        validators.check_arbitrary_marker_is_valid(self)
        validators.check_no_sweeping_acquire_pulses(self)

    def acquisition_signal(self, handle: str) -> str | None:
        return self._data.handle_acquires[handle]
