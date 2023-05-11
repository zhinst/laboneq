# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import List, Optional

from jsonschema import ValidationError

from laboneq._utils import cached_method
from laboneq.compiler.experiment_access import json_dumper
from laboneq.compiler.experiment_access.device_info import DeviceInfo
from laboneq.compiler.experiment_access.dsl_loader import DSLLoader
from laboneq.compiler.experiment_access.json_loader import JsonLoader
from laboneq.compiler.experiment_access.oscillator_info import OscillatorInfo
from laboneq.compiler.experiment_access.section_info import SectionInfo
from laboneq.compiler.experiment_access.signal_info import SignalInfo
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.core.validators import dicts_equal

_logger = logging.getLogger(__name__)


class ExperimentDAO:
    def __init__(self, experiment, core_device_setup=None, core_experiment=None):
        self._data = {}
        self._acquisition_type: AcquisitionType = None  # type: ignore
        if core_device_setup is not None and core_experiment is not None:
            self._load_from_core(core_device_setup, core_experiment)
        else:
            self._load_experiment(experiment)
        self.validate_experiment()

    def __eq__(self, other):
        if not isinstance(other, ExperimentDAO):
            return False

        return self._acquisition_type == other._acquisition_type and dicts_equal(
            self._data, other._data
        )

    def add_signal(
        self, device_id, channels, connection_type, signal_id, signal_type, modulation
    ):
        self._data["signals"][signal_id] = {
            "signal_id": signal_id,
            "signal_type": signal_type,
            "modulation": modulation,
            "offset": None,
        }

        self._data["signal_connections"][signal_id] = {
            "signal_id": signal_id,
            "device_id": device_id,
            "connection_type": connection_type,
            "channels": channels,
            "voltage_offset": None,
            "mixer_calibration": None,
            "precompensation": None,
            "lo_frequency": None,
            "range": None,
            "range_unit": None,
            "port_delay": None,
            "delay_signal": None,
            "port_mode": None,
            "threshold": None,
            "amplifier_pump": None,
        }

    def _load_experiment(self, experiment):
        loader = JsonLoader()
        try:
            validator = loader.schema_validator()
            validator.validate(experiment)
        except ValidationError as exception:
            _logger.warning("Failed to validate input:")
            for line in str(exception).splitlines():
                _logger.warning("validation error: %s", line)
        loader.load(experiment)
        self._data = loader.data()
        self._acquisition_type = loader.acquisition_type

    def _load_from_core(self, device_setup, experiment):
        loader = DSLLoader()
        loader.load(experiment, device_setup)
        self._data = loader.data()
        self._acquisition_type = loader.acquisition_type

    @staticmethod
    def dump(experiment_dao: "ExperimentDAO"):
        return json_dumper.dump(experiment_dao)

    @property
    def acquisition_type(self) -> AcquisitionType:
        return self._acquisition_type

    def server_infos(self):
        return copy.deepcopy(list(self._data["servers"].values()))

    def signals(self):
        return sorted([s["signal_id"] for s in self._data["signals"].values()])

    def devices(self) -> List[str]:
        return [d["id"] for d in self._data["devices"].values()]

    def global_leader_device(self) -> str:
        try:
            return next(
                d for d in self._data["devices"].values() if d.get("is_global_leader")
            )["id"]
        except StopIteration:
            return None

    @classmethod
    def _device_info_keys(cls):
        return [
            "id",
            "device_type",
            "serial",
            "server",
            "interface",
            "reference_clock_source",
            "is_qc",
        ]

    def device_info(self, device_id) -> Optional[DeviceInfo]:
        device_info = self._data["devices"].get(device_id)
        if device_info is not None:
            return DeviceInfo(**{k: device_info[k] for k in self._device_info_keys()})
        return None

    def device_infos(self) -> List[DeviceInfo]:
        return [
            DeviceInfo(**{k: device_info[k] for k in self._device_info_keys()})
            for device_info in self._data["devices"].values()
        ]

    def device_reference_clock(self, device_id):
        return self._data["devices"][device_id].get("reference_clock")

    def device_from_signal(self, signal_id):
        signal_connection = self._data["signal_connections"][signal_id]
        return signal_connection["device_id"]

    def devices_in_section_no_descend(self, section_id):
        signals = self._data["section_signals"].get(section_id, [])
        devices = {self.device_from_signal(s) for s in signals}
        return devices

    def _device_types_in_section_no_descend(self, section_id):
        devices = self.devices_in_section_no_descend(section_id)
        return {
            d["device_type"]
            for d in self._data["devices"].values()
            if d["id"] in devices
        }

    def device_types_in_section(self, section_id):
        if self.is_branch(section_id):
            return self.device_types_in_section(self.section_parent(section_id))
        retval = set()
        section_with_children = self.all_section_children(section_id)
        section_with_children.add(section_id)
        for child in section_with_children:
            retval = retval.union(self._device_types_in_section_no_descend(child))
        return retval

    @classmethod
    def _signal_info_keys(cls):
        return [
            "signal_id",
            "signal_type",
            "device_id",
            "device_serial",
            "device_type",
            "connection_type",
            "channels",
            "delay_signal",
            "modulation",
            "offset",
        ]

    @cached_method()
    def signal_info(self, signal_id):
        signal_info = self._data["signals"].get(signal_id)
        if signal_info is not None:
            signal_info_copy = copy.deepcopy(signal_info)
            signal_connection = self._data["signal_connections"][signal_id]

            for k in ["device_id", "connection_type", "channels", "delay_signal"]:
                signal_info_copy[k] = signal_connection[k]

            device_info = self._data["devices"][signal_connection["device_id"]]

            signal_info_copy["device_type"] = device_info["device_type"]
            signal_info_copy["device_serial"] = device_info["serial"]
            return SignalInfo(
                **{k: signal_info_copy[k] for k in self._signal_info_keys()}
            )
        else:
            raise Exception(f"Signal_id {signal_id} not found")

    def sections(self) -> List[str]:
        return list(self._data["sections"].keys())

    def section_info(self, section_id) -> SectionInfo:
        retval = self._data["sections"][section_id]

        if retval.count is not None:
            retval.count = int(retval.count)
        return retval

    def root_sections(self):
        return self._data["root_sections"]

    @cached_method()
    def _has_near_time_child(self, section_id) -> Optional[str]:
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
            if info.execution_type != "controller":
                if info.execution_type == "hardware" and nt_subsection is not None:
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
        return self._data["section_tree"].get(section_id, [])

    @cached_method()
    def all_section_children(self, section_id):
        retval = []

        direct_children = self.direct_section_children(section_id)
        retval = retval + direct_children
        for child in direct_children:
            retval = retval + list(self.all_section_children(child))

        return set(retval)

    @cached_method()
    def section_parent(self, section_id):
        try:
            return next(
                parent
                for parent, children in self._data["section_tree"].items()
                if section_id in children
            )
        except StopIteration:
            return None

    def is_branch(self, section_id):
        return self._data["sections"][section_id].state is not None

    def pqscs(self):
        return [p[0] for p in self._data["pqsc_ports"]]

    def pqsc_ports(self, pqsc_device_id):
        return [
            {"device": p[1], "port": p[2]}
            for p in self._data["pqsc_ports"]
            if p[0] == pqsc_device_id
        ]

    def dio_followers(self):
        return [d[1] for d in self._data["dios"]]

    def dio_leader(self, device_id):
        try:
            return next(d[0] for d in self._data["dios"] if d[1] == device_id)
        except StopIteration:
            return None

    def dio_connections(self):
        return [(dio[0], dio[1]) for dio in self._data["dios"]]

    def is_dio_leader(self, device_id):
        return bool({d[1] for d in self._data["dios"] if d[0] == device_id})

    def section_signals(self, section_id):
        return self._data["section_signals"].get(section_id, set())

    @cached_method()
    def section_signals_with_children(self, section_id):
        retval = set()
        section_with_children = self.all_section_children(section_id)
        section_with_children.add(section_id)
        for child in section_with_children:
            retval = retval.union(self.section_signals(child))
        return retval

    def pulses(self):
        return list(self._data["pulses"].keys())

    def pulse(self, pulse_id):
        pulse = self._data["pulses"].get(pulse_id)
        return pulse

    @classmethod
    def _oscillator_info_fields(cls):
        return ["id", "frequency", "frequency_param", "hardware"]

    def oscillator_info(self, oscillator_id) -> OscillatorInfo:
        oscillator = self._data["oscillators"].get(oscillator_id)
        if oscillator is None:
            return None
        return OscillatorInfo(
            **{k: oscillator[k] for k in self._oscillator_info_fields()}
        )

    def hardware_oscillators(self) -> List[OscillatorInfo]:
        oscillator_infos: List[OscillatorInfo] = []
        for device in self.devices():
            device_oscillators = self.device_oscillators(device)
            for oscillator_id in device_oscillators:
                info = self.oscillator_info(oscillator_id)
                if info is not None and info.hardware:
                    info.device_id = device
                    oscillator_infos.append(info)

        return list(sorted(oscillator_infos, key=lambda x: (x.device_id, x.id)))

    def device_oscillators(self, device_id):
        return [
            do["oscillator_id"]
            for do in self._data["device_oscillators"].get(device_id, [])
        ]

    def oscillators(self):
        return list(self._data["oscillators"].keys())

    def signal_oscillator(self, signal_id):
        oscillator = self._data["signal_oscillator"].get(signal_id)
        if oscillator is None:
            return None
        return self.oscillator_info(oscillator)

    def voltage_offset(self, signal_id):
        return self._data["signal_connections"][signal_id]["voltage_offset"]

    def mixer_calibration(self, signal_id):
        return self._data["signal_connections"][signal_id]["mixer_calibration"]

    def precompensation(self, signal_id):
        return self._data["signal_connections"][signal_id]["precompensation"]

    def lo_frequency(self, signal_id):
        return self._data["signal_connections"][signal_id]["lo_frequency"]

    def signal_range(self, signal_id):
        sc = self._data["signal_connections"][signal_id]
        return sc["range"], sc["range_unit"]

    def port_delay(self, signal_id):
        return self._data["signal_connections"][signal_id]["port_delay"]

    def port_mode(self, signal_id):
        return self._data["signal_connections"][signal_id]["port_mode"]

    def threshold(self, signal_id):
        return self._data["signal_connections"][signal_id]["threshold"]

    def amplifier_pump(self, signal_id):
        return self._data["signal_connections"][signal_id]["amplifier_pump"]

    def section_pulses(self, section_id, signal_id):
        retval = self._section_pulses_raw(section_id, signal_id)
        for sp in retval:
            pulse_id = sp.pulse_id
            if pulse_id is not None:
                pulse_def = self._data["pulses"].get(pulse_id)
                if pulse_def is not None:
                    if sp.length is None and sp.length_param is None:
                        sp.length = pulse_def.length
                        # TODO(2K): pulse_def has no length_param!
                        # if pulse_def.length_param is not None:
                        #     sp["length_param"] = pulse_def.length_param
                        #     sp["length"] = None

        return retval

    def _section_pulses_raw(self, section_id, signal_id):
        section_signal_pulses = (
            self._data["section_signal_pulses"].get(section_id, {}).get(signal_id, [])
        )
        return [copy.copy(ssp) for ssp in section_signal_pulses]

    def markers_on_signal(self, signal_id: str):
        return self._data["signal_markers"].get(signal_id)

    def triggers_on_signal(self, signal_id: str):
        return self._data["signal_trigger"].get(signal_id)

    def section_parameters(self, section_id):
        return [
            {k: p.get(k) for k in ["id", "start", "step", "values", "axis_name"]}
            for p in self._data["section_parameters"].get(section_id, [])
        ]

    def validate_experiment(self):
        all_parameters = set()
        for section_id in self.sections():
            for parameter in self.section_parameters(section_id):
                all_parameters.add(parameter["id"])

        for section_id in self.sections():
            for signal_id in self.section_signals(section_id):
                for section_pulse in self.section_pulses(section_id, signal_id):
                    pulse_id = section_pulse.pulse_id
                    if pulse_id is not None and pulse_id not in self._data["pulses"]:
                        raise RuntimeError(
                            f"Pulse {pulse_id} referenced in section {section_id} by a pulse on signal {signal_id} is not known."
                        )

                    for k in ["length_param", "amplitude_param", "offset_param"]:
                        param_name = getattr(section_pulse, k)
                        if param_name is not None:
                            if param_name not in all_parameters:
                                raise RuntimeError(
                                    f"Parameter {param_name} referenced in section {section_id} by a pulse on signal {signal_id} is not known."
                                )

    def acquisition_signal(self, handle: str) -> Optional[str]:
        return self._data["handle_acquires"][handle]
