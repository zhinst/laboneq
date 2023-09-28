# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Optional

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.data.compilation_job import (
    DeviceInfo,
    DeviceInfoType,
    OscillatorInfo,
    ParameterInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
)

logger = logging.getLogger(__name__)


class LoaderBase:
    def __init__(self):
        self.acquisition_type: Optional[AcquisitionType] = None
        self.global_leader_device_id: str | None = None

        self._devices: dict[str, DeviceInfo] = {}
        self._device_oscillators = {}
        self._oscillators: dict[str, OscillatorInfo] = {}
        self._pulses: dict[str, PulseDef] = {}
        self._sections: dict[str, SectionInfo] = {}

        # Todo (PW): This could be dropped and replaced by a look-up of
        #  `SectionInfo.parameters`. The loaders will require updating though.
        self._section_parameters = {}

        # Todo (PW): Unlike `SectionInfo.pulses`, `self._section_signal_pulses`
        #  is indexed by signal. We could drop `_section_signal_pulses` and instead
        #  do a linear search of the section's SSPs instead.
        #  The scheduler could indeed be refactored such that it does not need access by
        #  signal, so there is no performance down-side.
        self._section_signal_pulses: dict[str, dict[str, SectionSignalPulse]] = {}

        self._signals: dict[str, SignalInfo] = {}
        self._signal_markers = {}
        self._signal_trigger = {}
        self._root_sections = []
        self._handle_acquires: dict[str, str] = {}

        self._all_parameters: dict[str, ParameterInfo] = {}

    def data(self) -> dict[str, Any]:
        return {
            "devices": self._devices,
            "device_oscillators": self._device_oscillators,
            "oscillators": self._oscillators,
            "pulses": self._pulses,
            "root_sections": self._root_sections,
            "sections": self._sections,
            "section_parameters": self._section_parameters,
            "section_signal_pulses": self._section_signal_pulses,
            "signals": self._signals,
            "signal_markers": self._signal_markers,
            "signal_trigger": self._signal_trigger,
            "handle_acquires": self._handle_acquires,
            "global_leader_device_id": self.global_leader_device_id,
        }

    def add_device_oscillator(self, device_id, oscillator_id):
        o = {
            "device_id": device_id,
            "oscillator_id": oscillator_id,
        }
        if device_id not in self._device_oscillators:
            self._device_oscillators[device_id] = []
        d = self._device_oscillators[device_id]
        if o not in d:
            d.append(o)

    def _get_or_create_parameter(self, parameter_id) -> ParameterInfo:
        if (parameter := self._all_parameters.get(parameter_id)) is not None:
            return parameter
        param = self._all_parameters[parameter_id] = ParameterInfo(uid=parameter_id)
        return param

    def add_section_parameter(
        self,
        section_id,
        parameter_id,
        start=None,
        step=None,
        values_list=None,
        axis_name=None,
    ):
        param = self._get_or_create_parameter(parameter_id)
        param.start = start
        param.step = step
        param.values = values_list
        param.axis_name = axis_name

        self._section_parameters.setdefault(section_id, []).append(param)

    def add_section_signal(self, section_uid, signal_uid):
        assert section_uid in self._sections, "use `add_section()` first"
        section = self._sections[section_uid]

        assert signal_uid in self._signals, "use `add_signal()` first"
        signal = self._signals[signal_uid]

        if signal not in section.signals:
            section.signals.append(signal)

    def add_section_signal_pulse(
        self, section_id, signal_id: str, section_signal_pulse: SectionSignalPulse
    ):
        self._section_signal_pulses.setdefault(section_id, {}).setdefault(
            signal_id, []
        ).append(section_signal_pulse)
        if section_signal_pulse.acquire_params is not None:
            handle = section_signal_pulse.acquire_params.handle
            if handle is not None:
                self.add_handle_acquire(handle, signal_id)

    def add_signal_marker(self, signal_id, marker: str):
        self._signal_markers.setdefault(signal_id, set()).add(marker)

    def add_device(
        self,
        device_id: str,
        device_type: DeviceInfoType | str,
        reference_clock=None,
        reference_clock_source=None,
        is_qc=None,
    ):
        self._devices[device_id] = DeviceInfo(
            uid=device_id,
            device_type=DeviceInfoType(device_type),
            reference_clock=reference_clock,
            reference_clock_source=reference_clock_source,
            is_qc=is_qc,
        )

    def add_oscillator(self, oscillator_id, frequency, is_hardware):
        self._oscillators[oscillator_id] = OscillatorInfo(
            uid=oscillator_id, frequency=frequency, is_hardware=is_hardware
        )

    def add_signal(self, signal_id, signal_type: str | SignalInfoType):
        signal_info = SignalInfo(uid=signal_id, type=SignalInfoType(signal_type))
        assert signal_id not in self._signals
        self._signals[signal_id] = signal_info

    def add_signal_oscillator(self, signal_id, oscillator_id):
        signal_info: SignalInfo = self._signals[signal_id]
        signal_info.oscillator = self._oscillators[oscillator_id]

    def add_signal_connection(self, signal_id, signal_connection):
        signal_info: SignalInfo = self._signals[signal_id]
        signal_info.device = self._devices[signal_connection["device_id"]]
        signal_info.channels = signal_connection["channels"]
        signal_info.voltage_offset = signal_connection["voltage_offset"]
        signal_info.mixer_calibration = signal_connection["mixer_calibration"]
        signal_info.precompensation = signal_connection["precompensation"]
        signal_info.lo_frequency = signal_connection["lo_frequency"]
        signal_info.signal_range = signal_connection["range"]
        signal_info.port_delay = signal_connection["port_delay"]
        signal_info.delay_signal = signal_connection["delay_signal"]
        signal_info.port_mode = signal_connection["port_mode"]
        signal_info.threshold = signal_connection["threshold"]
        signal_info.amplitude = signal_connection["amplitude"]
        signal_info.amplifier_pump = signal_connection["amplifier_pump"]

    def add_section(self, section_id, section_info: SectionInfo):
        if section_info.handle is not None and section_info.user_register is not None:
            raise LabOneQException(
                f"Section {section_id} has both a handle and a user register set."
            )
        self._sections[section_id] = section_info

    def add_pulse(self, pulse_id, pulse_def: PulseDef):
        self._pulses[pulse_id] = pulse_def

    def add_section_child(self, parent_id, child_id):
        assert parent_id in self._sections, "use `add_section()` first"
        assert child_id in self._sections, "use `add_section()` first"

        parent: SectionInfo = self._sections[parent_id]
        parent.children.append(self._sections[child_id])

    def add_handle_acquire(self, handle: str, signal: str):
        if handle in self._handle_acquires:
            other_signal = self._handle_acquires[handle]
            if other_signal != signal:
                raise LabOneQException(
                    f"Acquisition handle '{handle}' used on multiple signals: {other_signal}, {signal}"
                )
        self._handle_acquires[handle] = signal
