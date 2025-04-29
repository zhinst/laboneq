# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import logging

from laboneq.data.calibration import PortMode
import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.data.compilation_job import (
    AmplifierPumpInfo,
    DeviceInfo,
    DeviceInfoType,
    MixerCalibrationInfo,
    OscillatorInfo,
    ParameterInfo,
    PrecompensationInfo,
    PulseDef,
    ReferenceClockSourceInfo,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
    SignalRange,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentData:
    devices: dict[str, DeviceInfo]
    device_oscillators: dict[str, set[str]]
    oscillators: dict[str, OscillatorInfo]
    pulses: dict[str, PulseDef]
    root_sections: list[str]
    sections: dict[str, SectionInfo]
    section_parameters: dict[str, list[ParameterInfo]]
    section_signal_pulses: dict[str, dict[str, list[SectionSignalPulse]]]
    signals: dict[str, SignalInfo]
    signal_markers: dict[str, set[str]]
    signal_trigger: dict[str, int]
    handle_acquires: dict[str, str]
    global_leader_device_id: str | None


class LoaderBase:
    def __init__(self):
        self.acquisition_type: AcquisitionType | None = None
        self.global_leader_device_id: str | None = None

        self._devices: dict[str, DeviceInfo] = {}
        self._device_oscillators: dict[str, set[str]] = {}
        self._oscillators: dict[str, OscillatorInfo] = {}
        self._pulses: dict[str, PulseDef] = {}
        self._sections: dict[str, SectionInfo] = {}

        # Todo (PW): This could be dropped and replaced by a look-up of
        #  `SectionInfo.parameters`. The loaders will require updating though.
        self._section_parameters: dict[str, list[ParameterInfo]] = {}

        # Todo (PW): Unlike `SectionInfo.pulses`, `self._section_signal_pulses`
        #  is indexed by signal. We could drop `_section_signal_pulses` and instead
        #  do a linear search of the section's SSPs instead.
        #  The scheduler could indeed be refactored such that it does not need access by
        #  signal, so there is no performance down-side.
        self._section_signal_pulses: dict[str, dict[str, list[SectionSignalPulse]]] = {}

        self._signals: dict[str, SignalInfo] = {}
        self._signal_markers: dict[str, set[str]] = {}
        self._signal_trigger: dict[str, int] = {}
        self._root_sections: list[str] = []
        self._handle_acquires: dict[str, str] = {}

        self._all_parameters: dict[str, ParameterInfo] = {}

    def data(self) -> ExperimentData:
        return ExperimentData(
            devices=self._devices,
            device_oscillators=self._device_oscillators,
            oscillators=self._oscillators,
            pulses=self._pulses,
            root_sections=self._root_sections,
            sections=self._sections,
            section_parameters=self._section_parameters,
            section_signal_pulses=self._section_signal_pulses,
            signals=self._signals,
            signal_markers=self._signal_markers,
            signal_trigger=self._signal_trigger,
            handle_acquires=self._handle_acquires,
            global_leader_device_id=self.global_leader_device_id,
        )

    def add_device_oscillator(self, device_id: str, oscillator_id: str):
        if device_id not in self._device_oscillators:
            self._device_oscillators[device_id] = set()
        self._device_oscillators[device_id].add(oscillator_id)

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
        count: int | None = None,
        values_list=None,
        axis_name=None,
    ):
        param = self._get_or_create_parameter(parameter_id)
        param.start = start
        param.step = step
        if values_list is not None:
            param.values = values_list
        elif start is not None and count is not None and step is not None:
            param.values = list(np.arange(count) * step + start)
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

    def add_signal_marker(self, signal_id: str, marker: str):
        self._signal_markers.setdefault(signal_id, set()).add(marker)

    def add_device(
        self,
        device_id: str,
        device_type: DeviceInfoType | str,
        reference_clock_source: str | None = None,
        is_qc=None,
    ):
        def _ref_clk_from_str(ref_clk: str | None) -> ReferenceClockSourceInfo | None:
            if ref_clk == "internal":
                return ReferenceClockSourceInfo.INTERNAL
            if ref_clk == "external":
                return ReferenceClockSourceInfo.EXTERNAL
            return None

        self._devices[device_id] = DeviceInfo(
            uid=device_id,
            device_type=DeviceInfoType(device_type),
            reference_clock_source=_ref_clk_from_str(reference_clock_source),
            is_qc=is_qc,
        )

    def add_oscillator(self, oscillator_id: str, frequency, is_hardware):
        self._oscillators[oscillator_id] = OscillatorInfo(
            uid=oscillator_id, frequency=frequency, is_hardware=is_hardware
        )

    def add_signal(self, signal_id, signal_type: str | SignalInfoType):
        signal_info = SignalInfo(uid=signal_id, type=SignalInfoType(signal_type))
        assert signal_id not in self._signals
        self._signals[signal_id] = signal_info

    def add_signal_oscillator(self, signal_id, oscillator_id):
        signal_info = self._signals[signal_id]
        signal_info.oscillator = self._oscillators[oscillator_id]

    def add_signal_connection(
        self,
        signal_id: str,
        device_id: str,
        channels: list[int],
        voltage_offset: float | ParameterInfo | None,
        mixer_calibration: MixerCalibrationInfo | None,
        precompensation: PrecompensationInfo | None,
        lo_frequency: float | ParameterInfo | None,
        signal_range: SignalRange | None,
        port_delay: float | ParameterInfo | None,
        delay_signal: float | ParameterInfo | None,
        port_mode: PortMode | None,
        threshold: float | list[float] | None,
        amplitude: float | ParameterInfo | None,
        amplifier_pump: AmplifierPumpInfo | None,
    ):
        signal_info = self._signals[signal_id]
        signal_info.device = self._devices[device_id]
        signal_info.channels = channels
        signal_info.voltage_offset = voltage_offset
        signal_info.mixer_calibration = mixer_calibration
        signal_info.precompensation = precompensation
        signal_info.lo_frequency = lo_frequency
        signal_info.signal_range = signal_range
        signal_info.port_delay = port_delay
        signal_info.delay_signal = delay_signal
        signal_info.port_mode = port_mode
        signal_info.threshold = threshold
        signal_info.amplitude = amplitude
        signal_info.amplifier_pump = amplifier_pump

    def add_section(self, section_id, section_info: SectionInfo):
        if (
            section_info.match_handle is not None
            and section_info.match_user_register is not None
        ):
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
