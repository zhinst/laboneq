# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

import numpy as np

from laboneq.compiler import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import ExecutionType
from laboneq.core.types.enums.acquisition_type import is_spectroscopy
from laboneq.data.calibration import PortMode
from laboneq.data.compilation_job import SignalInfoType, DeviceInfoType, ParameterInfo

import logging


_logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from laboneq.compiler.experiment_access import ExperimentDAO


def shfqa_unique_measure_pulse(dao: ExperimentDAO):
    pulses_on_qa_out = {}
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                signal_info = dao.signal_info(signal_id=signal_id)
                if (
                    signal_info.type != SignalInfoType.INTEGRATION
                    and signal_info.device.device_type == DeviceInfoType.SHFQA
                ):
                    if section_pulse.pulse is not None:
                        pulses_on_qa_out.setdefault(signal_info.uid, set()).add(
                            section_pulse.pulse
                        )

    for signal, pulses in pulses_on_qa_out.items():
        if len(pulses) > 1:
            raise LabOneQException(
                f"Multiple different pulses are being played on signal {signal}. SHFQA"
                f" generators can only hold a single pulse waveform. Therefore, playing"
                f" multiple readout pulses represented by different Python objects is"
                f" not possible on a SHFQA measurement line."
            )


def check_triggers_and_markers(dao: ExperimentDAO):
    for section_id in dao.sections():
        section_info = dao.section_info(section_id=section_id)

        if len(section_info.triggers) > 0:
            for trigger in section_info.triggers:
                if trigger["signal_id"] not in dao.signals():
                    raise LabOneQException(
                        f"Trigger on section {section_id} played on signal"
                        f" {trigger['signal_id']} not present in experiment."
                        f" Available signal(s) are {', '.join(dao.signals())}."
                    )
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                if section_pulse.pulse is None:
                    continue
                pulse_id = section_pulse.pulse.uid
                device_type = DeviceType.from_device_info_type(
                    section_pulse.signal.device.device_type
                )
                if (
                    device_type == DeviceType.HDAWG
                    and len(section_pulse.signal.channels) == 1
                    and any(
                        "marker2" == m.marker_selector for m in section_pulse.markers
                    )
                ):
                    raise LabOneQException(
                        f"Single channel RF Pulse {pulse_id} referenced in section {section_id}"
                        f" has marker 2 enabled. Please only use marker 1 on RF channels."
                    )

                if device_type.is_qa_device and not len(section_pulse.markers) == 0:
                    raise LabOneQException(
                        f"Pulse {pulse_id} referenced in section {section_id}"
                        f" has markers but is to be played on a QA device. QA"
                        f" devices do not support markers."
                    )


def missing_sweep_parameter_for_play(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            signal_info = dao.signal_info(signal_id=signal_id)

            if signal_info.oscillator is not None and isinstance(
                signal_info.oscillator.frequency, ParameterInfo
            ):
                param_id = signal_info.oscillator.frequency.uid
                cur_section = dao.section_info(section_id)
                param_found = False
                while cur_section is not None:
                    if param_id in [s.uid for s in cur_section.parameters]:
                        param_found = True
                        break
                    parent_section_id = dao.section_parent(cur_section.uid)
                    cur_section = (
                        dao.section_info(parent_section_id)
                        if parent_section_id is not None
                        else None
                    )

                if not param_found:
                    raise LabOneQException(
                        f"Pulse {signal_id} referenced in section {section_id}"
                        f" is trying to use sweep parameter {param_id} that"
                        f" is not present in any parent sections"
                    )


def check_ppc_sweeper(dao: ExperimentDAO):
    PPC_SWEEPER_FIELDS = [
        "pump_power",
        "pump_frequency",
        "probe_power",
        "probe_frequency",
        "cancellation_phase",
        "cancellation_attenuation",
    ]

    def awg_from_shfqa_signal(signal: str) -> None | tuple[str, int]:
        signal_info = dao.signal_info(signal)
        if signal_info.device.device_type != DeviceInfoType.SHFQA:
            # only SHFQA can drive PPC sweeps; don't bother with other instruments
            return
        # for SHFQA, the `channels` attribute holds what we need
        [channel] = signal_info.channels
        device_id = signal_info.device.uid
        return (device_id, channel)

    realtime_sweep_params = set()
    for section_id in dao.sections():
        section_info = dao.section_info(section_id)
        if section_info.execution_type == ExecutionType.REAL_TIME:
            realtime_sweep_params.update(p.uid for p in section_info.parameters)

    awgs_with_ppc_sweeps: set[tuple[str, int]] = set()
    for signal in dao.signals():
        amplifier_pump = dao.amplifier_pump(signal)
        for field in PPC_SWEEPER_FIELDS:
            val = getattr(amplifier_pump, field, None)
            if isinstance(val, ParameterInfo) and val.uid in realtime_sweep_params:
                signal_info = dao.signal_info(signal)
                [channel] = signal_info.channels
                device_id = signal_info.device.uid
                awgs_with_ppc_sweeps.add((device_id, channel))

    awgs_with_section_trigger: set[tuple[str, int]] = set()
    for section_id in dao.sections():
        section_info = dao.section_info(section_id)
        for trigger in section_info.triggers:
            awg = awg_from_shfqa_signal(trigger["signal_id"])
            if awg is not None:
                awgs_with_section_trigger.add(awg)

    awgs_with_automute: set[tuple[str, int]] = set()
    for signal_id in dao.signals():
        signal_info = dao.signal_info(signal_id)
        if (
            signal_info.automute
            and (awg := awg_from_shfqa_signal(signal_id)) is not None
        ):
            awgs_with_automute.add(awg)

    if conflicts := awgs_with_ppc_sweeps.intersection(awgs_with_automute):
        msg = (
            "Signals on the following channels drive both SHFPPC sweeps, and use the"
            " output auto-muting feature:\n"
            + "\n".join(
                f" - device {device_id}, channel {channel}"
                for device_id, channel in conflicts
            )
        )
        raise LabOneQException(msg)

    if conflicts := awgs_with_ppc_sweeps.intersection(awgs_with_section_trigger):
        msg = (
            "Signals on the following channels drive both SHFPPC sweeps, and use"
            " section triggers:\n"
            + "\n".join(
                f" - device {device_id}, channel {channel}"
                for device_id, channel in conflicts
            )
        )
        raise LabOneQException(msg)


def check_lo_frequency(dao: ExperimentDAO):
    for signal in dao.signals():
        signal_info = dao.signal_info(signal)

        if signal_info.device.device_type not in [
            DeviceInfoType.SHFQA,
            DeviceInfoType.SHFSG,
        ]:
            continue
        if signal_info.lo_frequency is None or signal_info.port_mode == PortMode.LF:
            continue

        if isinstance(signal_info.lo_frequency, ParameterInfo):
            values = signal_info.lo_frequency.values
        else:
            values = [signal_info.lo_frequency]

        for f in values:
            if abs(f % 200e6) > 1e-6:
                raise LabOneQException(
                    f"Cannot set local oscillator of signal {signal} to {f / 1e9:.3} GHz."
                    f" Only integer multiples of 200 MHz are accepted."
                )


def freq_sweep_on_acquire_line_requires_spectroscopy_mode(dao: ExperimentDAO):
    for signal in dao.signals():
        signal_info = dao.signal_info(signal)
        if (
            signal_info.device.device_type
            in [DeviceInfoType.SHFQA, DeviceInfoType.UHFQA]
            and signal_info.oscillator is not None
            and isinstance(signal_info.oscillator.frequency, ParameterInfo)
            and signal_info.oscillator.is_hardware
            and not is_spectroscopy(dao.acquisition_type)
        ):
            raise LabOneQException(
                f"Hardware oscillator sweep using oscillator {signal_info.oscillator.uid} on acquire line "
                f"{signal_info.uid} connected to UFHQA or SHFQQA device {signal_info.device.uid} "
                f"requires acquisition type to be set to spectroscopy"
            )


def check_phase_on_rf_signal_support(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            signal = dao.signal_info(signal_id)
            device = signal.device
            if device.device_type == DeviceInfoType.PRETTYPRINTERDEVICE:
                continue

            if (
                signal.oscillator is not None and not signal.oscillator.is_hardware
            ) or signal.type != SignalInfoType.RF:
                continue

            for ssp in dao.section_pulses(section_id, signal_id):
                if ssp.phase is not None or (
                    ssp.amplitude is not None
                    and (
                        isinstance(ssp.amplitude, ParameterInfo)
                        and not np.all(np.isreal(ssp.amplitude.values))
                    )
                    or not np.isreal(ssp.amplitude)
                ):
                    raise LabOneQException(
                        f"In section '{section_id}', signal '{signal_id}':"
                        " baseband phase modulation not possible for RF signal with HW oscillator"
                    )


def check_phase_increments_support(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            signal = dao.signal_info(signal_id)
            device = signal.device
            if device.device_type == DeviceInfoType.PRETTYPRINTERDEVICE:
                continue

            if (
                signal.oscillator is not None and not signal.oscillator.is_hardware
            ) or signal.type == SignalInfoType.IQ:
                continue

            for ssp in dao.section_pulses(section_id, signal_id):
                if ssp.increment_oscillator_phase is not None:
                    raise LabOneQException(
                        f"In section '{section_id}', signal '{signal_id}':"
                        " phase increments are only supported on IQ signals, or on RF signals with SW modulation"
                    )


def check_acquire_only_on_acquire_line(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                is_acquire = section_pulse.acquire_params is not None
                if (
                    is_acquire
                    and dao.signal_info(signal_id).type != SignalInfoType.INTEGRATION
                ):
                    raise LabOneQException(
                        f"In section '{section_id}, an acquire statement is issued on signal '{signal_id}'."
                        " acquire is only allowed on acquire lines."
                    )


def check_no_play_on_acquire_line(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                is_acquire = section_pulse.acquire_params is not None
                if (
                    not is_acquire
                    and dao.signal_info(signal_id).type == SignalInfoType.INTEGRATION
                    and section_pulse.pulse is not None
                ):
                    raise LabOneQException(
                        f"In section '{section_id}, a play statement is issued on signal '{signal_id}'."
                        " play is not allowed on acquire lines."
                    )


def check_arbirary_marker_is_valid(dao: ExperimentDAO):
    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                for marker in section_pulse.markers:
                    if marker.pulse_id is not None:
                        marker_pulse = dao.pulse(marker.pulse_id)
                        if (
                            marker_pulse.function is not None
                            and marker_pulse.function != "const"
                        ):
                            raise LabOneQException(
                                f"A pulse in section {section_id} attempts to play an arbitrary marker with a pulse functional other than `const'."
                                " At this time, only constants pulses or sampled pulses are supported"
                            )
                        if marker_pulse.samples is not None and not all(
                            [s == 0.0 or s == 1.0 for s in marker_pulse.samples]
                        ):
                            raise LabOneQException(
                                f"A pulse in section {section_id} attempts to play a sampled arbitrary marker with a sample not set to either 0 or 1."
                                " Please make sure that all samples of your markers are either 0 or 1."
                            )


def check_no_sweeping_acquire_pulses(dao: ExperimentDAO):
    from laboneq.dsl.parameter import Parameter

    for section_id in dao.sections():
        for signal_id in dao.section_signals(section_id):
            for section_pulse in dao.section_pulses(section_id, signal_id):
                is_acquire = section_pulse.acquire_params is not None
                if is_acquire and isinstance(section_pulse.length, Parameter):
                    raise LabOneQException(
                        f"In section '{section_id}', the length of an acquire statement is being swept with parameter {section_pulse.length}."
                        " Sweeping of acquire statements is not supported."
                    )
