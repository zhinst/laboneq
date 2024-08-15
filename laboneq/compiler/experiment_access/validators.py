# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from laboneq.compiler import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import ExecutionType, AcquisitionType
from laboneq.data.compilation_job import SignalInfoType, DeviceInfoType, ParameterInfo

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

    if awgs_with_ppc_sweeps and dao.acquisition_type in [
        AcquisitionType.SPECTROSCOPY_IQ,
        AcquisitionType.SPECTROSCOPY_PSD,
    ]:
        raise LabOneQException("SHFPPC not supported in SPECTROSCOPY mode")

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
