# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for building Rust experiment representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import laboneq._rust.compiler as compiler_rs
from laboneq.data.compilation_job import ParameterInfo

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
    from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
    from laboneq.data.compilation_job import DeviceInfo
    from laboneq.data.parameter import Parameter


def _resolve_parents(parameter: Parameter) -> set[str]:
    # NOTE: Legacy serializer fails if imported at the top level
    from laboneq.data.parameter import SweepParameter

    parents = set()

    def _traverse(param: Parameter):
        if not isinstance(param, SweepParameter):
            return
        for driver in param.driven_by:
            if driver.uid not in parents:
                parents.add(driver.uid)
                _traverse(driver)

    _traverse(parameter)
    return parents


def _resolve_all_driving_parameters(
    parameters: list[Parameter],
) -> dict[str, list[str]]:
    """Resolve all driving parameters for each sweep parameter."""
    parents: dict[str, set[str]] = {}
    for param in parameters:
        parents[param.uid] = _resolve_parents(param)
    return {k: list(v) for k, v in parents.items()}


def build_rs_experiment(
    experiment_dao: ExperimentDAO,
    sampling_rate_tracker: SamplingRateTracker,
    signal_objects: dict[str, SignalObj],
) -> compiler_rs.ExperimentInfo:
    """Builds a Rust representation of the experiment."""
    driving_parameters = _resolve_all_driving_parameters(experiment_dao.dsl_parameters)

    def maybe_parameter(value: Any) -> compiler_rs.SweepParameter | Any:
        if isinstance(value, ParameterInfo):
            return compiler_rs.SweepParameter(
                uid=value.uid,
                values=value.values,
                driven_by=driving_parameters.get(value.uid, []),
            )
        return value

    def create_device(device_info: DeviceInfo) -> compiler_rs.Device:
        return compiler_rs.Device(
            uid=device_info.uid,
            physical_device_uid=device_info.physical_device_uid,
            kind=device_info.device_type.name,
            is_shfqc=device_info.is_qc,
        )

    devices = {}
    signals = []
    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
        device_uid = signal_info.device.uid
        if device_uid not in devices:
            device = experiment_dao.device_info(device_uid)
            devices[device_uid] = create_device(device)

        osc = None
        if signal_info.oscillator is not None:
            osc = compiler_rs.Oscillator(
                uid=signal_info.oscillator.uid,
                frequency=maybe_parameter(signal_info.oscillator.frequency),
                is_hardware=signal_info.oscillator.is_hardware is True,
            )
        signal_obj = signal_objects[signal]
        s = compiler_rs.Signal(
            uid=signal,
            sampling_rate=sampling_rate_tracker.sampling_rate_for_device(
                device_uid, signal_info.type
            ),
            awg_key=hash(signal_obj.awg.key),
            device_uid=device_uid,
            oscillator=osc,
            lo_frequency=maybe_parameter(signal_info.lo_frequency),
            voltage_offset=maybe_parameter(signal_info.voltage_offset),
            amplifier_pump=compiler_rs.AmplifierPump(
                device=signal_info.amplifier_pump.ppc_device.uid,
                channel=signal_info.amplifier_pump.channel,
                pump_frequency=maybe_parameter(
                    signal_info.amplifier_pump.pump_frequency
                ),
                pump_power=maybe_parameter(signal_info.amplifier_pump.pump_power),
                cancellation_phase=maybe_parameter(
                    signal_info.amplifier_pump.cancellation_phase
                ),
                cancellation_attenuation=maybe_parameter(
                    signal_info.amplifier_pump.cancellation_attenuation
                ),
                probe_frequency=maybe_parameter(
                    signal_info.amplifier_pump.probe_frequency
                ),
                probe_power=maybe_parameter(signal_info.amplifier_pump.probe_power),
            )
            if signal_info.amplifier_pump is not None
            else None,
            kind=signal_info.type.name,
            channels=signal_info.channels,
            automute=signal_info.automute,
            port_mode=signal_info.port_mode.value
            if signal_info.port_mode is not None
            else None,
            signal_delay=signal_obj.delay_signal or 0.0,
            port_delay=maybe_parameter(signal_obj.port_delay),
            start_delay=signal_obj.start_delay or 0.0,
        )
        signals.append(s)
    return compiler_rs.build_experiment(
        experiment=experiment_dao.source_experiment,
        signals=signals,
        devices=list(devices.values()),
    )
