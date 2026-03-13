# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for building Rust experiment representation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from laboneq.data.compilation_job import DeviceInfoType, ParameterInfo, SignalInfo
from laboneq.implementation.utils.devices import parse_device_options

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
    from laboneq.data.compilation_job import DeviceInfo
    from laboneq.dsl.parameter import Parameter


def _resolve_parents(parameter: Parameter) -> set[str]:
    # NOTE: Legacy serializer fails if imported at the top level
    from laboneq.data.parameter import SweepParameter as DataSweepParameter
    from laboneq.dsl.parameter import SweepParameter

    parents = set()

    def _traverse(param: Parameter):
        # TODO(DSL cutover): Remove DataSweepParameter once setup calibration uses DSL types.
        if not isinstance(param, (SweepParameter, DataSweepParameter)):
            return
        for driver in param.driven_by or []:
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


def _resolve_default_options(
    options: str, device_type: DeviceInfoType, is_qc: bool
) -> list[str]:
    dev_type, options = parse_device_options(options)
    options = options or []
    # TODO(2K): This is a workaround, as options string is still not
    # enforced in the device setup.
    if dev_type is not None:
        return [dev_type, *options]
    if device_type == DeviceInfoType.UHFQA:
        dev_type = "UHFQA"
    elif device_type == DeviceInfoType.HDAWG:
        dev_type = "HDAWG8"
    elif is_qc:
        options = ["QC6CH"] if not options else options
        return ["SHFQC", *options]
    elif device_type == DeviceInfoType.SHFQA:
        dev_type = "SHFQA2"
    elif device_type == DeviceInfoType.SHFSG:
        dev_type = "SHFSG8"
    else:
        return []
    # TODO(2K): Add warning for missing options in the device setup
    out = [dev_type, *options]
    return out


def build_rs_experiment(
    experiment_dao: ExperimentDAO,
    signal_objects: dict[str, SignalObj],
    compiler_module: compiler_rs,
    desktop_setup: bool,
):
    """Builds a Rust representation of the experiment."""
    compiler_rs = compiler_module
    compiler_rs.init_logging(logging.getLogger("laboneq").getEffectiveLevel())

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
            options=_resolve_default_options(
                device_info.options, device_info.device_type, device_info.is_qc
            ),
            reference_clock=device_info.reference_clock_source.name
            if device_info.reference_clock_source is not None
            else None,
        )

    def create_output_routes(
        signal_info: SignalInfo,
    ) -> list[compiler_rs.OutputRoute]:
        return [
            compiler_rs.OutputRoute(
                source_channel=route.from_channel,
                amplitude_scaling=maybe_parameter(route.amplitude),
                phase_shift=maybe_parameter(route.phase),
            )
            for route in signal_info.output_routing or []
        ]

    devices = [
        create_device(device_info)
        for device_info in experiment_dao.device_infos()
        if device_info.device_type != DeviceInfoType.NONQC
    ]

    signals = []
    awg_allocation: dict[int, compiler_rs.AwgInfo] = {}

    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
        device_uid = signal_info.device.uid
        osc = None
        if signal_info.oscillator is not None:
            osc = compiler_rs.Oscillator(
                uid=signal_info.oscillator.uid,
                frequency=maybe_parameter(signal_info.oscillator.frequency),
                is_hardware=signal_info.oscillator.is_hardware is True,
            )
        signal_obj = signal_objects[signal]

        awg_key = hash(signal_obj.awg.key)
        if awg_key not in awg_allocation:
            awg_allocation[awg_key] = compiler_rs.AwgInfo(
                uid=awg_key, number=signal_obj.awg.awg_allocation
            )

        s = compiler_rs.Signal(
            uid=signal,
            awg_key=awg_key,
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
            signal_delay=signal_info.delay_signal or 0.0,
            port_delay=maybe_parameter(signal_obj.port_delay),
            range=(signal_obj.signal_range.value, signal_obj.signal_range.unit)
            if signal_obj.signal_range is not None
            else None,
            precompensation=_build_precompensation(signal_info, compiler_rs),
            added_outputs=create_output_routes(signal_info),
        )
        signals.append(s)

    use_capnp = os.environ.get("LABONEQ_SERIALIZATION", "capnp").lower() != "legacy"
    verify = os.environ.get("LABONEQ_VERIFY_SERIALIZATION", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    if use_capnp:
        use_packed = os.environ.get("LABONEQ_CAPNP_PACKED", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        capnp_bytes = compiler_rs.serialize_experiment(
            experiment_dao.source_experiment, packed=use_packed
        )
        capnp_result = compiler_rs.build_experiment_capnp(
            capnp_bytes,
            signals=signals,
            devices=devices,
            awgs=list(awg_allocation.values()),
            desktop_setup=desktop_setup,
            packed=use_packed,
        )
        if verify:
            legacy_result = compiler_rs.build_experiment(
                experiment=experiment_dao.source_experiment,
                signals=signals,
                devices=devices,
                awgs=list(awg_allocation.values()),
                desktop_setup=desktop_setup,
            )
            compiler_rs.assert_experiment_equivalent(capnp_result, legacy_result)
        return capnp_result
    else:
        return compiler_rs.build_experiment(
            experiment=experiment_dao.source_experiment,
            signals=signals,
            devices=devices,
            awgs=list(awg_allocation.values()),
            desktop_setup=desktop_setup,
        )


def _build_precompensation(
    signal_info: SignalInfo,
    compiler_rs: compiler_rs,
):
    if signal_info.precompensation is None:
        return None
    pc = signal_info.precompensation
    exponential = [
        compiler_rs.ExponentialCompensation(
            timeconstant=tc.timeconstant,
            amplitude=tc.amplitude,
        )
        for tc in pc.exponential or []
    ]
    high_pass = (
        None
        if not pc.high_pass
        else compiler_rs.HighPassCompensation(timeconstant=pc.high_pass.timeconstant)
    )
    bounce = (
        None
        if not pc.bounce
        else compiler_rs.BounceCompensation(
            delay=pc.bounce.delay,
            amplitude=pc.bounce.amplitude,
        )
    )
    fir = (
        None
        if not pc.FIR
        else compiler_rs.FirCompensation(
            coefficients=pc.FIR.coefficients,
        )
    )
    return compiler_rs.Precompensation(
        exponential=exponential,
        high_pass=high_pass,
        bounce=bounce,
        fir=fir,
    )
