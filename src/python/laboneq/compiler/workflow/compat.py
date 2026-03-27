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
    from laboneq.data.parameter import Parameter as DataParameter
    from laboneq.dsl.calibration import Oscillator
    from laboneq.dsl.parameter import Parameter


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

    # Builder the device setup capnp payload
    device_setup_capnp = compiler_rs.DeviceSetupBuilder()
    setup_builder = DeviceSetupCompat(device_setup_capnp, experiment_dao.dsl_parameters)

    for device in experiment_dao.device_infos():
        device_setup_capnp.add_instrument(
            uid=device.uid,
            device_type=device.device_type.name,
            options=_resolve_default_options(
                device.options, device.device_type, device.is_qc
            ),
            reference_clock_source=device.reference_clock_source.name
            if device.reference_clock_source is not None
            else None,
            physical_device_uid=device.physical_device_uid,
            is_shfqc=device.is_qc,
        )

    awg_allocation: dict[int, compiler_rs.AwgInfo] = {}

    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
        device_uid = signal_info.device.uid
        signal_obj = signal_objects[signal]
        awg_key = hash(signal_obj.awg.key)

        device_setup_capnp.add_signal_with_calibration(
            uid=signal_info.uid,
            ports=[str(ch) for ch in signal_info.channels],
            instrument_uid=device_uid,
            channel_type=signal_info.type.name,
            awg_core=awg_key,
            # Calibration parameters
            amplitude=setup_builder.maybe_parameter(signal_info.amplitude),
            oscillator=setup_builder.create_oscillator(signal_info.oscillator),
            lo_frequency=setup_builder.maybe_parameter(signal_info.lo_frequency),
            voltage_offset=setup_builder.maybe_parameter(signal_info.voltage_offset),
            amplifier_pump=compiler_rs.AmplifierPump(
                device=signal_info.amplifier_pump.ppc_device.uid,
                channel=signal_info.amplifier_pump.channel,
                pump_frequency=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.pump_frequency
                ),
                pump_power=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.pump_power
                ),
                cancellation_phase=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.cancellation_phase
                ),
                cancellation_attenuation=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.cancellation_attenuation
                ),
                probe_frequency=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.probe_frequency
                ),
                probe_power=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.probe_power
                ),
            )
            if signal_info.amplifier_pump is not None
            else None,
            port_mode=signal_info.port_mode.name.upper()
            if signal_info.port_mode is not None
            else None,
            automute=signal_info.automute,
            signal_delay=signal_info.delay_signal or 0.0,
            port_delay=setup_builder.maybe_parameter(signal_obj.port_delay),
            range=(signal_obj.signal_range.value, signal_obj.signal_range.unit)
            if signal_obj.signal_range is not None
            else None,
            precompensation=_build_precompensation(signal_info, compiler_rs),
            added_outputs=[
                device_setup_capnp.create_output_route(
                    source_signal=str(route.from_channel),
                    amplitude_scaling=setup_builder.maybe_parameter(route.amplitude),
                    phase_shift=setup_builder.maybe_parameter(route.phase),
                )
                for route in signal_info.output_routing or []
            ],
            threshold=t
            if (t := signal_info.threshold) is None or isinstance(t, list)
            else [t],
        )

        if awg_key not in awg_allocation:
            awg_allocation[awg_key] = compiler_rs.AwgInfo(
                uid=awg_key, number=signal_obj.awg.awg_allocation
            )

    use_packed = os.environ.get("LABONEQ_CAPNP_PACKED", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    capnp_bytes = compiler_rs.serialize_experiment(
        experiment_dao.source_experiment,
        device_setup=device_setup_capnp,
        packed=use_packed,
    )
    return compiler_rs.build_experiment_capnp(
        capnp_bytes,
        awgs=list(awg_allocation.values()),
        desktop_setup=desktop_setup,
        packed=use_packed,
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


class DeviceSetupCompat:
    """Helper class to build the device setup payload for the Rust compiler."""

    def __init__(
        self, builder: compiler_rs.DeviceSetupBuilder, dsl_parameters: list[Parameter]
    ):
        self.builder = builder
        self._dsl_parameters = {param.uid: param for param in dsl_parameters}

        self._oscillators: dict[int, compiler_rs.OscillatorRef] = {}
        self._resolved_parameters: dict[int, compiler_rs.SweepParameter] = {}

    def create_oscillator(
        self, oscillator: Oscillator | None
    ) -> compiler_rs.OscillatorRef | None:
        """Helper function to create or retrieve an oscillator reference for the given oscillator.

        This is a minor optimization to avoid creating duplicate oscillator entries in the device setup payload.
        The Rust compiler will still handle deduplication based on the oscillator UID, but this avoids unnecessary entries in the first place.

        We only deduplicate based on the oscillator instance, not by UID, as we'd have to
        validate the uniqueness of UIDs separately: We leave the oscillator UID uniqueness validation to the Rust compiler.
        """
        if oscillator is not None:
            osc_id = id(oscillator)
            if osc_id not in self._oscillators:
                self._oscillators[osc_id] = self.builder.create_oscillator(
                    uid=oscillator.uid,
                    frequency=self.maybe_parameter(oscillator.frequency),
                    modulation=oscillator.modulation_type.name.upper()
                    if oscillator.modulation_type
                    else "AUTO",
                )
            return self._oscillators[osc_id]
        return None

    def maybe_parameter(self, value: Any) -> compiler_rs.SweepParameter | Any:
        """Helper function to convert a parameter info or data parameter to a sweep parameter if it's a sweep, otherwise return the value as is."""
        from laboneq.data.parameter import Parameter as DataParameter

        if isinstance(
            value,
            (ParameterInfo, DataParameter),
        ):
            return _param_to_dsl(self._dsl_parameters[value.uid])
        return value


def _param_to_dsl(param: DataParameter | Parameter) -> Parameter:
    """Ensure the given parameter is converted to a DSL parameter if it's a data parameter, otherwise return as is."""
    # TODO: Carry the original DSL parameters through the compilation job to avoid re-creating them here.
    from laboneq.data.parameter import (
        LinearSweepParameter as DataLinearSweepParameter,
    )
    from laboneq.data.parameter import SweepParameter as DataSweepParameter
    from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter

    if isinstance(param, DataSweepParameter):
        return SweepParameter(
            uid=param.uid,
            values=param.values,
            axis_name=param.axis_name,
            driven_by=[_param_to_dsl(p) for p in param.driven_by or []],
        )
    elif isinstance(param, DataLinearSweepParameter):
        return LinearSweepParameter(
            uid=param.uid,
            start=param.start,
            stop=param.stop,
            count=param.count,
            axis_name=param.axis_name,
        )
    return param
