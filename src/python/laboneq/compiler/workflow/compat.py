# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for building Rust experiment representation."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

from laboneq.data.compilation_job import DeviceInfoType, ParameterInfo, SignalInfo
from laboneq.implementation.utils.devices import parse_device_options

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
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
    compiler_module: compiler_rs,
    desktop_setup: bool,
    compiler_settings: dict | None = None,
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

    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
        device_uid = signal_info.device.uid
        ports = list(signal_info.channel_to_port.values())
        device_setup_capnp.add_signal_with_calibration(
            uid=signal_info.uid,
            ports=ports,
            instrument_uid=device_uid,
            channel_type=signal_info.type.name,
            # Calibration parameters
            amplitude=setup_builder.maybe_parameter(signal_info.amplitude),
            oscillator=setup_builder.create_oscillator(signal_info.oscillator),
            lo_frequency=setup_builder.maybe_parameter(signal_info.lo_frequency),
            voltage_offset=setup_builder.maybe_parameter(signal_info.voltage_offset),
            amplifier_pump=compiler_rs.AmplifierPump(
                device=signal_info.amplifier_pump.ppc_device.uid,
                channel=signal_info.amplifier_pump.channel,
                alc_on=signal_info.amplifier_pump.alc_on,
                pump_on=signal_info.amplifier_pump.pump_on,
                pump_filter_on=signal_info.amplifier_pump.pump_filter_on,
                pump_frequency=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.pump_frequency
                ),
                pump_power=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.pump_power
                ),
                cancellation_on=signal_info.amplifier_pump.cancellation_on,
                cancellation_phase=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.cancellation_phase
                ),
                cancellation_attenuation=setup_builder.maybe_parameter(
                    signal_info.amplifier_pump.cancellation_attenuation
                ),
                cancellation_source=signal_info.amplifier_pump.cancellation_source.name,
                cancellation_source_frequency=signal_info.amplifier_pump.cancellation_source_frequency,
                probe_on=signal_info.amplifier_pump.probe_on,
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
            port_delay=setup_builder.maybe_parameter(signal_info.port_delay),
            range=(signal_info.signal_range.value, signal_info.signal_range.unit)
            if signal_info.signal_range is not None
            else None,
            precompensation=_build_precompensation(signal_info, compiler_rs),
            added_outputs=[
                device_setup_capnp.create_output_route(
                    source_signal=route.from_port,
                    amplitude_scaling=setup_builder.maybe_parameter(route.amplitude),
                    phase_shift=setup_builder.maybe_parameter(route.phase),
                )
                for route in signal_info.output_routing or []
            ],
            mixer_calibration=(
                device_setup_capnp.create_mixer_calibration(
                    voltage_offsets=[
                        setup_builder.maybe_parameter(offset)
                        for offset in signal_info.mixer_calibration.voltage_offsets
                    ],
                    correction_matrix=[
                        [setup_builder.maybe_parameter(element) for element in row]
                        for row in signal_info.mixer_calibration.correction_matrix
                    ],
                )
            )
            if signal_info.mixer_calibration is not None
            else None,
            threshold=t
            if (t := signal_info.threshold) is None or isinstance(t, list)
            else [t],
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
        desktop_setup=desktop_setup,
        packed=use_packed,
        compiler_settings=_sanitize_compiler_settings(compiler_settings),
    )


def _sanitize_compiler_settings(settings: dict | None) -> dict:
    if settings is None:
        return {}
    settings = settings.copy()  # Create a copy to avoid mutating the original
    # Ensure resolution bits are non-negative integers.
    # This did not previously raise an error so therefore we sanitize the input instead of raising an error.
    if "AMPLITUDE_RESOLUTION_BITS" in settings:
        settings["AMPLITUDE_RESOLUTION_BITS"] = max(
            settings["AMPLITUDE_RESOLUTION_BITS"], 0
        )
    if "PHASE_RESOLUTION_BITS" in settings:
        settings["PHASE_RESOLUTION_BITS"] = max(settings["PHASE_RESOLUTION_BITS"], 0)
    if "MAX_EVENTS_TO_PUBLISH" in settings:
        if isinstance(
            settings["MAX_EVENTS_TO_PUBLISH"], float
        ):  # We support e.g. 1e6 for convenience, but we need to convert it to int for the Rust compiler
            settings["MAX_EVENTS_TO_PUBLISH"] = int(settings["MAX_EVENTS_TO_PUBLISH"])

    if "EXPAND_LOOPS_FOR_SCHEDULE" in settings:
        warnings.warn(
            "Setting `EXPAND_LOOPS_FOR_SCHEDULE` is deprecated.\n"
            "Use the expand_loops_for_schedule argument of laboneq.pulse_sheet_viewer.pulse_sheet_viewer.view_pulse_sheet"
            " to set loop expansion for the pulse sheet viewer",
            FutureWarning,
            stacklevel=2,
        )

    if "SHFSG_FORCE_COMMAND_TABLE" in settings:
        warnings.warn(
            "The setting `SHFSG_FORCE_COMMAND_TABLE` has no effect and will be removed in a future version",
            FutureWarning,
            stacklevel=2,
        )
        settings.pop("SHFSG_FORCE_COMMAND_TABLE")

    if "HDAWG_FORCE_COMMAND_TABLE" in settings:
        warnings.warn(
            "The setting `HDAWG_FORCE_COMMAND_TABLE` has no effect and will be removed in a future version",
            FutureWarning,
            stacklevel=2,
        )
        settings.pop("HDAWG_FORCE_COMMAND_TABLE")

    if "SHFQA_MIN_PLAYWAVE_HINT" in settings:
        warnings.warn(
            "The setting `SHFQA_MIN_PLAYWAVE_HINT` has no effect.",
            FutureWarning,
            stacklevel=2,
        )
        settings.pop("SHFQA_MIN_PLAYWAVE_HINT")

    if "SHFQA_MIN_PLAYZERO_HINT" in settings:
        warnings.warn(
            "The setting `SHFQA_MIN_PLAYZERO_HINT` has no effect.",
            FutureWarning,
            stacklevel=2,
        )
        settings.pop("SHFQA_MIN_PLAYZERO_HINT")

    if ("MAX_EVENTS_TO_PUBLISH" in settings) and ("OUTPUT_EXTRAS" not in settings):
        warnings.warn(
            "Setting `MAX_EVENTS_TO_PUBLISH` has no effect unless used together with `OUTPUT_EXTRAS=True`.",
            FutureWarning,
            stacklevel=2,
        )

    return settings


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
