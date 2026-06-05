# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for building Rust experiment representation."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING

from laboneq.compiler.workflow import compiler_hooks
from laboneq.data.compilation_job import (
    ExperimentInfo,
    SignalInfo,
)

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.data.compilation_job import (
        ExperimentInfo,
        SignalInfo,
    )
    from laboneq.dsl.calibration import Oscillator


def serialize_capnp(
    experiment_info: ExperimentInfo,
    device_class: int,
) -> bytes:
    """Serializes the experiment into a Cap'n Proto payload."""
    compiler_module = compiler_hooks.resolve_compiler_module(device_class)
    compiler_module.init_logging(logging.getLogger("laboneq").getEffectiveLevel())

    # Builder the device setup capnp payload
    device_setup_capnp = compiler_module.DeviceSetupBuilder()
    setup_builder = DeviceSetupCompat(device_setup_capnp)

    if device_class == 1 and experiment_info.setup_description is not None:
        assert experiment_info.setup_description.data is not None, (
            "Setup description `.data` must not be None if setup description is provided"
        )
        device_setup_capnp.set_zqcs_setup_description(
            experiment_info.setup_description.data
        )

    for device in experiment_info.devices:
        device_setup_capnp.add_instrument(
            uid=device.uid,
            device_type=device.device_type.name,
            options=device.options.upper().split("/") if device.options else [],
            reference_clock_source=device.reference_clock_source.name
            if device.reference_clock_source is not None
            else None,
        )

    for signal_info in experiment_info.signals:
        device_uid = signal_info.device_uid
        device_setup_capnp.add_signal_with_calibration(
            uid=signal_info.uid,
            ports=signal_info.ports,
            instrument_uid=device_uid,
            channel_type=signal_info.type.name,
            # Calibration parameters
            amplitude=signal_info.amplitude,
            oscillator=setup_builder.create_oscillator(signal_info.oscillator),
            lo_frequency=signal_info.lo_frequency,
            voltage_offset=signal_info.voltage_offset,
            amplifier_pump=compiler_module.AmplifierPump(
                device=signal_info.amplifier_pump.ppc_device.uid,
                channel=signal_info.amplifier_pump.channel,
                alc_on=signal_info.amplifier_pump.alc_on,
                pump_on=signal_info.amplifier_pump.pump_on,
                pump_filter_on=signal_info.amplifier_pump.pump_filter_on,
                pump_frequency=signal_info.amplifier_pump.pump_frequency,
                pump_power=signal_info.amplifier_pump.pump_power,
                cancellation_on=signal_info.amplifier_pump.cancellation_on,
                cancellation_phase=signal_info.amplifier_pump.cancellation_phase,
                cancellation_attenuation=signal_info.amplifier_pump.cancellation_attenuation,
                cancellation_source=signal_info.amplifier_pump.cancellation_source.name,
                cancellation_source_frequency=signal_info.amplifier_pump.cancellation_source_frequency,
                probe_on=signal_info.amplifier_pump.probe_on,
                probe_frequency=signal_info.amplifier_pump.probe_frequency,
                probe_power=signal_info.amplifier_pump.probe_power,
            )
            if signal_info.amplifier_pump is not None
            else None,
            port_mode=signal_info.port_mode.name.upper()
            if signal_info.port_mode is not None
            else None,
            automute=signal_info.automute,
            signal_delay=signal_info.delay_signal or 0.0,
            port_delay=signal_info.port_delay,
            range=(signal_info.signal_range.value, signal_info.signal_range.unit)
            if signal_info.signal_range is not None
            else None,
            precompensation=_build_precompensation(signal_info, compiler_module),
            added_outputs=[
                device_setup_capnp.create_output_route(
                    source_signal=route.from_port,
                    amplitude_scaling=route.amplitude,
                    phase_shift=route.phase,
                )
                for route in signal_info.output_routing or []
            ],
            mixer_calibration=(
                device_setup_capnp.create_mixer_calibration(
                    voltage_offsets=signal_info.mixer_calibration.voltage_offsets,
                    correction_matrix=signal_info.mixer_calibration.correction_matrix,
                )
            )
            if signal_info.mixer_calibration is not None
            else None,
            threshold=t
            if (t := signal_info.threshold) is None or isinstance(t, list)
            else [t],
        )

    return compiler_module.serialize_experiment(
        experiment_info.src,
        device_setup=device_setup_capnp,
        packed=use_packed_capnp(),
    )


def use_packed_capnp() -> bool:
    """Determines whether to use packed Cap'n Proto serialization based on the environment variable."""
    return os.environ.get("LABONEQ_CAPNP_PACKED", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def sanitize_compiler_settings(settings: dict | None) -> dict:
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
            strict=pc.FIR.strict,
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

    def __init__(self, builder: compiler_rs.DeviceSetupBuilder):
        self.builder = builder

        self._oscillators: dict[int, compiler_rs.OscillatorRef] = {}

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
                    frequency=oscillator.frequency,
                    modulation=oscillator.modulation_type.name.upper()
                    if oscillator.modulation_type
                    else "AUTO",
                )
            return self._oscillators[osc_id]
        return None
