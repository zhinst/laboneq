# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import asdict, is_dataclass
import attrs
from typing import Any, Callable, Optional

from laboneq.core.types import enums as legacy_enums
from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.data import calibration
from laboneq.dsl import calibration as legacy_calibration
from laboneq.dsl import parameter as legacy_parameter
from laboneq.implementation.legacy_adapters.utils import (
    LogicalSignalPhysicalChannelUID,
    raise_not_implemented,
)


def _change_type(source: Any, target: Any) -> Any:
    if source is None:
        return None
    if is_dataclass(source):
        return target(**asdict(source))
    if attrs.has(source.__class__):
        return target(**attrs.asdict(source))
    raise_not_implemented(source)


def convert_maybe_parameter(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, legacy_parameter.SweepParameter):
        # local import to avoid circular dependency:
        from laboneq.implementation.legacy_adapters.converters_experiment_description import (
            convert_SweepParameter,
        )

        return convert_SweepParameter(obj)
    if isinstance(obj, legacy_parameter.LinearSweepParameter):
        # local import to avoid circular dependency:
        from laboneq.implementation.legacy_adapters.converters_experiment_description import (
            convert_LinearSweepParameter,
        )

        return convert_LinearSweepParameter(obj)
    if isinstance(obj, (float, int)):
        return obj
    raise_not_implemented(obj)


def convert_modulation_type(
    obj: legacy_enums.ModulationType | None,
) -> calibration.ModulationType | None:
    if obj is None:
        return None
    if obj == legacy_enums.ModulationType.AUTO:
        return calibration.ModulationType.AUTO
    if obj == legacy_enums.ModulationType.HARDWARE:
        return calibration.ModulationType.HARDWARE
    if obj == legacy_enums.ModulationType.SOFTWARE:
        return calibration.ModulationType.SOFTWARE
    raise_not_implemented(obj)


def convert_oscillator(
    obj: legacy_calibration.Oscillator | None,
) -> Optional[calibration.Oscillator]:
    if obj is None:
        return None
    return calibration.Oscillator(
        uid=obj.uid,
        frequency=convert_maybe_parameter(obj.frequency),
        modulation_type=convert_modulation_type(obj.modulation_type),
    )


def convert_mixer_calibration(
    obj: legacy_calibration.MixerCalibration | None,
) -> Optional[calibration.MixerCalibration]:
    if obj is None:
        return None
    if isinstance(obj, legacy_calibration.MixerCalibration):
        return calibration.MixerCalibration(
            uid=obj.uid,
            voltage_offsets=[
                convert_maybe_parameter(val) for val in obj.voltage_offsets or []
            ],
            correction_matrix=[
                [convert_maybe_parameter(ij) for ij in row]
                for row in obj.correction_matrix or []
            ],
        )
    raise_not_implemented(obj)


def convert_exponential(
    obj: legacy_calibration.ExponentialCompensation | None,
) -> Optional[calibration.ExponentialCompensation]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [convert_exponential(x) for x in obj]
    return calibration.ExponentialCompensation(
        timeconstant=obj.timeconstant, amplitude=obj.amplitude
    )


def convert_highpass_compensation(
    obj: legacy_calibration.HighPassCompensation | None,
) -> Optional[calibration.HighPassCompensation]:
    if obj is None:
        return None
    return calibration.HighPassCompensation(timeconstant=obj.timeconstant)


def convert_precompensation(
    obj: legacy_calibration.Precompensation | None,
) -> Optional[calibration.Precompensation]:
    if obj is None:
        return obj
    new = calibration.Precompensation()
    new.uid = obj.uid
    new.exponential = convert_exponential(obj.exponential)
    new.high_pass = convert_highpass_compensation(obj.high_pass)
    new.bounce = _change_type(obj.bounce, calibration.BounceCompensation)
    new.FIR = _change_type(obj.FIR, calibration.FIRCompensation)
    return new


def convert_amplifier_pump(
    obj: legacy_calibration.AmplifierPump | None,
) -> Optional[calibration.AmplifierPump]:
    if obj is None:
        return obj
    return calibration.AmplifierPump(
        uid=obj.uid,
        pump_on=obj.pump_on,
        pump_frequency=convert_maybe_parameter(obj.pump_frequency),
        pump_power=convert_maybe_parameter(obj.pump_power),
        pump_filter_on=obj.pump_filter_on,
        cancellation_on=obj.cancellation_on,
        cancellation_phase=convert_maybe_parameter(obj.cancellation_phase),
        cancellation_attenuation=convert_maybe_parameter(obj.cancellation_attenuation),
        cancellation_source=obj.cancellation_source,
        cancellation_source_frequency=convert_maybe_parameter(
            obj.cancellation_source_frequency
        ),
        alc_on=obj.alc_on,
        probe_on=obj.probe_on,
        probe_frequency=convert_maybe_parameter(obj.probe_frequency),
        probe_power=convert_maybe_parameter(obj.probe_power),
    )


def convert_port_mode(
    obj: legacy_enums.PortMode | None,
) -> Optional[calibration.PortMode]:
    if obj is None:
        return None
    if obj == legacy_enums.PortMode.LF:
        return calibration.PortMode.LF
    if obj == legacy_enums.PortMode.RF:
        return calibration.PortMode.RF
    raise_not_implemented(obj)


def format_ls_pc_uid(seq: str) -> str:
    return LogicalSignalPhysicalChannelUID(seq).uid


def convert_calibration(
    target: legacy_calibration.Calibration,
    uid_formatter: Callable[[str], str] = format_ls_pc_uid,
) -> calibration.Calibration:
    cals = {}
    legacy_ls: legacy_calibration.SignalCalibration
    for uid, legacy_ls in target.calibration_items.items():
        if legacy_ls is None:
            continue
        else:
            new = calibration.SignalCalibration()
            new.oscillator = convert_oscillator(getattr(legacy_ls, "oscillator", None))
            if legacy_ls.local_oscillator is not None:
                if (
                    legacy_ls.local_oscillator.modulation_type
                    == ModulationType.SOFTWARE
                ):
                    raise ValueError(
                        "Encountered `ModulationType.SOFTWARE` in local oscillator configuration "
                        "which is not allowed. Make sure modulation type for "
                        "all local oscillator calibration settings is set to "
                        "either `ModulationType.HARDWARE` or `ModulationType.AUTO`."
                    )
                new.local_oscillator_frequency = convert_maybe_parameter(
                    legacy_ls.local_oscillator.frequency
                )
            new.mixer_calibration = convert_mixer_calibration(
                legacy_ls.mixer_calibration
            )
            new.precompensation = convert_precompensation(legacy_ls.precompensation)
            new.port_delay = convert_maybe_parameter(legacy_ls.port_delay)
            new.delay_signal = getattr(legacy_ls, "delay_signal", None)
            new.port_mode = convert_port_mode(legacy_ls.port_mode)
            new.voltage_offset = convert_maybe_parameter(legacy_ls.voltage_offset)
            new.range = legacy_ls.range
            new.threshold = getattr(legacy_ls, "threshold", None)
            new.amplitude = convert_maybe_parameter(legacy_ls.amplitude)
            new.amplifier_pump = convert_amplifier_pump(legacy_ls.amplifier_pump)
            if legacy_ls.added_outputs is not None:
                for router in legacy_ls.added_outputs:
                    routing = calibration.OutputRouting(
                        source_signal=router.source,
                        amplitude=convert_maybe_parameter(router.amplitude_scaling),
                        phase=convert_maybe_parameter(router.phase_shift),
                    )
                    new.output_routing.append(routing)
            new.automute = getattr(legacy_ls, "automute", None)
        cals[uid_formatter(uid)] = new

    return calibration.Calibration(cals)
