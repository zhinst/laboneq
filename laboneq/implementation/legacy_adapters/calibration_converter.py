# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from laboneq.core.types import enums as legacy_enums
from laboneq.data import calibration, parameter
from laboneq.dsl import calibration as legacy_calibration
from laboneq.dsl import parameter as legacy_parameter
from laboneq.implementation.legacy_adapters.utils import (
    LogicalSignalPhysicalChannelUID,
    raise_not_implemented,
)


def _change_dataclass_type(source: Any, target: Any) -> Any:
    if source is None:
        return None
    if is_dataclass(source):
        return target(**asdict(source))
    raise_not_implemented(source)


def convert_maybe_parameter(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, legacy_parameter.SweepParameter):
        return parameter.SweepParameter(
            uid=obj.uid, values=obj.values, axis_name=obj.axis_name
        )
    if isinstance(obj, legacy_parameter.LinearSweepParameter):
        return parameter.LinearSweepParameter(
            uid=obj.uid,
            start=obj.start,
            stop=obj.stop,
            count=obj.count,
            axis_name=obj.axis_name,
        )
    if isinstance(obj, (float, int)):
        return obj
    raise_not_implemented(obj)


def convert_modulation_type(
    obj: Optional[legacy_enums.ModulationType],
) -> Optional[calibration.ModulationType]:
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
    obj: Optional[legacy_calibration.Oscillator],
) -> Optional[calibration.Oscillator]:
    if obj is None:
        return None
    return calibration.Oscillator(
        uid=obj.uid,
        frequency=convert_maybe_parameter(obj.frequency),
        modulation_type=convert_modulation_type(obj.modulation_type),
    )


def convert_mixer_calibration(
    obj: Optional[legacy_calibration.MixerCalibration],
) -> Optional[calibration.MixerCalibration]:
    if obj is None:
        return None
    if isinstance(obj, legacy_calibration.MixerCalibration):
        return calibration.MixerCalibration(**asdict(obj))
    raise_not_implemented(obj)


def convert_exponential(
    obj: Optional[legacy_calibration.ExponentialCompensation],
) -> Optional[calibration.ExponentialCompensation]:
    if obj is None:
        return None
    if isinstance(obj, list):
        return [convert_exponential(x) for x in obj]
    return calibration.ExponentialCompensation(
        timeconstant=obj.timeconstant, amplitude=obj.amplitude
    )


def convert_highpass_compensation(
    obj: Optional[legacy_calibration.HighPassCompensation],
) -> Optional[calibration.HighPassCompensation]:
    if obj is None:
        return None
    return calibration.HighPassCompensation(timeconstant=obj.timeconstant)


def convert_precompensation(
    obj: Optional[legacy_calibration.Precompensation],
) -> Optional[calibration.Precompensation]:
    if obj is None:
        return obj
    new = calibration.Precompensation()
    new.uid = obj.uid
    new.exponential = convert_exponential(obj.exponential)
    new.high_pass = convert_highpass_compensation(obj.high_pass)
    new.bounce = _change_dataclass_type(obj.bounce, calibration.BounceCompensation)
    new.FIR = _change_dataclass_type(obj.FIR, calibration.FIRCompensation)
    return new


def convert_amplifier_pump(
    obj: Optional[legacy_calibration.AmplifierPump],
) -> Optional[calibration.AmplifierPump]:
    if obj is None:
        return obj
    return calibration.AmplifierPump(
        uid=obj.uid,
        pump_freq=convert_maybe_parameter(obj.pump_freq),
        pump_power=convert_maybe_parameter(obj.pump_power),
        alc_engaged=obj.alc_engaged,
        use_probe=obj.use_probe,
        probe_frequency=convert_maybe_parameter(obj.probe_frequency),
        probe_power=convert_maybe_parameter(obj.probe_power),
    )


def convert_port_mode(
    obj: Optional[legacy_enums.PortMode],
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
    for uid, legacy_ls in target.calibration_items.items():
        if legacy_ls is None:
            continue
        else:
            new = calibration.SignalCalibration()
            new.oscillator = convert_oscillator(legacy_ls.oscillator)
            if legacy_ls.local_oscillator is not None:
                new.local_oscillator_frequency = convert_maybe_parameter(
                    legacy_ls.local_oscillator.frequency
                )
            new.mixer_calibration = convert_mixer_calibration(
                legacy_ls.mixer_calibration
            )
            new.precompensation = convert_precompensation(legacy_ls.precompensation)
            new.port_delay = convert_maybe_parameter(legacy_ls.port_delay)
            new.delay_signal = legacy_ls.delay_signal
            new.port_mode = convert_port_mode(legacy_ls.port_mode)
            new.voltage_offset = legacy_ls.voltage_offset
            new.range = legacy_ls.range
            new.threshold = legacy_ls.threshold
            new.amplitude = convert_maybe_parameter(legacy_ls.amplitude)
            new.amplifier_pump = convert_amplifier_pump(legacy_ls.amplifier_pump)
        cals[uid_formatter(uid)] = new

    return calibration.Calibration(cals)
