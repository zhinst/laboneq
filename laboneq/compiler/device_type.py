# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from enum import Enum
import math


@dataclass(eq=True, frozen=True)
class DeviceTraits:
    str_value: str
    sampling_rate: float
    min_play_wave: int
    sample_multiple: int
    supports_zsync: bool
    supports_reset_osc_phase: bool
    supports_binary_waves: bool
    supports_complex_waves: bool
    supports_mixer_calibration: bool
    channels_per_awg: int
    iq_phase: float
    sampling_rate_2GHz: float = None
    num_integration_units_per_acquire_signal: int = None
    oscillator_set_latency: float = 0.0
    reset_osc_duration: float = 0.0


class DeviceType(DeviceTraits, Enum):
    def __new__(cls, value: DeviceTraits):
        # This is needed to ensure DeviceType(<str_value>) resolves to the corresponding enum
        obj = object.__new__(cls)
        obj._value_ = value.str_value
        return obj

    def __init__(self, value: DeviceTraits):
        # This is needed to ensure DeviceType instance (created in custom __new__() above)
        # gets properly initialized with the original DeviceTraits value members
        super().__init__(**asdict(value))

    HDAWG = DeviceTraits(
        str_value="hdawg",
        sampling_rate=2.4e9,
        sampling_rate_2GHz=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_zsync=True,
        supports_reset_osc_phase=True,
        supports_binary_waves=True,
        supports_complex_waves=False,
        supports_mixer_calibration=True,
        channels_per_awg=2,
        iq_phase=0.0,
        # todo (PW): exact worst-case runtime unknown. This is SD's estiamte for upper
        # bound.
        reset_osc_duration=40e-9,
    )

    UHFQA = DeviceTraits(
        str_value="uhfqa",
        sampling_rate=1.8e9,
        min_play_wave=16,
        sample_multiple=8,
        supports_zsync=False,
        supports_reset_osc_phase=True,
        supports_binary_waves=True,  # Todo (Pol): useful or not?
        supports_complex_waves=False,
        supports_mixer_calibration=False,
        channels_per_awg=2,
        num_integration_units_per_acquire_signal=2,
        iq_phase=math.pi / 4,
        # todo (PW): exact worst-case runtime unknown. This is SD's estimate for upper
        # bound.
        reset_osc_duration=40e-9,
    )

    SHFQA = DeviceTraits(
        str_value="shfqa",
        sampling_rate=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_zsync=True,
        supports_reset_osc_phase=True,  # Todo (Pol): useful or not?
        supports_binary_waves=False,
        supports_complex_waves=True,
        supports_mixer_calibration=False,
        channels_per_awg=1,
        num_integration_units_per_acquire_signal=1,
        iq_phase=0.0,
        oscillator_set_latency=88e-9,
        # todo (PW): exact worst-case runtime unknown. This is SD's estimate for upper
        # bound.
        reset_osc_duration=40e-9,
    )
    SHFSG = DeviceTraits(
        str_value="shfsg",
        sampling_rate=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_zsync=True,
        supports_reset_osc_phase=True,
        supports_binary_waves=True,
        supports_complex_waves=False,
        supports_mixer_calibration=True,
        channels_per_awg=1,
        iq_phase=0.0,
        oscillator_set_latency=88e-9,
        # todo (PW): exact worst-case runtime unknown. This is SD's estimate for upper
        # bound.
        reset_osc_duration=40e-9,
    )

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
