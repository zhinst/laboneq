# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from laboneq.data.compilation_job import DeviceInfoType


@dataclass(eq=True, frozen=True)
class DeviceTraits:
    """Device specific traits.

    Args:
        max_ct_entries: Maximum number of command table entries.
    """

    str_value: str
    sampling_rate: float
    min_play_wave: int
    sample_multiple: int
    supports_zsync: bool
    supports_reset_osc_phase: bool
    supports_binary_waves: bool
    supports_complex_waves: bool
    supports_digital_iq_modulation: bool
    supports_precompensation: bool
    channels_per_awg: int
    is_qa_device: bool
    amplitude_register_count: int = 1
    sampling_rate_2GHz: float = None
    num_integration_units_per_acquire_signal: int = None
    oscillator_set_latency: float = 0.0
    reset_osc_duration: float = 0.0
    supports_oscillator_switching: bool = False
    lo_frequency_granularity: Optional[float] = None
    min_lo_frequency: Optional[float] = None
    max_lo_frequency: Optional[float] = None
    max_ct_entries: Optional[int] = None
    integration_dsp_latency: Optional[float] = None
    spectroscopy_dsp_latency: Optional[float] = None
    has_prng: bool = False
    device_class: int = 0x0


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

    @classmethod
    def from_device_info_type(cls, value: DeviceInfoType):
        return cls(value.name.lower())

    HDAWG = DeviceTraits(
        str_value="hdawg",
        sampling_rate=2.4e9,
        sampling_rate_2GHz=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        amplitude_register_count=4,
        supports_zsync=True,
        supports_reset_osc_phase=True,
        supports_binary_waves=True,
        supports_complex_waves=False,
        supports_digital_iq_modulation=True,
        supports_precompensation=True,
        channels_per_awg=2,
        # @2.4GHz, device grid of 16 samples
        # - 7x16 = 112 cycles (~ 46.7ns) for the setDouble sequence to ensure gapless playback (measured)
        # - 4x16 = 64 additional cycles for the frequency switching (~ 24ns after the above 112 cycles, measured)
        # - 27x16 = 432 cycles for if/else tree for up to 9 tests for 512 steps, assuming ~6 sequencer
        #           cycles (1/8 of sampling rate) per test with possible wait states (estimated worst case)
        # As the value below is given in seconds, calculate it for the worst case of running at 2.0GHz:
        oscillator_set_latency=304e-9,
        # Verified by PW (2022-10-13) on dev8047, proc. FPGA 68603. Observed ~77 ns.
        reset_osc_duration=80e-9,
        supports_oscillator_switching=False,
        # Schema v.1.1.0
        max_ct_entries=1024,
        is_qa_device=False,
        has_prng=True,
        device_class=0x0,
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
        supports_digital_iq_modulation=False,
        supports_precompensation=False,
        channels_per_awg=2,
        num_integration_units_per_acquire_signal=2,
        # Verified by PW (2022-10-13) on dev2086, rev 68366. Observed ~25 ns.
        reset_osc_duration=40e-9,
        supports_oscillator_switching=False,
        is_qa_device=True,
        device_class=0x0,
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
        supports_digital_iq_modulation=False,
        supports_precompensation=False,
        channels_per_awg=1,
        num_integration_units_per_acquire_signal=1,
        oscillator_set_latency=88e-9,
        # Verified by PW (2022-10-13) on dev12093, rev 68689. Observed ~50 ns.
        reset_osc_duration=56e-9,
        lo_frequency_granularity=100e6,
        supports_oscillator_switching=False,
        min_lo_frequency=1e9,
        max_lo_frequency=8.5e9,
        is_qa_device=True,
        integration_dsp_latency=212e-9,
        spectroscopy_dsp_latency=220e-9,
        device_class=0x0,
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
        supports_digital_iq_modulation=True,
        supports_precompensation=False,
        channels_per_awg=1,
        oscillator_set_latency=88e-9,
        # todo (PW): exact worst-case runtime unknown.
        # Verified by PW (2022-10-13) on dev12117, rev 68689. Observed ~35 ns.
        reset_osc_duration=56e-9,
        lo_frequency_granularity=100e6,
        supports_oscillator_switching=True,
        min_lo_frequency=1e9,
        max_lo_frequency=8.5e9,
        # Schema v.1.2.0
        max_ct_entries=4096,
        is_qa_device=False,
        has_prng=True,
        device_class=0x0,
    )
    PRETTYPRINTERDEVICE = DeviceTraits(
        str_value="prettyprinterdevice",
        sampling_rate=2.4e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_zsync=False,
        supports_reset_osc_phase=False,
        supports_binary_waves=False,
        supports_complex_waves=False,
        supports_digital_iq_modulation=False,
        supports_precompensation=False,
        channels_per_awg=2,
        is_qa_device=False,
        device_class=0x1,
    )

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


def validate_local_oscillator_frequency(value: float, device_type: DeviceType):
    """Validate correct local oscillator frequencies.

    Raises:
        ValueError: The value is invalid for given device type.
    """
    if device_type.min_lo_frequency is not None:
        if value < device_type.min_lo_frequency:
            raise ValueError(
                f"({device_type}) Local oscillator frequency {value} Hz is smaller than minimum "
                f"{device_type.min_lo_frequency} Hz."
            )

    if device_type.max_lo_frequency:
        if value > device_type.max_lo_frequency:
            raise ValueError(
                f"({device_type}) Local oscillator frequency {value} Hz is larger than maximum "
                f"{device_type.max_lo_frequency} Hz."
            )

    if device_type.lo_frequency_granularity is not None:
        if value % device_type.lo_frequency_granularity != 0:
            raise ValueError(
                f"({device_type}) Local oscillator frequency {value} "
                f"(device {device_type}) is not multiple of {device_type.lo_frequency_granularity} Hz."
            )
