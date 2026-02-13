# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from laboneq.data.compilation_job import DeviceInfo, DeviceInfoType


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
    supports_binary_waves: bool
    supports_complex_waves: bool
    supports_precompensation: bool
    channels_per_awg: int
    is_qa_device: bool
    sampling_rate_2GHz: float = None
    oscillator_set_latency: float = 0.0
    reset_osc_duration: float = 0.0
    lo_frequency_granularity: Optional[float] = None
    min_lo_frequency: Optional[float] = None
    max_lo_frequency: Optional[float] = None
    max_ct_entries: Optional[int] = None
    supports_output_mute: bool = False
    device_class: int = 0x0
    max_result_vector_length: int | None = None
    scope_max_segments: int | None = None

    def scope_memory_size_samples(self, device_info: DeviceInfo) -> int:
        if self == DeviceType.SHFQA and device_info.is_qc:
            return 64 * 1024
        if self == DeviceType.SHFQA:
            return 256 * 1024
        if self == DeviceType.UHFQA:
            return 4096
        if self == DeviceType.PRETTYPRINTERDEVICE:
            return 133120

        return 0


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
        return cls(value.name.lower())  # @IgnoreException

    HDAWG = DeviceTraits(
        str_value="hdawg",
        sampling_rate=2.4e9,
        sampling_rate_2GHz=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_binary_waves=True,
        supports_complex_waves=False,
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
        max_ct_entries=1024,
        is_qa_device=False,
        device_class=0x0,
    )

    UHFQA = DeviceTraits(
        str_value="uhfqa",
        sampling_rate=1.8e9,
        min_play_wave=16,
        sample_multiple=8,
        supports_binary_waves=True,  # Todo (Pol): useful or not?
        supports_complex_waves=False,
        supports_precompensation=False,
        channels_per_awg=2,
        # Verified by PW (2022-10-13) on dev2086, rev 68366. Observed ~25 ns.
        reset_osc_duration=40e-9,
        is_qa_device=True,
        device_class=0x0,
        max_result_vector_length=1 << 20,
        scope_max_segments=1,
    )

    SHFQA = DeviceTraits(
        str_value="shfqa",
        sampling_rate=2.0e9,
        # TODO(2K):
        # https://docs.zhinst.com/shfqa_user_manual/specifications.html#digital-signal-processing-specifications
        # - minimum waveform length
        # - waveform granularity
        # - minimum weight length
        # - integration weight granularity
        # min_play_wave=4,
        # sample_multiple=4,
        min_play_wave=32,
        sample_multiple=16,
        supports_binary_waves=False,
        supports_complex_waves=True,
        supports_precompensation=False,
        channels_per_awg=1,
        oscillator_set_latency=88e-9,
        # Verified by PW (2022-10-13) on dev12093, rev 68689. Observed ~50 ns.
        reset_osc_duration=56e-9,
        lo_frequency_granularity=100e6,
        min_lo_frequency=1e9,
        max_lo_frequency=8.5e9,
        is_qa_device=True,
        device_class=0x0,
        supports_output_mute=True,
        max_result_vector_length=1 << 19,
        scope_max_segments=1024,
    )
    SHFSG = DeviceTraits(
        str_value="shfsg",
        sampling_rate=2.0e9,
        min_play_wave=32,
        sample_multiple=16,
        supports_binary_waves=True,
        supports_complex_waves=False,
        supports_precompensation=False,
        channels_per_awg=1,
        oscillator_set_latency=88e-9,
        # todo (PW): exact worst-case runtime unknown.
        # Verified by PW (2022-10-13) on dev12117, rev 68689. Observed ~35 ns.
        reset_osc_duration=56e-9,
        lo_frequency_granularity=100e6,
        min_lo_frequency=1e9,
        max_lo_frequency=8.5e9,
        max_ct_entries=4096,
        is_qa_device=False,
        supports_output_mute=True,
        device_class=0x0,
    )
    PRETTYPRINTERDEVICE = DeviceTraits(
        str_value="prettyprinterdevice",
        sampling_rate=2.0e9,
        min_play_wave=4,
        sample_multiple=4,
        supports_binary_waves=False,
        supports_complex_waves=False,
        supports_precompensation=False,
        channels_per_awg=1,
        is_qa_device=False,
        device_class=0x1,
        oscillator_set_latency=36e-9,
        reset_osc_duration=32e-9,
    )

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
