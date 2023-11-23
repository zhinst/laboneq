# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Module for calculating on device delays and adjusting them to the grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, NamedTuple
import numpy as np
import math
from laboneq.data.compilation_job import OutputRoute


class OnDeviceDelayInfo(NamedTuple):
    """On device delay information.

    Attributes:
        sampling_rate: Sampling rate of the device in ns.
        sample_multiple: Sample multiple of the device.
        delay_samples: Delay in samples.
    """

    sampling_rate: float
    sample_multiple: int
    delay_samples: int


@dataclass
class OnDeviceDelayCompensation:
    """On device delay compensation.

    Attributes:
        on_signal: Signal delay in ns.
        on_port: Port delay in ns.
    """

    on_signal: float
    on_port: float

    @property
    def total_delay(self) -> float:
        return self.on_signal + self.on_port


def calculate_output_router_delays(
    mapping: Mapping[str, Sequence[OutputRoute]]
) -> dict[str, int]:
    """Calculate delays introduced from using Output router.

    Using output router introduces a constant delay of 52 samples.
    """
    signal_on_device_delays = {}
    for uid, routing in mapping.items():
        if routing:
            signal_on_device_delays[uid] = 52
            for output in routing:
                if output.from_signal:
                    assert (
                        output.from_signal in mapping
                    ), "Source signal does not exists."
                    signal_on_device_delays[output.from_signal] = 52
        else:
            if uid not in signal_on_device_delays:
                signal_on_device_delays[uid] = 0
    return signal_on_device_delays


def compensate_on_device_delays(
    items: dict[str, OnDeviceDelayInfo]
) -> dict[str, dict[str, OnDeviceDelayCompensation]]:
    """Compensate on device delays.

    Args:
        items: A mapping, where key is an uid and values are
            sampling_rate, sample_multiple and delays_samples.

    Returns:
        Signal and device port delays which are adjusted to delays on device per uid.
    """
    if not items:
        return {}
    unique_sequencer_rates = set()
    max_delay = 0
    for values in items.values():
        (sampling_rate, sample_multiple, delay_samples) = values
        sequencer_rate = sampling_rate / sample_multiple
        unique_sequencer_rates.add(int(sequencer_rate))

        delay = delay_samples / sampling_rate
        if max_delay < delay:
            max_delay = delay

    common_sequencer_rate = np.gcd.reduce(list(unique_sequencer_rates))
    system_grid = 1.0 / common_sequencer_rate
    max_delay = math.ceil(max_delay / system_grid) * system_grid
    key_to_delays = {}
    for key, values in items.items():
        (sampling_rate, sample_multiple, delay_samples) = values
        initial_delay_samples = 0
        max_delay_samples = max_delay * sampling_rate
        initial_delay_samples += max_delay_samples - delay_samples

        delay_signal = (
            (initial_delay_samples // sample_multiple) / sampling_rate * sample_multiple
        )
        port_delay = (initial_delay_samples % sample_multiple) / sampling_rate
        delay_signal = delay_signal if abs(delay_signal) > 1e-12 else 0
        port_delay = port_delay if abs(port_delay) > 1e-12 else 0
        key_to_delays[key] = OnDeviceDelayCompensation(
            on_signal=delay_signal, on_port=port_delay
        )
    return key_to_delays
