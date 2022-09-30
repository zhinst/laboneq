# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from dataclasses import dataclass, field


@dataclass(init=True, repr=True, order=True)
class DeviceOutputSignals:
    """Data structure for storing the output of the signal simulator.

    Access the simulation results as follows:

    >>> device_output_signals = compiled_experiment.output_signals
    >>> device_id = "device_uhfqa"
    >>> awg_id = "0"
    >>> channel_id = 0
    >>> waveform = device_output_signals.signal_by_device_uid(device_id)[awg_id][channel_id]
    >>> waveform
    Waveform(data=array([0., 0., 0., ..., 1., 1., 1.]), sampling_frequency=1800000000.0, time_axis=array([-6.66666667e-08, -6.61111111e-08, -6.55555556e-08, ...,
        1.09983333e-05,  1.09988889e-05,  1.09994444e-05]), time_axis_at_port=array([-6.66666667e-08, -6.61111111e-08, -6.55555556e-08, ...,
        1.09983333e-05,  1.09988889e-05,  1.09994444e-05]), uid='1')


    Warnings:
        This API is highly experimental and may change in future versions.

    """

    #: Data structure that maps device UIDs to the simulation results
    device_signals_map: Any = field(default_factory=dict)

    def __eq__(self, other):
        if self is other:
            return True

        if self.device_signals_map.keys() != other.device_signals_map.keys():
            return False
        for device_id in self.device_signals_map:
            other_device_signal = other.device_signals_map[device_id]
            if other_device_signal.keys() != self.device_signals_map[device_id].keys():
                return False
            for signal_id in self.device_signals_map[device_id]:
                if (
                    self.device_signals_map[device_id][signal_id]
                    != self.device_signals_map[device_id][signal_id]
                ):
                    return False
        return True

    def map(self, device_uid: str, signal_uid: str, channel_index: int, waveform):
        if device_uid not in self.device_signals_map.keys():
            self.device_signals_map[device_uid] = dict()
        if signal_uid not in self.device_signals_map[device_uid]:
            self.device_signals_map[device_uid][signal_uid] = list()

        channels = self.device_signals_map[device_uid][signal_uid]
        delta = channel_index + 1 - len(channels)
        for i in range(delta):
            channels.append(None)

        channels[channel_index] = waveform

    def count_channels(self) -> int:
        """Return the total number of channels across all devices."""
        count = 0
        for uid in self.device_signals_map:
            for awg in self.device_signals_map[uid]:
                waveform_list = self.device_signals_map[uid][awg]
                count += len(waveform_list)
        return count

    @property
    def signals(self):
        """Return a flat list of all signals across all devices.

        Each item in the return value is a dict with the following keys:

        - ``device_uid``: the device UID
        - ``signal_uid``: an identifier for the signal (AWG) on the device
        - ``channels``: a list of :py:class:`~.dsl.result.waveform.Waveform`, one for each channel.
        """
        signals = list()
        for device_id in self.device_signals_map:
            for signal_id in self.device_signals_map[device_id]:
                signals.append(
                    {
                        "device_uid": device_id,
                        "signal_uid": signal_id,
                        "channels": self.device_signals_map[device_id][signal_id],
                    }
                )
        return signals

    def signal_by_device_uid(self, device_uid):
        """Retrieve the simulation results by device UID."""
        return self.device_signals_map.get(device_uid)
