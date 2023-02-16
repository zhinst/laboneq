# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO


class SamplingRateTracker:
    def __init__(self, experiment_dao: ExperimentDAO, clock_settings):
        self._experiment_dao = experiment_dao
        self._clock_settings = clock_settings
        self._sampling_rate_cache = {}
        self._sequencer_rate_cache = {}

    def sampling_rate_for_device(self, device_id):
        if device_id not in self._sampling_rate_cache:

            device_type = DeviceType(
                self._experiment_dao.device_info(device_id).device_type
            )
            if (
                device_type == DeviceType.HDAWG
                and self._clock_settings["use_2GHz_for_HDAWG"]
            ):
                sampling_rate = DeviceType.HDAWG.sampling_rate_2GHz
            else:
                sampling_rate = device_type.sampling_rate
            self._sampling_rate_cache[device_id] = sampling_rate
        else:
            sampling_rate = self._sampling_rate_cache[device_id]

        return sampling_rate

    def sequencer_rate_for_device(self, device_id):
        if device_id not in self._sequencer_rate_cache:
            device_type = DeviceType(
                self._experiment_dao.device_info(device_id).device_type
            )
            if (
                device_type == DeviceType.HDAWG
                and self._clock_settings["use_2GHz_for_HDAWG"]
            ):
                sampling_rate = DeviceType.HDAWG.sampling_rate_2GHz
            else:
                sampling_rate = device_type.sampling_rate
            sequencer_rate = sampling_rate / device_type.sample_multiple
            self._sequencer_rate_cache[device_id] = sequencer_rate
        else:
            sequencer_rate = self._sequencer_rate_cache[device_id]

        return sequencer_rate
