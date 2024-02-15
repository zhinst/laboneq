# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Iterator

from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.communication import DaqNodeSetAction
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.data.calibration import CancellationSource
from laboneq.data.recipe import Initialization


class DeviceSHFPPC(DeviceZI):
    attribute_keys = {
        "cancellation_phase": AttributeName.PPC_CANCELLATION_PHASE,
        "cancellation_attenuation": AttributeName.PPC_CANCELLATION_ATTENUATION,
        "pump_frequency": AttributeName.PPC_PUMP_FREQUENCY,
        "pump_power": AttributeName.PPC_PUMP_POWER,
        "probe_frequency": AttributeName.PPC_PROBE_FREQUENCY,
        "probe_power": AttributeName.PPC_PROBE_POWER,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFPPC"
        self.dev_opts = []
        self._use_internal_clock = False
        self._channels = 4  # TODO(2K): Update from device

    def _key_to_path(self, key: str, ch: int):
        keys_to_paths = {
            "pump_on": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/on",
            "pump_frequency": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/freq",
            "pump_power": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/power",
            "pump_filter_on": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/filter",
            "cancellation_on": f"/{self.serial}/ppchannels/{ch}/cancellation/on",
            "cancellation_source": f"/{self.serial}/ppchannels/{ch}/cancellation/source",
            "cancellation_source_frequency": f"/{self.serial}/ppchannels/{ch}/cancellation/sourcefreq",
            "cancellation_phase": f"/{self.serial}/ppchannels/{ch}/cancellation/phaseshift",
            "cancellation_attenuation": f"/{self.serial}/ppchannels/{ch}/cancellation/attenuation",
            "alc_on": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/alc",
            "probe_on": f"/{self.serial}/ppchannels/{ch}/synthesizer/probe/on",
            "probe_frequency": f"/{self.serial}/ppchannels/{ch}/synthesizer/probe/freq",
            "probe_power": f"/{self.serial}/ppchannels/{ch}/synthesizer/probe/power",
        }
        return keys_to_paths.get(key)

    def update_clock_source(self, force_internal: bool | None):
        self._use_internal_clock = force_internal is True

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)
        ppchannels = initialization.ppchannels or []
        for settings in ppchannels:
            channel = settings["channel"]
            for key, attribute_name in DeviceSHFPPC.attribute_keys.items():
                if key in settings:
                    yield DeviceAttribute(
                        name=attribute_name, index=channel, value_or_param=settings[key]
                    )

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        return []

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector()
        ppchannels = {
            settings["channel"]: settings
            for settings in initialization.ppchannels or []
        }

        def _convert(value):
            if isinstance(value, bool):
                return 1 if value else 0
            return value

        # each channel uses the neighboring channel's synthesizer for generating the pump tone
        probe_synth_channel = [1, 0, 3, 2]

        for ch, settings in ppchannels.items():
            for key, value in settings.items():
                if key == "channel":
                    continue
                if key == "probe_on" and value:
                    probe_channel = ppchannels.get(probe_synth_channel[ch])
                    if probe_channel is not None and probe_channel["pump_on"]:
                        raise LabOneQControllerException(
                            f"{self.dev_repr}: cannot use probe tone on"
                            f" channel {ch} while the pump tone generation is also"
                            f" enabled on channel {probe_synth_channel[ch]}"
                        )
                elif key == "cancellation_source":
                    if value == CancellationSource.INTERNAL:
                        value = 0
                    else:
                        assert value == CancellationSource.EXTERNAL
                        value = 1
                        if settings.get("cancellation_source_frequency") is None:
                            raise LabOneQControllerException(
                                f"{self.dev_repr}: Using the external"
                                f" cancellation source requires specifying the"
                                f" cancellation frequency"
                            )
                if value is None or key in DeviceSHFPPC.attribute_keys:
                    # Skip not set values, or values that are bound to sweep params and will
                    # be set during the NT execution.
                    continue
                nc.add(self._key_to_path(key, ch), _convert(value))
        return await self.maybe_async(nc)

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector()
        nc.extend(super().collect_prepare_nt_step_nodes(attributes, recipe_data))
        for ch in range(self._channels):
            for key, attr_name in DeviceSHFPPC.attribute_keys.items():
                [value], updated = attributes.resolve(keys=[(attr_name, ch)])
                if value is not None and key == "cancellation_phase":
                    value *= 180 / math.pi
                if updated:
                    path = self._key_to_path(key, ch)
                    nc.add(path, value)
        return nc
