# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.communication import DaqNodeAction, DaqNodeSetAction
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.data.recipe import Initialization


class DeviceSHFPPC(DeviceZI):
    attribute_keys = {
        "pump_freq": AttributeName.PPC_PUMP_FREQ,
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
            "_on": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/on",
            "pump_freq": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/freq",
            "pump_power": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/power",
            "cancellation": f"/{self.serial}/ppchannels/{ch}/cancellation/on",
            "alc_engaged": f"/{self.serial}/ppchannels/{ch}/synthesizer/pump/alc",
            "use_probe": f"/{self.serial}/ppchannels/{ch}/synthesizer/probe/on",
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

    def check_errors(self):
        pass

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        return []

    def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeAction]:
        nodes_to_set: list[DaqNodeAction] = []
        ppchannels = initialization.ppchannels or []

        def _convert(value):
            if isinstance(value, bool):
                return 1 if value else 0
            return value

        for settings in ppchannels:
            ch = settings["channel"]
            nodes_to_set.append(
                DaqNodeSetAction(self._daq, self._key_to_path("_on", ch), 1)
            )
            for key, value in settings.items():
                if value is None or key in [*DeviceSHFPPC.attribute_keys, "channel"]:
                    # Skip not set values, or values that are bound to sweep params and will
                    # be set during the NT execution.
                    continue
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq, self._key_to_path(key, ch), _convert(value)
                    )
                )
        return nodes_to_set

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        nodes_to_set = super().collect_prepare_nt_step_nodes(attributes, recipe_data)
        for ch in range(self._channels):
            for key, attr_name in DeviceSHFPPC.attribute_keys.items():
                [value], updated = attributes.resolve(keys=[(attr_name, ch)])
                if updated:
                    path = self._key_to_path(key, ch)
                    nodes_to_set.append(DaqNodeSetAction(self._daq, path, value))
        return nodes_to_set
