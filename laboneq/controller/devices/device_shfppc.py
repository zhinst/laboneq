# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from laboneq.controller.communication import DaqNodeAction, DaqNodeSetAction
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_1_4_0 import Initialization
from laboneq.controller.recipe_processor import DeviceRecipeData
from laboneq.controller.util import SweepParamsTracker


class DeviceSHFPPC(DeviceZI):
    param_keys = ["pump_freq", "pump_power", "probe_frequency", "probe_power"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFPPC"
        self.dev_opts = []
        self._use_internal_clock = False
        self._param_to_paths: dict[str, list[str]] = defaultdict(
            list
        )  # TODO(2K): Move to DeviceZI

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

    def allocate_params(self, initialization: Initialization.Data):
        ppchannels = initialization.ppchannels or {}
        for ch, settings in ppchannels.items():
            for key in DeviceSHFPPC.param_keys:
                setting = settings.get(key)
                if isinstance(setting, str):
                    self._param_to_paths[setting].append(self._key_to_path(key, ch))

    def check_errors(self):
        pass

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        return []

    def collect_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> list[DaqNodeAction]:
        nodes_to_set: list[DaqNodeAction] = []
        ppchannels = initialization.ppchannels or {}

        def _convert(value):
            if isinstance(value, bool):
                return 1 if value else 0
            return value

        for ch, settings in ppchannels.items():
            nodes_to_set.append(
                DaqNodeSetAction(self._daq, self._key_to_path("_on", ch), 1)
            )
            for key, value in settings.items():
                if (
                    value is None
                    or key in DeviceSHFPPC.param_keys
                    and isinstance(value, str)
                ):
                    # Skip not set values, or values that are bound to sweep params and will
                    # be set during the NT execution.
                    continue
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq, self._key_to_path(key, ch), _convert(value)
                    )
                )
        return nodes_to_set

    def collect_prepare_sweep_step_nodes_for_param(
        self, sweep_params_tracker: SweepParamsTracker
    ) -> list[DaqNodeAction]:
        nodes_to_set: list[DaqNodeAction] = []
        for param in sweep_params_tracker.updated_params():
            for path in self._param_to_paths.get(param, []):
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq, path, sweep_params_tracker.get_param(param)
                    )
                )
        return nodes_to_set
