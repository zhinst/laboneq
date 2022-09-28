# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from typing import List, Dict, Any
from numpy import typing as npt
from weakref import ref, ReferenceType

from laboneq.controller.communication import DaqNodeAction
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
)
from laboneq.controller.recipe_1_4_0 import (
    Initialization,
    IntegratorAllocation,
    OscillatorParam,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode


@dataclass
class DeviceQualifier:
    dry_run: bool = True
    driver: str = None
    server: Any = None  # Actual type/value depends on driver
    options: Dict[str, str] = None


class DeviceBase(ABC):
    def __init__(self, device_qualifier: DeviceQualifier):
        self._device_qualifier: DeviceQualifier = device_qualifier
        self._logger = logging.getLogger(__name__)
        self._downlinks: Dict[str, ReferenceType[DeviceBase]] = {}
        self._uplinks: Dict[str, ReferenceType[DeviceBase]] = {}

    @property
    def device_qualifier(self):
        return self._device_qualifier

    @property
    def dry_run(self):
        return self._device_qualifier.dry_run

    @property
    def parameters(self):
        return []

    @property
    def has_awg(self) -> bool:
        return False

    @property
    def dev_repr(self) -> str:
        return self._device_qualifier.driver

    @abstractmethod
    def connect(self):
        pass

    def _get_option(self, key):
        return self._device_qualifier.options.get(key)

    def _warn_for_unsupported_param(self, param_assert, param_name, channel):
        if not param_assert:
            channel_clause = (
                "" if channel is None else f" specified for the channel {channel}"
            )
            self._logger.warning(
                "%s: parameter '%s'%s is not supported on this device type.",
                self.dev_repr,
                param_name,
                channel_clause,
            )

    def add_downlink(self, port: str, linked_device: "DeviceBase"):
        self._downlinks[port] = ref(linked_device)

    def add_uplink(self, port: str, linked_device: "DeviceBase"):
        self._uplinks[port] = ref(linked_device)

    def remove_all_links(self):
        self._downlinks.clear()
        self._uplinks.clear()

    def is_leader(self):
        # Check also downlinks, to exclude standalone devices
        return len(self._uplinks) == 0 and len(self._downlinks) > 0

    def is_follower(self):
        # Treat standalone devices as followes
        return (
            len(self._uplinks) > 0
            or len(self._uplinks) == 0
            and len(self._downlinks) == 0
        )

    def free_allocations(self):
        pass

    def allocate_osc(self, osc_param: OscillatorParam.Data):
        raise LabOneQControllerException(
            f"{self.dev_repr}: this device type doesn't support HW oscillators."
        )

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> List[DaqNodeAction]:
        return []

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: List[int],
        num_results: int,
        hw_averages: int,
    ):
        return None  # default -> no results available from the device

    def get_input_monitor_data(self, channel: int, num_results: int):
        return None  # default -> no results available from the device

    @abstractmethod
    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        pass

    @abstractmethod
    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ):
        pass

    def collect_osc_initialization_nodes(self) -> List[DaqNodeAction]:
        return []

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        return []

    @abstractmethod
    def upload_awg_program(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        pass

    @abstractmethod
    def collect_awg_after_upload_nodes(self, initialization: Initialization.Data):
        pass

    @abstractmethod
    def collect_prepare_sweep_step_nodes_for_param(self, param: str, value: float):
        pass

    @abstractmethod
    def collect_trigger_configuration_nodes(self, initialization):
        pass

    @abstractmethod
    def collect_execution_nodes(self):
        pass

    def wait_for_conditions_to_start(self):
        pass

    def wait_for_execution_ready(self):
        pass

    @abstractmethod
    def collect_conditions_to_close_loop(self, acquisition_units):
        pass

    @abstractmethod
    def configure_as_leader(self, initialization):
        pass

    @abstractmethod
    def collect_follower_configuration_nodes(self, initialization):
        pass

    @abstractmethod
    def check_errors(self):
        pass

    @abstractmethod
    def collect_reset_nodes(self):
        return []

    def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        pass

    def shut_down(self):
        pass

    def disconnect(self):
        pass
