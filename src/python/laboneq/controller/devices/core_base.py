# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.data.recipe import NtStepKey


class CoreBase(ABC):
    def __init__(
        self,
        *,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        core_index: int,
    ):
        self._api = api
        self._subscriber = subscriber
        self._device_uid = device_uid
        self._serial = serial
        self._core_index = core_index

    @abstractmethod
    async def disable_output(self, outputs: set[int], invert: bool):
        """Disable the output(s) of the core if it matches the criteria."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def allocate_resources(self):
        """Initialize or reset core resources in preparation for execution."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def apply_core_initialization(self, device_recipe_data: DeviceRecipeData):
        """Apply initialization settings to the core. Once per experiment."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def load_awg_program(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        """Load an AWG program into the core."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        """Set the nodes for a single near-time step."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        """Collect warning nodes from the core."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        """Return conditions that must be met for execution to be ready."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def start_execution(self, with_pipeliner: bool):
        """Start the execution of the core's sequencer program."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def conditions_for_execution_done(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        """Return conditions that indicate execution is done."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SHFChannelBase(CoreBase):
    @abstractmethod
    async def teardown_one_step_execution(self, with_pipeliner: bool):
        """Tear down the core after a single NT step execution."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class SHFBaseProtocol(Protocol):
    @property
    def uid(self) -> str: ...

    @property
    def serial(self) -> str: ...

    @property
    def dev_repr(self) -> str: ...

    @property
    def _enable_runtime_checks(self) -> bool: ...

    @property
    def _api(self) -> InstrumentConnection: ...

    @property
    def _is_plus(self) -> bool: ...

    def _warn_for_unsupported_param(self, param_assert, param_name, channel): ...
