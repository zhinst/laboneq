# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.recipe_processor import RecipeData


class ChannelBase(ABC):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
    ):
        self._api = api
        self._subscriber = subscriber
        self._device_uid = device_uid
        self._serial = serial
        self._channel = channel

    @property
    def channel(self) -> int:
        return self._channel

    async def disable_output(self, outputs: set[int], invert: bool):
        """Disable the output of the channel if it matches the criteria."""
        if (self._channel in outputs) != invert:
            await self._api.set_parallel(self._disable_output())

    @abstractmethod
    def _disable_output(self) -> NodeCollector:
        """Return node(s) to disable the output of the channel."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def allocate_resources(self):
        """Initialize or reset channel resources in preparation for execution."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def load_awg_program(self):
        """Load an AWG program into the channel's AWG."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        """Set the nodes for a single near-time step."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        """Collect warning nodes from the channel."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def start_execution(self, with_pipeliner: bool):
        """Start the execution of the channel's sequencer program."""
        raise NotImplementedError("This method should be implemented by subclasses.")
