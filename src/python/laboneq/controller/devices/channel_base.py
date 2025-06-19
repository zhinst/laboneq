# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod


class ChannelBase(ABC):
    @abstractmethod
    def allocate_resources(self):
        """Initialize or reset channel resources in preparation for execution."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    async def load_awg_program(self):
        """Load an AWG program into the channel's AWG."""
        raise NotImplementedError("This method should be implemented by subclasses.")
