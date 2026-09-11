# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class LogicalSignal:
    """A logical signal of a device setup, referenced by group and name."""

    name: str
    group: str

    def path(self) -> str:
        return f"{self.group}/{self.name}"
