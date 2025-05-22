# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceUsage:
    desc: str
    """A descriptor of the resource, e.g. command table of device this, AWG core that."""
    usage: float
    """Usage, as in used / max available."""


class ResourceLimitationError(Exception):
    def __init__(self, message: str, hint: float | None = None):
        super().__init__(message)
        self.hint = hint
        """Represents the usage of the said resource."""
