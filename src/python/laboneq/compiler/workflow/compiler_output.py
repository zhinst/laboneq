# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from laboneq.compiler.common.iface_compiler_output import (
    CombinedOutput,
)


@dataclass
class CombinedRTCompilerOutputContainer:
    """Container (by device class) for the compiler artifacts, after linking."""

    combined_output: dict[int, CombinedOutput]
    schedule: dict[str, Any] = field(default_factory=dict)

    def get_artifacts(self):
        if len(self.combined_output) == 1:
            return next(iter(self.combined_output.values())).get_artifacts()
        else:
            return {
                device_class: combined_output.get_artifacts()
                for device_class, combined_output in self.combined_output.items()
            }
