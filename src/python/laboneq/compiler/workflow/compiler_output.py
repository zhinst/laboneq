# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from laboneq.compiler.common.iface_compiler_output import (
    CombinedOutput,
)


@dataclass
class CombinedRTCompilerOutputContainer:
    """Container for the compiler artifacts, after linking."""

    device_class: int
    combined_output: CombinedOutput
    schedule: dict[str, Any] | None = None

    def get_artifacts(self):
        return self.combined_output.get_artifacts()

    def get_first_combined_output(self) -> CombinedOutput | None:
        """Get the first combined output, if available."""
        return self.combined_output if self.combined_output is not None else None
