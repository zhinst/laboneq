# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from laboneq.compiler.common.awg_info import AwgKey
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

    def get_feedback_register_configurations(self, key: AwgKey):
        for output in self.combined_output.values():
            if key in output.feedback_register_configurations:
                return output.feedback_register_configurations[key]

    def add_total_execution_time(self, other):
        for device_class, combined_output in self.combined_output.items():
            combined_output.total_execution_time += other.codegen_output[
                device_class
            ].total_execution_time

    @property
    def total_execution_time(self):
        return max(
            [c.total_execution_time for c in self.combined_output.values()], default=0.0
        )

    @property
    def max_execution_time_per_step(self):
        return max(
            [c.max_execution_time_per_step for c in self.combined_output.values()],
            default=0,
        )
