# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator


class SweepParamsTracker:
    def __init__(self):
        self.sweep_param_values: dict[str, float] = {}
        self.sweep_param_updates: set[str] = set()

    def set_param(self, param: str, value: float):
        self.sweep_param_values[param] = value
        self.sweep_param_updates.add(param)

    def updated_params(self) -> Iterator[tuple[str, float]]:
        for param in self.sweep_param_updates:
            yield param, self.sweep_param_values[param]

    def clear_for_next_step(self):
        self.sweep_param_updates.clear()

    def get_param(self, param: str) -> float:
        return self.sweep_param_values[param]
