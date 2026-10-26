# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from laboneq.controller.utilities.exception import LabOneQControllerException

if TYPE_CHECKING:
    import numpy.typing as npt

    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.dsl.experiment.pulse import Pulse


class NearTimeReplacements:
    def __init__(self):
        # pulse_uid: replacement
        self._pulses: dict[str, npt.ArrayLike | "Pulse"] = {}
        self._pulses_updated_after_last_read: bool = False

        # param_uid: replacement
        self._phase_increments: dict[str, int | float] = {}
        self._phase_increments_updated_after_last_read: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self._pulses) == 0 and len(self._phase_increments) == 0

    @property
    def pulses_updated_after_last_read(self) -> bool:
        return self._pulses_updated_after_last_read

    @property
    def phase_increments_updated_after_last_read(self) -> bool:
        return self._phase_increments_updated_after_last_read

    @contextmanager
    def pulses(self):
        yield self._pulses
        self._pulses_updated_after_last_read = False

    @contextmanager
    def phase_increments(self):
        yield self._phase_increments
        self._phase_increments_updated_after_last_read = False

    def add_pulse_replacement(
        self, pulse_uid: str, replacement: npt.ArrayLike | "Pulse"
    ):
        self._pulses[pulse_uid] = replacement
        self._pulses_updated_after_last_read = True

    def add_phase_increment_replacement(self, param_uid: str, value: int | float):
        self._phase_increments[param_uid] = value
        self._phase_increments_updated_after_last_read = True

    def clear(self):
        self._pulses.clear()
        self._phase_increments.clear()
        self._pulses_updated_after_last_read = False
        self._phase_increments_updated_after_last_read = False


def check_replacements(
    replacements: NearTimeReplacements, recipe_data: RecipeData
) -> bool:
    """Returns whether there are replacements to apply.

    Raises if there are replacements and the experiment is chunked, because
    replacements do not support chunking.
    """
    if replacements.is_empty:
        return False
    if recipe_data.rt_execution_info.is_chunked:
        raise LabOneQControllerException(
            "Cannot apply near-time artifact replacements for chunked experiment."
        )
    return True
