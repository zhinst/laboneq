# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from laboneq.controller.runtime_context import RuntimeContext
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.exceptions import AbortExecution

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.controller.controller_core import ControllerCore
    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.controller.toolkit_adapter import ToolkitDevices
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.result.results import Results


class RuntimeContextImpl(RuntimeContext):
    def __init__(
        self,
        controller: ControllerCore,
        recipe_data: RecipeData,
        experiment_results: ExperimentResults,
        devices: ToolkitDevices,
        do_emulation: bool,
    ):
        self._devices = devices
        self._controller = controller
        self._recipe_data = recipe_data
        self._experiment_results = experiment_results
        self._do_emulation = do_emulation

    # Backwards compatibility after migration to the new architecture
    @property
    def results(self) -> Results:
        return self._last_results

    @deprecated(
        "`RuntimeContext.get_results()` is deprecated."
        " Use `copy.deepcopy(runtime_context.results)` instead.",
        category=FutureWarning,
    )
    def get_results(self) -> Results:
        """Return a copy of the results of the experiment up to the current point in time.

        !!! version-changed "Deprecated in version 26.10.0"
            Use `copy.deepcopy(runtime_context.results)` instead.
        """
        return copy.deepcopy(self._last_results)

    # Backwards compatibility after migration to the new architecture
    @property
    def _last_results(self) -> Results:
        assert self._experiment_results is not None
        from laboneq.dsl.result.results import Results

        return Results(
            acquired_results=self._experiment_results.acquired_results,
            neartime_callback_results=self._experiment_results.neartime_callback_results,
            execution_errors=self._experiment_results.execution_errors,
        )

    def abort_execution(self):
        raise AbortExecution

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        if isinstance(pulse_uid, str):
            eff_pulse_uid = pulse_uid
        elif hasattr(pulse_uid, "uid"):
            eff_pulse_uid = pulse_uid.uid
        else:
            raise LabOneQControllerException(
                "Cannot replace pulse that does not have uid"
            )

        self._controller.register_pulse_replacement(
            pulse_uid=eff_pulse_uid,
            replacement=pulse_or_array,
        )

    def replace_phase_increment(
        self,
        parameter_uid: str,
        new_value: int | float,
    ):
        self._controller.register_phase_increment_replacement(
            parameter_uid=parameter_uid,
            new_value=new_value,
        )

    @property
    def devices(self) -> ToolkitDevices:
        return self._devices

    @property
    def emulated(self) -> bool:
        return self._do_emulation
