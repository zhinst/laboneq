# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from numpy import typing as npt

from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.simple_proxy import SimpleProxy
from laboneq.core.exceptions import AbortExecution

if TYPE_CHECKING:
    from laboneq.dsl.result.results import Results
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.controller.controller import Controller
    from laboneq.controller.recipe_processor import RecipeData


class ProtectedSession(SimpleProxy):
    def __init__(
        self,
        wrapped_session: Any,
        controller: Controller,
        recipe_data: RecipeData,
        experiment_results: ExperimentResults,
    ):
        super().__init__(wrapped_session)
        self._controller = controller
        self._recipe_data = recipe_data
        self._experiment_results = experiment_results

    # Backwards compatibility after migration to the new architecture
    @property
    def results(self) -> Results:
        return self._last_results

    # Backwards compatibility after migration to the new architecture
    def get_results(self) -> Results:
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

    def disconnect(self):
        raise LabOneQControllerException(
            "'disconnect' is not allowed from the near-time callback."
        )

    def abort_execution(self):
        """Abort the execution of an experiment.

        Note: This currently exclusively works when called from within a near-time callback."""
        raise AbortExecution

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        self._controller.replace_pulse(
            recipe_data=self._recipe_data,
            pulse_uid=pulse_uid,
            pulse_or_array=pulse_or_array,
        )

    def replace_phase_increment(
        self,
        parameter_uid: str,
        new_value: int | float,
    ):
        self._controller.replace_phase_increment(
            recipe_data=self._recipe_data,
            parameter_uid=parameter_uid,
            new_value=new_value,
        )
