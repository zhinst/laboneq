# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import typing
from typing import Any

from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.simple_proxy import SimpleProxy
from laboneq.core.exceptions import AbortExecution
from laboneq.data.experiment_results import ExperimentResults

if typing.TYPE_CHECKING:
    from laboneq.dsl.result.results import Results


class ProtectedSession(SimpleProxy):
    def __init__(self, wrapped_session: Any):
        super().__init__(wrapped_session)
        self._experiment_results: ExperimentResults | None = None

    def _set_experiment_results(self, experiment_results: ExperimentResults):
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
