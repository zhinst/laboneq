# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy import typing as npt

from laboneq.controller.runtime_context import RuntimeContext
from laboneq.core.exceptions import AbortExecution

if TYPE_CHECKING:
    from laboneq.controller.controller import Controller
    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.controller.toolkit_adapter import ToolkitDevices
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.result.results import Results


class ConnectionState:
    """Session connection state.

    Attributes:
        connected (bool):
            True if the session is connected to instruments.
            False otherwise.
        emulated (bool):
            True if the session is running in emulation mode.
            False otherwise.

        Used for testing purposes only in LabOneQ applications.
        TODO: Remove when the example has been fixed.
    """

    connected: bool = True
    emulated: bool = False


@dataclass
class LegacySessionData:
    """Data from the legacy session to be passed to the RuntimeContextImpl for backwards compatibility.

    TODO: Remove when the deprecated properties have been removed.
    """

    experiment: Any
    experiment_calibration: Any
    signal_map: Any
    device_setup: Any
    device_calibration: Any


class RuntimeContextImpl(RuntimeContext):
    def __init__(
        self,
        controller: Controller,
        recipe_data: RecipeData,
        experiment_results: ExperimentResults,
        devices: ToolkitDevices,
        do_emulation: bool,
        legacy_session_data: LegacySessionData,
    ):
        self._devices = devices
        self._controller = controller
        self._recipe_data = recipe_data
        self._experiment_results = experiment_results
        self._do_emulation = do_emulation
        self._legacy_session_data = legacy_session_data

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

    def abort_execution(self):
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

    @property
    def devices(self) -> ToolkitDevices:
        return self._devices

    @property
    def emulated(self) -> bool:
        return self._do_emulation

    @property
    def experiment(self):
        """
        Object holding the experiment definition.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0. Please
        use function arguments of the neartime callback instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )
        return self._legacy_session_data.experiment

    @property
    def experiment_calibration(self):
        """
        Object holding the experiment calibration.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0. Please
        use function arguments of the neartime callback instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )
        return self._legacy_session_data.experiment_calibration

    @property
    def signal_map(self):
        """
        Dict holding the signal mapping.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0. Please
        use function arguments of the neartime callback instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )
        return self._legacy_session_data.signal_map

    @property
    def device_setup(self):
        """
        Object holding the device setup of the QCCS system.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0. Please
        use function arguments of the neartime callback instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )
        return self._legacy_session_data.device_setup

    @property
    def device_calibration(self):
        """
        Object holding the calibration of the device setup.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0. Please
        use function arguments of the neartime callback instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )
        return self._legacy_session_data.device_calibration

    @property
    def connection_state(self) -> ConnectionState:
        """
        Connection state.

        Used for testing purposes only in LabOneQ applications.

        TODO: Remove when the example has been fixed.

        !!! version-changed "Deprecated in version 26.4.0"
        This property will be removed in version 26.7.0.
        Please use `RuntimeContext.emulated` instead.
        """
        warnings.warn(
            "Deprecated, please use function arguments of the neartime callback.",
            FutureWarning,
            stacklevel=2,
        )

        connection_state = ConnectionState()
        connection_state.emulated = self._do_emulation
        return connection_state
