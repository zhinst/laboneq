# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, NoReturn, Union

from laboneq.data.experiment_results import ExperimentResults
from numpy import typing as npt

from laboneq.controller.toolkit_adapter import ToolkitDevices
from laboneq.core.exceptions import AbortExecution, LabOneQException
from laboneq.core.types import CompiledExperiment
from laboneq.core.utilities.environment import is_testing
from laboneq.core.utilities.laboneq_compile import laboneq_compile
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.device.io_units.logical_signal import (
    LogicalSignalRef,
    resolve_logical_signal_ref,
)
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.result import Results
from laboneq.implementation.legacy_adapters.converters_target_setup import (
    convert_dsl_to_target_setup,
)
from laboneq.laboneq_logging import initialize_logging

from laboneq.controller import Controller

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


_logger = logging.getLogger(__name__)


class ConnectionState:
    """Session connection state.

    Attributes:
        connected (bool):
            True if the session is connected to instruments.
            False otherwise.
        emulated (bool):
            True if the session is running in emulation mode.
            False otherwise.
    """

    connected: bool = False
    emulated: bool = False


_FLEXIBLE_FEEDBACK_SETTING = "FLEXIBLE_FEEDBACK"


def _requires_neartime_callback() -> NoReturn:
    raise LabOneQException(
        "This method only works when called from a near-time callback."
    )


class Session:
    """This Session class represents the main endpoint for the user interaction with the QCCS system.

    The session holds:

    * the wiring definition of the devices
    * the experiment definition that should be run on the devices
    * the calibration of the devices for experiment
    * the compiled experiment
    * the result of the executed experiment

    The Session is a stateful object that hold all of the above.
    The expected steps to interact with the session are:

    * initial state (construction)
    * setting the device setup (optionally during construction)
    * (optional) setting the calibration of the devices
    * connecting to the devices (or the emulator)
    * compiling the experiment
    * running the experiment
    * accessing the results of the last run experiment

    The session is serializable in every state.
    """

    def __init__(
        self,
        device_setup: DeviceSetup | None = None,
        log_level: int | str | None = None,
        performance_log: bool = False,
        configure_logging: bool = True,
        _last_results=None,
        compiled_experiment: CompiledExperiment | None = None,
        experiment: Experiment | None = None,
        include_results_metadata: bool = False,
        server_log: bool = False,
    ):
        """Constructor of the session.

        Args:
            device_setup: Device setup that should be used for this session.
                The device setup can also be passed to the session after the construction
                of the object.
            log_level: Log level of the session.
                If no log level is specified, the session will use the logging.INFO level.
                Other possible levels refer to the logging python package and
                `laboneq.laboneq_logging`.
            performance_log: Flag to enable performance logging.
                When True, the system creates a separate logfile containing logs aimed to analyze system performance.
            configure_logging:
                Whether to configure logger. Can be disabled for custom logging use cases.
            compiled_experiment:
                If specified, set the current compiled experiment.
            experiment:
                If specified, set the current experiment.
            include_results_metadata:
                If True, `Session.run` will return a `Results` object with the deprecated `.experiment`,
                and `.device_setup` attributes populated. Otherwise, it will
                return a `Results` object with these attributes not populated.
            server_log:
                If `True`, the data server log - including device firmware logs - will be forwarded to the LabOneQ
                log under the logger named `server.log.<server_uid>`. Additionally, it will be written to the file
                `server.log` alongside the regular LabOneQ log, assuming the standard logging configuration is used.

        !!! version-removed "Removed in version 2.57.0"
            Removed the `register_user_function` method that was deprecated in 2.19.0.
            Use `register_neartime_callback` instead.

        !!! version-changed "Changed in version 2.55.0"
            The deprecated `.compiled_experiment` attribute was removed from `Results`. The
            `include_results_metadata` argument thus no longer populates this attribute on `Results`.
            Track the compiled experiment separately instead.

        !!! version-changed "Changed in version 2.54.0"
            The following deprecated methods for saving and loading were removed:
                - `load`
                - `save`
                - `save_signal_map`
                - `load_signal_map`
                - `save_results`
                - `save_experiment`
                - `load_experiment`
                - `save_device_setup`
                - `load_device_setup`
                - `save_device_calibration`
                - `load_device_calibration`
                - `save_compiled_experiment`
                - `load_compiled_experiment`
                - `save_experiment_calibration`
                - `load_experiment_calibration`
            Use the `load` and `save` functions from the `laboneq.simple` module instead.

        !!! version-added "Added in version 2.52.0"
            Added the `include_results_metadata` argument.
        """
        self._device_setup = device_setup if device_setup else DeviceSetup()
        self._controller: Controller | None = None
        self._connection_state: ConnectionState = ConnectionState()
        self._experiment_definition = experiment
        self._compiled_experiment = compiled_experiment
        self._last_results = _last_results
        self._include_results_metadata = include_results_metadata
        if configure_logging:
            if not is_testing():
                # Only initialize logging outside pytest
                # pytest initializes the logging itself
                initialize_logging(
                    log_level=log_level,
                    performance_log=performance_log,
                    server_log=server_log,
                )
            self._logger = logging.getLogger("laboneq")
        else:
            self._logger = logging.getLogger("null")
        self._neartime_callbacks: Dict[str, Callable] = {}
        self._toolkit_devices = ToolkitDevices()

    @property
    def devices(self) -> ToolkitDevices:
        """Connected devices included in the system setup.

        Allows the modification/inspection of the state of the device and its nodes.

        Devices exist once the session is connected. After disconnecting, devices
        are empty.

        Usage:

        ``` pycon
            >>> session.connect()
            >>> session.devices["device_hdawg"].awgs[0].outputs[0].amplitude(1)
            >>> session.devices["DEV1234"].awgs[0].outputs[0].amplitude()
            1
        ```
        """
        return self._toolkit_devices

    def __eq__(self, other):
        if not isinstance(other, Session):
            return False
        return self is other or (
            self._device_setup == other._device_setup
            and self._experiment_definition == other._experiment_definition
            and self._compiled_experiment == other._compiled_experiment
            and self._last_results == other._last_results
            and self._neartime_callbacks == other._neartime_callbacks
        )

    def _assert_connected(self) -> Controller:
        """Verifies that the session is connected to the devices."""
        if self._connection_state.connected and self._controller is not None:
            return self._controller
        raise LabOneQException(
            "Session not connected.\n"
            "The call requires an established connection to devices in order to execute the experiment.\n"
            "Call connect() first. Use connect(do_emulation=True) if you want to emulate the devices' behavior only."
        )

    def register_neartime_callback(self, func, name: str | None = None):
        """Registers a near-time callback to be referred from the experiment's `call` operation.

        Args:
            func (function): Near-time callback that is registered.
            name (str):     Optional name to use as the argument to experiment's `call` operation to refer to this
                            function. If not provided, function name will be used.
        """

        if name is None:
            name = func.__name__
        self._neartime_callbacks[name] = func

    def connect(
        self,
        do_emulation=False,
        ignore_version_mismatch=False,
        reset_devices=False,
        use_async_api: bool | None = None,
        disable_runtime_checks: bool = True,
        timeout: float | None = None,
    ) -> ConnectionState:
        """Connects the session to the QCCS system.

        Args:
            do_emulation (bool): Specifies if the session should connect to a emulator
                                 (in the case of 'True') or the real system (in the case of 'False').

            ignore_version_mismatch (bool): Ignore version mismatches.
                If set to `False` (default), the following checks are made for compatibility:

                - Check LabOne and LabOne Q version compatibility.
                - Check LabOne and Zurich Instruments' devices firmware version compatibility.

                The following states raise an exception:

                - Device firmware requires an update
                - Device firmware requires an downgrade
                - Device update is in progress

                It is suggested to keep the versions aligned and up-to-date to avoid any unexpected behaviour.

            reset_devices (bool): Load the factory preset after connecting for device which support it.

            use_async_api (bool): Enable the async backend of LabOne Q controller. Defaults to `True`.

            disable_runtime_checks (bool): Disable the runtime checks performed
                by device firmware. Defaults to `True`.

            timeout (float): Specifies the timeout for the initial connection to the instrument in seconds.

        Returns:
            connection_state:
                The connection state of the session.
        """
        if use_async_api is not None:
            warnings.warn(
                "The 'use_async_api' argument currently has no effect and "
                "will be removed in version 2.53.0. Please adjust your code to not supply this "
                "argument.",
                FutureWarning,
                stacklevel=2,
            )

        self._ignore_version_mismatch = ignore_version_mismatch
        if (
            self._connection_state.connected
            and self._connection_state.emulated != do_emulation
        ):
            self.disconnect()
        self._connection_state.emulated = do_emulation
        target_setup = convert_dsl_to_target_setup(self._device_setup)

        controller = Controller(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=self._neartime_callbacks,
            parent_session=self,
        )
        controller.connect(
            do_emulation=self._connection_state.emulated,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout,
        )
        self._controller = controller
        if self._connection_state.emulated:
            self._toolkit_devices = ToolkitDevices()
        else:
            self._toolkit_devices = ToolkitDevices(controller.devices)
        self._connection_state.connected = True
        return self._connection_state

    def disconnect(self) -> ConnectionState:
        """Disconnects instruments from the data server and closes the connection for this session.

        Returns:
            connection_state:
                The connection state of the session.
        """
        self._connection_state.connected = False
        if self._controller is not None:
            self._controller.disconnect()
        self._controller = None
        self._toolkit_devices = ToolkitDevices()
        return self._connection_state

    def disable_outputs(
        self,
        devices: str | list[str] | None = None,
        signals: LogicalSignalRef | list[LogicalSignalRef] | None = None,
        unused_only: bool = False,
    ):
        """Turns off / disables the device outputs.

        Args:
            devices:
                Optional. Device or list of devices, if not specified - all devices.
                All or unused (see 'unused_only') outputs of these devices will be
                disabled. Can't be used together with 'signals'.
            signals:
                Optional. Logical signal or a list of logical signals. Outputs mapped
                by these logical signals will be disabled. Can't be used together
                with 'devices' or 'unused_only'.
            unused_only:
                Optional. If set to True, only outputs not mapped by any logical
                signals will be disabled. Can't be used together with 'signals'.
        """
        controller = self._assert_connected()
        if devices is not None and signals is not None:
            raise LabOneQException(
                "Ambiguous outputs specification: disable_outputs() accepts either 'devices' or "
                "'signals', but not both."
            )
        if unused_only and signals is not None:
            raise LabOneQException(
                "Ambiguous outputs specification: disable_outputs() accepts either 'signals' or "
                "'unused_only=True', but not both."
            )
        if devices is not None and not isinstance(devices, list):
            devices = [devices]
        if signals is not None and not isinstance(signals, list):
            signals = [signals]
        logical_signals = (
            None
            if signals is None
            else [resolve_logical_signal_ref(s) for s in signals]
        )
        controller.disable_outputs(devices, logical_signals, unused_only)

    @property
    def connection_state(self) -> ConnectionState:
        """Session connection state."""
        return self._connection_state

    def compile(
        self,
        experiment: Experiment,
        compiler_settings: Dict | None = None,
    ) -> CompiledExperiment:
        """Compiles the specified experiment and stores it in the compiled_experiment property.

        Requires connected LabOne Q session (`session.connect()`) either with or without emulation mode.

        Args:
            experiment: Experiment instance that should be compiled.
            compiler_settings: Extra options passed to the compiler.
        """
        self._experiment_definition = experiment
        self._compiled_experiment = laboneq_compile(
            device_setup=self.device_setup,
            experiment=self.experiment,
            compiler_settings={
                **(compiler_settings or {}),
            },
        )
        self._last_results = None
        return self._compiled_experiment

    @property
    def compiled_experiment(self) -> CompiledExperiment | None:
        """Access to the compiled experiment.

        The compiled experiment can be assigned to a different session if the device setup is matching.
        """
        return self._compiled_experiment

    def run(
        self,
        experiment: Union[Experiment, CompiledExperiment] | None = None,
        include_results_metadata: bool | None = None,
    ) -> Results:
        """Executes the compiled experiment.

        Requires connected LabOne Q session (`session.connect()`) either with or without emulation mode.

        If no experiment is specified, the last compiled experiment is run.
        If an experiment is specified, the provided experiment is assigned to the
        internal experiment of the session.

        Args:
            experiment: Optional. Experiment instance that should be
                run. The experiment will be compiled if it has not been yet. If no
                experiment is specified the previously assigned and compiled experiment
                is used.
            include_results_metadata:
                If true, return a `Results` object with the deprecated `.experiment`,
                and `.device_setup` attributes populated.
                If false, return a `Results` object with these attributes not populated.
                If None, the setting falls back to that passed to `include_results_metadata`
                when this session was created.

        Returns:
            results:
                A `Results` object.

        Raises:
            LabOneQException:
                If the session is not connected.
            LabOneQControllerException:
                If errors are reported by the controller while executing the experiment.

        !!! version-changed "Changed in version 2.55.0"
            The deprecated `.compiled_experiment` attribute was removed from `Results`. The
            `include_results_metadata` argument thus no longer populates this attribute on `Results`.
            Track the compiled experiment separately instead.

        !!! version-changed "Changed in version 2.52.0"
            Replaced the `include_metadata` argument with `include_results_metadata`.

        !!! version-added "Added in version 2.51.0"
            Added the `include_metadata` argument to control whether to include experiment and
            device setup in the results.
        """
        if include_results_metadata is None:
            include_results_metadata = self._include_results_metadata

        controller = self._assert_connected()
        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
            else:
                self.compile(experiment)
        if self.compiled_experiment is None:
            raise LabOneQException("No experiment available to run.")

        self._last_results = None
        handle = None
        try:
            handle = controller.submit_compiled(
                self.compiled_experiment.scheduled_experiment
            )
            controller.wait_submission(handle)
            controller.stop_workers()
        finally:
            results = (
                ExperimentResults()
                if handle is None
                else controller.submission_results(handle)
            )
            results_kwargs: dict[str, Any] = {}
            if include_results_metadata:
                results_kwargs["experiment"] = self.compiled_experiment.experiment
                results_kwargs["device_setup"] = self.device_setup
            self._last_results = Results(
                acquired_results=results.acquired_results,
                neartime_callback_results=results.neartime_callback_results,
                execution_errors=results.execution_errors,
                pipeline_jobs_timestamps=results.pipeline_jobs_timestamps,
                **results_kwargs,
            )

        return self.results

    def submit(
        self,
        experiment: Experiment | CompiledExperiment | None = None,
        queue: Callable[[str, CompiledExperiment | None, DeviceSetup], Any]
        | None = None,
    ) -> Results:
        """Asynchronously submit experiment to the given queue.

        If no experiment is specified, the last compiled experiment is run.
        If an experiment is specified, the provided experiment is assigned to the
        internal experiment of the session.

        Args:
            experiment: Optional. Experiment instance that should be
                run. The experiment will be compiled if it has not been yet. If no
                experiment is specified the previously assigned and compiled experiment
                is used.
            queue: The name of connector to a queueing system which should do the actual
                run on a setup. `queue` must be callable with the signature
                ``(name: str, experiment: CompiledExperiment | None, device_setup: DeviceSetup)``
                which returns an object with which users can query results.

        Returns:
            results:
                An object with which users can query results. Details depend on the
                implementation of the queue.
        """
        if queue is None:
            raise LabOneQException(
                "The 'queue' parameter must be provided and cannot be None."
            )
        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
                return queue(
                    experiment.experiment.uid,
                    self.compiled_experiment,
                    self.device_setup,
                )
            else:
                self._assert_connected()
                self.compile(experiment)
                return queue(
                    experiment.uid, self.compiled_experiment, self.device_setup
                )
        else:
            return queue("", self.compiled_experiment, self.device_setup)

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        """
        Replaces a specific pulse with new sample data on the device.

        This is useful when called from within a near-time callback, and allows fast
        waveform replacement within near-time loops without recompilation of the experiment.

        Args:
            pulse_uid: Pulse to replace, can be a Pulse object or the UID of the pulse.
            pulse_or_array:
                Replacement pulse, can be a Pulse object or array of values.
                Needs to have the same length as the pulse it replaces.
        """
        _requires_neartime_callback()

    def replace_phase_increment(
        self,
        parameter_uid: str,
        new_value: int | float,
    ):
        """Replace the value of a parameter that drives phase increments value.

        If the parameter spans multiple iterations of a loop, it will replace the
        parameter by the same value in _all_ the iterations.


        Args:
            parameter_uid: The name of the parameter to replace.
            new_value: The new replacement value.

        """
        _requires_neartime_callback()

    def get_results(self) -> Results:
        """
        Returns a deep copy of the result of the last experiment execution.

        Raises an exception if no experiment results are available.

        Returns:
            results:
                A deep copy of the results of the last experiment.
        """
        if not self._last_results:
            raise LabOneQException(
                "No results available. Execute run() or simulate_outputs() in order to generate an experiment's result."
            )
        return deepcopy(self._last_results)

    @property
    def results(self) -> Results:
        """
        Object holding the result of the last experiment execution.

        !!! Attention
            This accessor is provided for better
            performance, unlike `get_result` it doesn't make a copy, but instead returns the reference to the live
            result object being updated during the session run. Care must be taken for not modifying this object from
            the user code, otherwise behavior is undefined.
        """
        return self._last_results

    @property
    def experiment(self):
        """
        Object holding the experiment definition.
        """
        return self._experiment_definition

    @property
    def experiment_calibration(self):
        """
        Object holding the calibration of the experiment.
        """
        return self._experiment_definition.get_calibration()

    @experiment_calibration.setter
    def experiment_calibration(self, value):
        """
        Sets the calibration of the experiment.
        """
        self._experiment_definition.set_calibration(value)

    @property
    def signal_map(self):
        """
        Dict holding the signal mapping.
        """
        return self._experiment_definition.get_signal_map()

    @signal_map.setter
    def signal_map(self, value):
        """
        Sets the signal mapping.
        """
        self._experiment_definition.set_signal_map(value)

    @property
    def device_setup(self):
        """
        Object holding the device setup of the QCCS system.
        """
        return self._device_setup

    @property
    def device_calibration(self):
        """
        Object holding the calibration of the device setup.
        """
        return self._device_setup.get_calibration()

    @device_calibration.setter
    def device_calibration(self, value):
        """
        Sets the calibration of the device setup.
        """
        self._device_setup.set_calibration(value)

    @property
    def log_level(self) -> int:
        """The current log level."""
        return self._logger.level

    @log_level.setter
    def log_level(self, value: int):
        self._logger.setLevel(value)

    @property
    def logger(self):
        """The current logger instance used by the session."""
        return self._logger

    @logger.setter
    def logger(self, logger):
        """
        Sets the logger instance of the session.
        """
        self._logger = logger

    @staticmethod
    def _session_fields():
        return {
            "compiled_experiment": CompiledExperiment,
            "device_setup": DeviceSetup,
            "experiment": Experiment,
            "_last_results": Results,
        }

    def abort_execution(self):
        """Abort the execution of an experiment.

        !!! note

            This currently exclusively works when called from within a near-time callback.
            The function does not return, and instead passes control directly back to the
            LabOne Q runtime.

        """
        raise AbortExecution(
            "Experiment execution can only be aborted from within a near-time callback"
        )
