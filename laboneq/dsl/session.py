# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

from numpy import typing as npt

from laboneq._observability.tracing import trace
from laboneq.controller.toolkit_adapter import ToolkitDevices
from laboneq.core.exceptions import AbortExecution, LabOneQException
from laboneq.core.types import CompiledExperiment
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.device.io_units.logical_signal import (
    LogicalSignalRef,
    resolve_logical_signal_ref,
)
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.laboneq_facade import LabOneQFacade
from laboneq.dsl.result import Results
from laboneq.dsl.serialization import Serializer

if TYPE_CHECKING:
    from laboneq.controller import Controller
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
        log_level: int = logging.INFO,
        performance_log: bool = False,
        configure_logging: bool = True,
        _last_results=None,
        compiled_experiment: CompiledExperiment | None = None,
        experiment: Experiment | None = None,
    ):
        """Constructor of the session.

        Args:
            device_setup: Device setup that should be used for this session.
                The device setup can also be passed to the session after the construction
                of the object.
            log_level: Log level of the session.
                If no log level is specified, the session will use the logging.INFO level.
                Other possible levels refer to the logging python package.
            performance_log: Flag to enable performance logging.
                When True, the system creates a separate logfile containing logs aimed to analyze system performance.
            configure_logging:
                Whether to configure logger. Can be disabled for custom logging use cases.
            compiled_experiment:
                If specified, set the current compiled experiment.
            experiment:
                If specified, set the current experiment.

        !!! version-changed "Changed in version 2.0"
            - Removed `pass_v3_to_compiler` argument.
            - Removed `max_simulation_time` instance variable.
        """
        self._device_setup = device_setup if device_setup else DeviceSetup()
        self._controller: Controller = None
        self._connection_state: ConnectionState = ConnectionState()
        self._experiment_definition = experiment
        self._compiled_experiment = compiled_experiment
        self._last_results = _last_results
        if configure_logging:
            LabOneQFacade.init_logging(
                log_level=log_level, performance_log=performance_log
            )
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(log_level)
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

    def _assert_connected(self, fail=True, message=None) -> bool:
        """Verifies that the session is connected to the devices.

        Args:
            fail (bool):    If true, the function will throw an exception in case the session is not
                            connected to the device setup.
            message (str):  Optional message of the optional exception and the internal log in
                            case the session is not connected to the devices.
        """
        if message is None:
            message = ""
        if self._connection_state.connected:
            return True
        default_message = (
            "Session not connected.\n"
            "The call requires an established connection to devices in order to execute the experiment.\n"
            "Call connect() first. Use connect(do_emulation=True) if you want to emulate the devices' behavior only."
        )
        message = message or default_message
        if fail:
            raise LabOneQException(message)
        self.logger.error(message)
        return False

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

    def register_user_function(self, func, name: str | None = None):
        """Registers a near-time callback to be referred from the experiment's `call` operation.

        Args:
            func (function): Near-time callback that is registered.
            name (str):     Optional name to use as the argument to experiment's `call` operation to refer to this
                            function. If not provided, function name will be used.

        !!! version-changed "Deprecated in version 2.19.0"
            The `register_user_function` method was deprecated in version 2.19.0.
            Use `register_neartime_callback` instead.
        """
        warnings.warn(
            "The 'register_user_function' method is deprecated. Use 'register_neartime_callback' instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.register_neartime_callback(func, name)

    @trace("session.connect()")
    def connect(
        self, do_emulation=False, ignore_version_mismatch=False, reset_devices=False
    ) -> ConnectionState:
        """Connects the session to the QCCS system.

        Args:
            do_emulation (bool): Specifies if the session should connect to a emulator
                                 (in the case of 'True') or the real system (in the case of 'False').

            ignore_version_mismatch (bool): Ignore version mismatches.
                If set to `False` (default), the following checks are made for compatibility:

                - Check LabOne and LabOne Q version compatibility.
                - Check LabOne and Zurich Instruments' devices firmware version compatibility.

                The following states raises an exception:

                - Device firmware requires an update
                - Device firmware requires an downgrade
                - Device update is in progress

                It is suggested to keep the versions aligned and up-to-date to avoid any unexpected behaviour.

                !!! version-changed "Changed in version 2.4"
                    Renamed `ignore_lab_one_version_error` to `ignore_version_mismatch` and include
                    LabOne and device firmware version compatibility check.

            reset_devices (bool): Load the factory preset after connecting for device which support it.

        Returns:
            connection_state:
                The connection state of the session.
        """
        self._ignore_version_mismatch = ignore_version_mismatch
        self._reset_devices = reset_devices
        if (
            self._connection_state.connected
            and self._connection_state.emulated != do_emulation
        ):
            self.disconnect()
        self._connection_state.emulated = do_emulation
        LabOneQFacade.connect(self)
        self._connection_state.connected = True
        return self._connection_state

    def disconnect(self) -> ConnectionState:
        """Disconnects the session from the devices.

        Returns:
            connection_state:
                The connection state of the session.
        """
        self._connection_state.connected = False
        LabOneQFacade.disconnect(self)
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
        if signals is not None:
            signals = [resolve_logical_signal_ref(s) for s in signals]
        self._controller.disable_outputs(devices, signals, unused_only)

    @property
    def connection_state(self) -> ConnectionState:
        """Session connection state."""
        return self._connection_state

    @trace("session.compile()")
    def compile(
        self,
        experiment: Experiment,
        compiler_settings: Dict | None = None,
    ) -> CompiledExperiment | None:
        """Compiles the specified experiment and stores it in the compiled_experiment property.

        Requires connected LabOne Q session (`session.connect()`) either with or without emulation mode.

        Args:
            experiment: Experiment instance that should be compiled.
            compiler_settings: Extra options passed to the compiler.

        !!! version-changed "Changed in version 2.4"
            Raises error if `Session` is not connected.

        !!! version-changed "Changed in version 2.0"
            Removed `do_simulation` argument.
            Use [OutputSimulator][laboneq.simulator.output_simulator.OutputSimulator] instead.
        """
        self._assert_connected(fail=True)
        self._experiment_definition = experiment
        self._compiled_experiment = LabOneQFacade.compile(
            self, self.logger, compiler_settings=compiler_settings
        )
        self._last_results = None
        return self._compiled_experiment

    @property
    def compiled_experiment(self) -> CompiledExperiment | None:
        """Access to the compiled experiment.

        The compiled experiment can be assigned to a different session if the device setup is matching.
        """
        return self._compiled_experiment

    @trace("session.run()")
    def run(
        self,
        experiment: Union[Experiment, CompiledExperiment] | None = None,
    ) -> Results | None:
        """Executes the compiled experiment.

        Requires connected LabOne Q session (`session.connect()`) either with or without emulation mode.

        If no experiment is specified, the last compiled experiment is run.
        If an experiment is specified, the provided experiment is assigned to the
        internal experiment of the session.

        !!! version-changed "Changed in version 2.0"
            Removed `do_simulation` argument.
            Use [OutputSimulator][laboneq.simulator.output_simulator.OutputSimulator] instead.

        Args:
            experiment: Optional. Experiment instance that should be
                run. The experiment will be compiled if it has not been yet. If no
                experiment is specified the previously assigned and compiled experiment
                is used.

        Returns:
            results:
                A `Results` object in case of success. `None` if the session is not
                connected.

        !!! version-changed "Changed in version 2.4"
            Raises error if session is not connected.
        """
        self._assert_connected(fail=True)
        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
            else:
                self.compile(experiment)

        self._last_results = Results(
            experiment=self.experiment,
            device_setup=self.device_setup,
            compiled_experiment=self.compiled_experiment,
            acquired_results={},
            neartime_callback_results={},
            execution_errors=[],
        )
        LabOneQFacade.run(self)
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
        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
                return queue(
                    experiment.experiment.uid,
                    self.compiled_experiment,
                    self.device_setup,
                )
            else:
                self._assert_connected(fail=True)
                self.compile(experiment)
                return queue(
                    experiment.uid, self.compiled_experiment, self.device_setup
                )
        else:
            return queue("", self.compiled_experiment, self.device_setup)

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        LabOneQFacade.replace_pulse(self, pulse_uid, pulse_or_array)

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

    @staticmethod
    def load(filename: str) -> Session:
        """Loads the session from a serialized file.
        A restored session from a loaded file will end up in the same state of the session that saved the file in the first place.

        Args:
            filename: Filename (full path) of the file that should be loaded into the session.

        Returns:
            session:
                A new session loaded from the file.
        """

        import json

        with open(filename, mode="r") as file:
            session_dict = json.load(file)
        constructor_args = {}

        for field, field_type in Session._session_fields().items():
            _logger.info("Loading %s of type %s", field, field_type)
            constructor_args[field] = Serializer.load(session_dict[field], field_type)

        return Session(**constructor_args)

    def save(self, filename: str):
        """Stores the session from a serialized file.
        A restored session from a loaded file will end up in the same state of the session that saved the file in the first place.

        Args:
            filename: Filename (full path) of the file where the session should be stored in.
        """
        # TODO ErC: Error handling

        serialized_dict = {}
        for field in Session._session_fields().keys():
            serialized_dict[field] = Serializer.to_dict(getattr(self, field))

        json_string = json.dumps(serialized_dict)
        try:
            with open(filename, mode="w") as file:
                file.write(json_string)
        except IOError as e:
            raise LabOneQException() from e

    def load_device_setup(self, filename: str):
        """Loads a device setup from a given file into the session.

        Args:
            filename: Filename (full path) of the setup should be loaded into the session.
        """
        self._device_setup = DeviceSetup.load(filename)

    def save_device_setup(self, filename: str):
        """Saves the device setup from the session into a given file.

        Args:
            filename: Filename (full path) of the file where the setup should be stored in.
        """
        if self._device_setup is None:
            self.logger.info("No device setup set in this session.")
        else:
            self._device_setup.save(filename)

    def load_device_calibration(self, filename: str):
        """Loads a device calibration from a given file into the session.

        Args:
            filename: Filename (full path) of the calibration should be loaded into the session.
        """
        calibration = Calibration.load(filename)
        self._device_setup.set_calibration(calibration)

    def save_device_calibration(self, filename: str):
        """Saves the device calibration from the session into a given file.

        Args:
            filename: Filename (full path) of the file where the calibration should be stored in.
        """
        if self._device_setup is None:
            self.logger.info("No device setup set in this session.")
        else:
            calibration = self._device_setup.get_calibration()
            calibration.save(filename)

    def load_experiment(self, filename: str):
        """Loads an experiment from a given file into the session.

        Args:
            filename: Filename (full path) of the experiment should be loaded into the session.
        """
        self._experiment_definition = Experiment.load(filename)

    def save_experiment(self, filename: str):
        """Saves the experiment from the session into a given file.

        Args:
            filename: Filename (full path) of the file where the experiment should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info(
                "No experiment set in this session. Execute run() or compile() in order to set an experiment in this session."
            )
        else:
            self._experiment_definition.save(filename)

    def load_compiled_experiment(self, filename: str):
        """Loads a compiled experiment from a given file into the session.

        Args:
            filename:
                Filename (full path) of the experiment should be loaded
                into the session.
        """

        self._compiled_experiment = CompiledExperiment.load(filename)

    def save_compiled_experiment(self, filename: str):
        """Saves the compiled experiment from the session into a given file.

        Args:
            filename:
                Filename (full path) of the file where the experiment
                should be stored in.
        """
        if self._compiled_experiment is None:
            self.logger.info(
                "No compiled experiment set in this session. Execute run() or compile() "
                "in order to compile an experiment."
            )
        else:
            self._compiled_experiment.save(filename)

    def load_experiment_calibration(self, filename: str):
        """Loads a experiment calibration from a given file into the session.

        Args:
            filename: Filename (full path) of the calibration should be loaded into the session.
        """
        calibration = Calibration.load(filename)
        self._experiment_definition.set_calibration(calibration)

    def save_experiment_calibration(self, filename: str):
        """Saves the experiment calibration from the session into a given file.

        Args:
            filename: Filename (full path) of the file where the calibration should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info("No experiment set in this session.")
        else:
            calibration = self._experiment_definition.get_calibration()
            calibration.save(filename)

    def load_signal_map(self, filename: str):
        """Loads a signal map from a given file and sets it to the experiment in the session.

        Args:
            filename: Filename (full path) of the mapping that should be loaded into the session.
        """
        self._experiment_definition.load_signal_map(filename)

    def save_signal_map(self, filename: str):
        """Saves the signal mapping from experiment in the session into a given file.

        Args:
            filename: Filename (full path) of the file where the mapping should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info("No experiment set in this session.")
        else:
            signal_map = self._experiment_definition.get_signal_map()
            Serializer.to_json_file(signal_map, filename)

    def save_results(self, filename: str):
        """Saves the result from the session into a given file.

        Args:
            filename: Filename (full path) of the file where the result should be stored in.
        """
        if self._last_results is None:
            self.logger.info(
                "No results available in this session. Execute run() or simulate_outputs() in order to generate an experiment's result."
            )
        else:
            self._last_results.save(filename)

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
