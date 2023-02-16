# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from numpy import typing as npt

from laboneq._observability.tracing import trace
from laboneq.controller.toolkit_adapter import ToolkitDevices
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types import CompiledExperiment
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.laboneq_facade import LabOneQFacade
from laboneq.dsl.result import Results
from laboneq.dsl.serialization import Serializer

if TYPE_CHECKING:
    from laboneq.controller import Controller
    from laboneq.dsl.experiment.pulse import Pulse


class ConnectionState:
    connected: bool = False
    emulated: bool = False


class Session:
    """This Session class represents the main endpoint for the user interaction with the QCCS system.

    The session holds
        * the wiring definition of the devices
        * the experiment definition that should be run on the devices
        * the calibration of the devices for experiment
        * the compiled experiment
        * the result of the executed experiment

    The Session is a statefull object that hold all of the above. The expected steps to interact with the session are:
        * initial state (construction)
        * setting the device setup (optionally during construction)
        * (optional) setting the calibration of the devices
        * connecting to the devices (or the emulator)
        * compiling the experiment
        * running the experiment
        * accessing the results of the last run experiment

    The session is serializable in every state.
    """

    class _SessionDeco:
        """Collection of decorators specific to the Session class"""

        @staticmethod
        def entrypoint(func):
            """Decorator for Session public methods.

            Add this decorator to every public method of the Session,
            that may be directly called by the user. These methods
            may also call each other, and the decorator ensures that
            in case of an occasional error, the 1st method (directly
            called by the user) is reported in the error message,
            even if the error happens inside a nested method call.
            """

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                self._call_stack.append(func.__name__)
                res = func(self, *args, **kwargs)
                self._call_stack.pop()
                return res

            return wrapper

    def __init__(
        self,
        device_setup: DeviceSetup = None,
        log_level: int = logging.INFO,
        performance_log=False,
        configure_logging=True,
        _last_results=None,
        compiled_experiment=None,
        experiment=None,
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
            configure_logging: Whether to configure logger. Can be disabled for custom logging use cases.

        .. versionchanged:: 2.0
            Removed `pass_v3_to_compiler` argument.
            Removed `max_simulation_time` instance variable.
        """
        self._device_setup = device_setup if device_setup else DeviceSetup()
        self._controller: Controller = None
        self._connection_state: ConnectionState = ConnectionState()
        self._experiment_definition = experiment
        self._compiled_experiment = compiled_experiment
        # Keeps call stack of public methods (when calling each other)
        self._call_stack: List[str] = list()
        self._last_results = _last_results
        if configure_logging:
            LabOneQFacade.init_logging(
                log_level=log_level, performance_log=performance_log
            )
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(log_level)
        else:
            self._logger = logging.getLogger("null")
        self._user_functions: Dict[str, Callable] = {}
        self._toolkit_devices = ToolkitDevices()

    @property
    def devices(self) -> ToolkitDevices:
        """Connected devices included in the system setup.

        Allows the modification/inspection of the state of the device and its nodes.

        Devices exist once the session is connected. After disconnecting, devices
        are empty.

        Usage:

            >>> session.connect()
            >>> session.devices["device_hdawg"].awgs[0].outputs[0].amplitude(1)
            >>> session.devices["DEV1234"].awgs[0].outputs[0].amplitude()
            1
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
            and self._user_functions == other._user_functions
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
            f"Session not connected.\n"
            f"{self._call_stack[0]}() requires an established connection to devices in order to execute the experiment.\n"
            f"Call connect() first. Use connect(do_emulation=True) if you want to emulate the devices' behavior only."
        )
        message = message or default_message
        if fail:
            raise LabOneQException(message)
        self.logger.error(message)
        return False

    @_SessionDeco.entrypoint
    def register_user_function(self, func, name: str = None):
        """Registers a user function to be referred from the experiment's `call` operation.

        Args:
            func (function): User function that is registered.
            name (str):     Optional name to use as the argument to experiment's `call` operation to refer to this
                            function. If not provided, function name will be used.
        """

        if name is None:
            name = func.__name__
        self._user_functions[name] = func

    @_SessionDeco.entrypoint
    @trace("session.connect()")
    def connect(
        self, do_emulation=False, ignore_lab_one_version_error=False
    ) -> ConnectionState:
        """Connects the session to the QCCS system.

        Args:
            do_emulation (bool): Specifies if the session should connect to a emulator
                                 (in the case of 'True') or the real system (in the case of 'False').

            ignore_lab_one_version_error (bool): Ignore LabOne and LabOne Q version mismatch error.
        """
        self._ignore_lab_one_version_error = ignore_lab_one_version_error
        if (
            self._connection_state.connected
            and self._connection_state.emulated != do_emulation
        ):
            self.disconnect()
        self._connection_state.emulated = do_emulation
        LabOneQFacade.connect(self)
        self._connection_state.connected = True
        return self._connection_state

    @_SessionDeco.entrypoint
    def disconnect(self) -> ConnectionState:
        """Disconnects the session from the devices."""
        self._connection_state.connected = False
        LabOneQFacade.disconnect(self)
        return self._connection_state

    @property
    def connection_state(self) -> ConnectionState:
        """State of the connection."""
        return self._connection_state

    @_SessionDeco.entrypoint
    @trace("session.compile()")
    def compile(
        self,
        experiment: Experiment,
        compiler_settings: Dict = None,
    ) -> Optional[CompiledExperiment]:
        """Compiles the specified experiment and stores it in the compiled_experiment property.

        Args:
            experiment: Experiment instance that should be compiled.
            compiler_settings: Extra options passed to the compiler.

        .. versionchanged:: 2.0

            Removed `do_simulation` argument. Use :class:`~.OutputSimulator` instead.
        """
        if not self._assert_connected(fail=False):
            return

        self._experiment_definition = experiment
        self._compiled_experiment = LabOneQFacade.compile(
            self, self.logger, compiler_settings=compiler_settings
        )
        self._last_results = None
        return self._compiled_experiment

    @property
    def compiled_experiment(self) -> Optional[CompiledExperiment]:
        """Access to the compiled experiment.

        The compiled experiment can be assigned to a different session if the device setup is matching.
        """
        return self._compiled_experiment

    @_SessionDeco.entrypoint
    @trace("session.run()")
    def run(
        self,
        experiment: Optional[Union[Experiment, CompiledExperiment]] = None,
    ) -> Optional[Results]:
        """Executes the compiled experiment.

        If no experiment is specified, the last compiled experiment is run.
        If an experiment is specified, the provided experiment is assigned to the
        internal experiment of the session.

        .. versionchanged:: 2.0

            Removed `do_simulation` argument. Use :class:`~.OutputSimulator` instead.

        Args:
            experiment: Optional. Experiment instance that should be
                run. The experiment will be compiled if it has not been yet. If no
                experiment is specified the previously assigned and compiled experiment
                is used.

        Returns:
            A `Results` object in case of success. `None` if the session is not
            connected.
        """
        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
            else:
                self.compile(experiment)

        if not self._assert_connected(fail=False):
            return

        self._last_results = Results(
            experiment=self.experiment,
            device_setup=self.device_setup,
            compiled_experiment=self.compiled_experiment,
            acquired_results={},
            user_func_results={},
            execution_errors=[],
        )
        LabOneQFacade.run(self)
        return self.results

    @_SessionDeco.entrypoint
    def submit(
        self,
        experiment: Optional[Union[Experiment, CompiledExperiment]] = None,
        queue: Callable[[str, CompiledExperiment, DeviceSetup], Any] = None,
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
                ``(name: str, experiment: CompiledExperiment, device_setup: DeviceSetup)``
                which returns an object with which users can query results.

        Returns:
            An object with which users can query results. Details depend on the
            implementation of the queue.
        """

        if experiment:
            if isinstance(experiment, CompiledExperiment):
                self._compiled_experiment = experiment
            else:
                self.compile(experiment)

        return queue(experiment.uid, self.compiled_experiment, self.device_setup)

    @_SessionDeco.entrypoint
    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        LabOneQFacade.replace_pulse(self, pulse_uid, pulse_or_array)

    def get_results(self) -> Results:
        """
        Object holding the result of the last experiment execution. Referencing the results creates a deep copy.
        """
        if not self._last_results:
            raise LabOneQException(
                "No results available. Execute run() or simulate_outputs() in order to generate an experiment's result."
            )
        return deepcopy(self._last_results)

    @property
    def results(self) -> Results:
        """
        Object holding the result of the last experiment execution. Attention! This accessor is provided for better
        performance, unlike 'get_result' it doesn't make a copy, but instead returns the reference to the live
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
    def log_level(self):
        return self._logger.level

    @property
    def logger(self):
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
    def load(filename):
        """Loads the session from a serialized file.
        A restored session from a loaded file will end up in the same state of the session that saved the file in the first place.

        Args:
            filename (str): Filename (full path) of the file that should be loaded into the session.
        """

        import json

        with open(filename, mode="r") as file:
            session_dict = json.load(file)
        constructor_args = {}

        for field, field_type in Session._session_fields().items():
            logging.getLogger(__name__).info("Loading %s of type %s", field, field_type)
            constructor_args[field] = Serializer.load(session_dict[field], field_type)

        return Session(**constructor_args)

    def save(self, filename):
        """Stores the session from a serialized file.
        A restored session from a loaded file will end up in the same state of the session that saved the file in the first place.

        Args:
            filename (str): Filename (full path) of the file where the session should be stored in.
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

    def load_device_setup(self, filename):
        """Loads a device setup from a given file into the session.

        Args:
            filename (str): Filename (full path) of the setup should be loaded into the session.
        """
        self._device_setup = DeviceSetup.load(filename)

    def save_device_setup(self, filename):
        """Saves the device setup from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the setup should be stored in.
        """
        if self._device_setup is None:
            self.logger.info("No device setup set in this session.")
        else:
            self._device_setup.save(filename)

    def load_device_calibration(self, filename):
        """Loads a device calibration from a given file into the session.

        Args:
            filename (str): Filename (full path) of the calibration should be loaded into the session.
        """
        calibration = Calibration.load(filename)
        self._device_setup.set_calibration(calibration)

    def save_device_calibration(self, filename):
        """Saves the device calibration from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the calibration should be stored in.
        """
        if self._device_setup is None:
            self.logger.info("No device setup set in this session.")
        else:
            calibration = self._device_setup.get_calibration()
            calibration.save(filename)

    def load_experiment(self, filename):
        """Loads an experiment from a given file into the session.

        Args:
            filename (str): Filename (full path) of the experiment should be loaded into the session.
        """
        self._experiment_definition = Experiment.load(filename)

    def save_experiment(self, filename):
        """Saves the experiment from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the experiment should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info(
                "No experiment set in this session. Execute run() or compile() in order to set an experiment in this session."
            )
        else:
            self._experiment_definition.save(filename)

    def load_compiled_experiment(self, filename):
        """Loads a compiled experiment from a given file into the session.

        Args:
            filename (str): Filename (full path) of the experiment should be loaded
            into the session.
        """

        self._compiled_experiment = CompiledExperiment.load(filename)

    def save_compiled_experiment(self, filename):
        """Saves the compiled experiment from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the experiment
            should be stored in.
        """
        if self._compiled_experiment is None:
            self.logger.info(
                "No compiled experiment set in this session. Execute run() or compile() "
                "in order to compile an experiment."
            )
        else:
            self._compiled_experiment.save(filename)

    def load_experiment_calibration(self, filename):
        """Loads a experiment calibration from a given file into the session.

        Args:
            filename (str): Filename (full path) of the calibration should be loaded into the session.
        """
        calibration = Calibration.load(filename)
        self._experiment_definition.set_calibration(calibration)

    def save_experiment_calibration(self, filename):
        """Saves the experiment calibration from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the calibration should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info("No experiment set in this session.")
        else:
            calibration = self._experiment_definition.get_calibration()
            calibration.save(filename)

    def load_signal_map(self, filename):
        """Loads a signal map from a given file and sets it to the experiment in the session.

        Args:
            filename (str): Filename (full path) of the mapping that should be loaded into the session.
        """
        self._experiment_definition.load_signal_map(filename)

    def save_signal_map(self, filename):
        """Saves the signal mapping from experiment in the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the mapping should be stored in.
        """
        if self._experiment_definition is None:
            self.logger.info("No experiment set in this session.")
        else:
            signal_map = self._experiment_definition.get_signal_map()
            Serializer.to_json_file(signal_map, filename)

    def save_results(self, filename):
        """Saves the result from the session into a given file.

        Args:
            filename (str): Filename (full path) of the file where the result should be stored in.
        """
        if self._last_results is None:
            self.logger.info(
                "No results available in this session. Execute run() or simulate_outputs() in order to generate an experiment's result."
            )
        else:
            self._last_results.save(filename)
