# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import logging
from enum import Enum
from abc import ABC, abstractmethod
import time
from functools import lru_cache
from typing import List

import zhinst.core as zi

from laboneq.controller.devices.zi_emulator import ziDAQServerEmulator
from zhinst.toolkit import Session as TKSession
from .util import LabOneQControllerException
from .versioning import LabOneVersion
from .cache import Cache, CacheTreeNode


class CachingStrategy(Enum):
    CACHE = "cache"
    NO_CACHE = "no_cache"


class DaqNodeAction(ABC):
    def __init__(self, daq, path, caching_strategy):
        if daq is None:
            raise LabOneQControllerException("DaqNodeAction requires valid daq")
        if not isinstance(path, str):
            raise LabOneQControllerException("Path must be a string")

        self.daq: DaqWrapper = daq
        self.path = path
        self.caching_strategy = caching_strategy


class DaqNodeSetAction(DaqNodeAction):
    def __init__(
        self, daq, path, value, filename=None, caching_strategy=CachingStrategy.CACHE
    ):
        super().__init__(daq, path, caching_strategy)

        if value is None:
            raise LabOneQControllerException(
                "DaqNodeSetAction requires valid value to set"
            )

        self.value = value
        self.filename = filename

    def to_string(self):
        return f"set {self.path} -> {self.value}"

    def __str__(self):
        return self.to_string()


class DaqNodeGetAction(DaqNodeAction):
    def __init__(self, daq, path, caching_strategy=CachingStrategy.CACHE):
        super().__init__(daq, path, caching_strategy)

    def to_string(self):
        return f"get {self.path}"

    def __str__(self):
        return self.to_string()


class DaqNodeWaitAction(DaqNodeAction):
    def __init__(
        self, daq, path, expected_value, caching_strategy=CachingStrategy.NO_CACHE
    ):
        super().__init__(daq, path, caching_strategy)

        if expected_value is None:
            raise QCCSControllerException(
                "DaqNodeSetAction requires valid value to set"
            )

        self.expected_value = expected_value

    def to_string(self):
        return f"{self.path} == {self.expected_value}"

    def __str__(self):
        return self.to_string()

    def is_fulfilled(self, value):
        return value == self.expected_value


class ZiApiWrapperBase(ABC):
    def __init__(self, name):
        self._name = name
        self._logger = logging.getLogger(__name__)
        self._node_logger = logging.getLogger("node.log")
        self._node_cache_root = Cache(f"cache({name})")

    @abstractmethod
    def _api_wrapper(self, method_name, *args, **kwargs):
        pass

    @abstractmethod
    def extract_cache_tree(self):
        pass

    @abstractmethod
    def inject_cache_tree(self, cache_tree_node):
        pass

    @property
    def name(self):
        return self._name

    def _log_node(self, str):
        self._node_logger.debug(str)

    def _log_set(self, method_name: str, daq_action):
        path = daq_action.path

        # @TODO(andreyk): remove this after node logging is refactored
        # it's needed because node logger doesn't support  multiline strings
        value = daq_action.value if daq_action.filename is None else daq_action.filename

        # this is a hack to maintain compatibility with old ground truth files
        # @TODO(andreyk): remove compatibility hack and refactor the node logger to support lists and other types
        if (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, complex)
            or isinstance(value, str)
        ):
            self._log_node(f"{method_name} {path} {value}")

        self._logger.debug("%s - %s -> %s", method_name, path, value)

    def _log_sets(self, method_name, daq_actions: list):
        for action in daq_actions:
            self._log_set(method_name, action)

    def _log_get(self, method_name: str, path: str):
        self._log_node(f"{method_name} {path} -")
        self._logger.debug("%s - %s", method_name, path)

    def _log_gets(self, method_name, daq_actions):
        for daq_action in daq_actions:
            self._log_get(method_name, daq_action.path)

    def get(self, daq_action):
        self._log_get("get", daq_action.path)
        daq_reply = self._api_wrapper("get", daq_action.path, flat=True)
        return self._api_reply_to_val_history_dict(daq_reply)[daq_action.path][-1]

    def _actions_to_set_api_input(self, daq_actions):
        return [[action.path, action.value] for action in daq_actions]

    def batch_set(self, daq_actions: List[DaqNodeAction]):
        """Set the list of nodes in one call to API

        Parameters:
            daq_actions: a list of DaqNodeSetAction objects

        Returns:
            when all nodes are set
        """

        if not isinstance(daq_actions, list):
            raise LabOneQControllerException("List expected")

        node_list = [daq_action.path for daq_action in daq_actions]
        self._logger.debug("Batch set node list: %s", node_list)

        api_input = []
        daq_actions_to_execute = []

        for action in daq_actions:
            if not isinstance(action, DaqNodeSetAction):
                raise LabOneQControllerException(
                    "List elements must be of type DaqNodeSetAction"
                )
            if action.caching_strategy == CachingStrategy.CACHE:
                if self._node_cache_root.set(action.path, action.value) is not None:
                    continue
            else:
                self._logger.debug("set not caching: %s", action.path)

            daq_actions_to_execute.append(action)

        self._log_sets("set", daq_actions_to_execute)

        api_input = self._actions_to_set_api_input(daq_actions_to_execute)
        self._logger.debug("API set node list: %s", api_input)

        return self._api_wrapper("set", api_input)

    def _subscribe_to_wait_conditions(self, wait_conditions):
        for wait_condition in wait_conditions:
            # subscribe to any changes during polling
            self._api_wrapper("subscribe", wait_condition.path)

    def _api_reply_to_val_history_dict(self, daq_reply):
        """Converts a DAQ reply with flat=True to a path-value_history dict
        e.g.: { path: { "value" : [ val1, val2 ] }} to { path: [ val1, val2 ] }
        """

        res = {}
        for path in daq_reply:
            res[path] = daq_reply[path]["value"]
        return res

    def execute(self):
        return self._api_wrapper("execute")


@dataclass
class ServerQualifier:
    dry_run: bool = True
    host: str = None
    port: int = None
    api_level: int = None
    ignore_lab_one_version_error: bool = False


class DaqWrapper(ZiApiWrapperBase):
    def __init__(self, name, server_qualifier: ServerQualifier):
        super().__init__(name)
        self._server_qualifier = server_qualifier
        self._awg_module_wrappers = []
        self._is_valid = False
        self._dataserver_version = LabOneVersion.LATEST
        self._vector_counter = 0

        if not server_qualifier.dry_run:
            from laboneq._token import token_check

            token_check()

        ZiApiClass = ziDAQServerEmulator if server_qualifier.dry_run else zi.ziDAQServer

        try:
            self._zi_api_object = ZiApiClass(
                self.server_qualifier.host,
                self.server_qualifier.port,
                self.server_qualifier.api_level,
            )
        except RuntimeError as exp:
            raise LabOneQControllerException(str(exp))

        path = "/zi/about/version"
        version_str = self.batch_get([DaqNodeGetAction(self, path)])[path]
        try:
            self._dataserver_version = LabOneVersion(version_str)
        except ValueError:
            err_msg = f"Version {version_str} is not supported by QCCS."
            if server_qualifier.ignore_lab_one_version_error:
                self._logger.warning("Ignoring that %s", err_msg)
                self._dataserver_version = LabOneVersion.LATEST
            else:
                raise LabOneQControllerException(err_msg)

        if not server_qualifier.dry_run:
            [major, minor] = zi.__version__.split(".")[0:2]
            zi_python_version = f"{major}.{minor}"
            if zi_python_version != version_str:
                err_msg = f"Version of dataserver ({version_str}) and zi python ({zi_python_version}) do not match."
                if self.server_qualifier.ignore_lab_one_version_error:
                    self._logger.warning("Ignoring that %s", err_msg)
                else:
                    raise LabOneQControllerException(err_msg)

        self._logger.info(
            "Connected to Zurich Instrument's Data Server version %s at %s:%s",
            version_str,
            self.server_qualifier.host,
            self.server_qualifier.port,
        )

        self._is_valid = True

    def __str__(self):
        return f"DAQ - {self._server_qualifier.host}:{self._server_qualifier.port}"

    def _api_wrapper(self, method_name, *args, **kwargs):
        try:
            api_method = getattr(self._zi_api_object, method_name)
            retval = api_method(*args, **kwargs)
            return retval
        except Exception as ex:
            raise LabOneQControllerException(
                f"Exception {ex} when calling method {method_name} with {args} and {kwargs}"
            )

    @property
    @lru_cache(maxsize=1)
    def toolkit_session(self) -> TKSession:
        """Toolkit session from the initialized DAQ session."""
        return TKSession(
            server_host=self.server_qualifier.host,
            server_port=self.server_qualifier.port,
            connection=self._zi_api_object,
        )

    @property
    def server_qualifier(self):
        return self._server_qualifier

    def create_awg_module(self, name):
        self._logger.info("Create AWG module %s", name)
        self._awg_module_wrappers = [
            wrapper for wrapper in self._awg_module_wrappers if wrapper.name != name
        ]

        module = AwgModuleWrapper(name, self._zi_api_object.awgModule())
        self._awg_module_wrappers.append(module)
        return module

    def is_valid(self):
        return self._is_valid

    def extract_cache_tree(self):
        self._logger.debug("Extracting cache")
        cache_tree = CacheTreeNode(self._node_cache_root)
        for awg_module in self._awg_module_wrappers:
            cache_tree.add_child(awg_module.name, awg_module.extract_cache_tree())
        return cache_tree

    def inject_cache_tree(self, cache_tree_node):
        self._logger.debug("Injecting cache")
        self._node_cache_root = cache_tree_node.cache
        for awg_module in self._awg_module_wrappers:
            if awg_module.name in cache_tree_node.children:
                awg_module.inject_cache_tree(cache_tree_node.children[awg_module.name])

    @property
    def dataserver_version(self):
        return self._dataserver_version

    def connectDevice(self, serial: str, interface: str):
        if not isinstance(serial, str) or not isinstance(interface, str):
            raise LabOneQControllerException("Serial and interface must be strings")

        self._logger.debug("connectDevice %s:%s", serial, interface)
        self._api_wrapper("connectDevice", serial, interface)

    def disconnectDevice(self, serial: str):
        if not isinstance(serial, str):
            raise LabOneQControllerException("Serial must be string")

        self._logger.debug("disconnectDevice %s", serial)
        self._api_wrapper("disconnectDevice", serial)

    def sync(self):
        self._logger.debug("sync")
        return self._api_wrapper("sync")

    def get_raw(self, path):
        # @TODO(andreyk): remove this method and refactor call site
        if not isinstance(path, str):
            raise LabOneQControllerException("Path must be a string")

        self._log_get("get", path)
        return self._api_wrapper("get", path, flat=True)

    def prepare_conditions(self, wait_conditions):
        self._logger.debug("Preparing wait for: ")
        for wait_condition in wait_conditions:
            if not isinstance(wait_condition, DaqNodeWaitAction):
                raise QCCSControllerException(
                    "List elements must be DaqNodeWaitAction objects"
                )

            self._logger.debug("  %s", wait_condition.to_string())
        self._logger.debug("Subscribing...")
        self._subscribe_to_wait_conditions(wait_conditions)
        self._logger.debug("Subscribing done")

    def wait_all_conditions(self, wait_conditions, min_wait_time: float):
        """Receives a list of waiting conditions and returns when all of them are fulfilled

        E.g. wait for a node path /device/path/node to become 1 and /device/path/node2 to become 0

        Parameters:
            waiting_conditions: a list of DaqNodeWaitAction objects

        Returns:
            when all conditions are fulfilled
        """
        if not isinstance(wait_conditions, list):
            raise QCCSControllerException("List expected")

        if min_wait_time is None:
            self._logger.warning(
                "No estimation available for the execution time, assuming 10 sec."
            )
            min_wait_time = 10.0
        elif min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            self._logger.info("Estimated RT execution time: %.2f s.", min_wait_time)
        guarded_wait_time = round(
            min_wait_time * 1.1 + 1
        )  # +10% and fixed 1sec guard time

        self._logger.debug("Waiting on: ")
        for wait_condition in wait_conditions:
            if not isinstance(wait_condition, DaqNodeWaitAction):
                raise QCCSControllerException(
                    "List elements must be DaqNodeWaitAction objects"
                )

            self._logger.debug("  %s", wait_condition.to_string())

        conditions_still_open = {
            wait_condition.path: wait_condition for wait_condition in wait_conditions
        }

        poll_start = time.time()
        while True:
            daq_reply = self._api_wrapper("poll", 0.001, 10, flat=True)
            current_values = self._api_reply_to_val_history_dict(daq_reply)

            for path in current_values:
                value_history = current_values[path]

                for value in value_history:

                    if not path in conditions_still_open:
                        continue

                    wait_condition = conditions_still_open[path]

                    if not wait_condition.is_fulfilled(value):
                        continue

                    conditions_still_open.pop(path)
                    self._logger.debug(
                        "Condition fulfilled: %s", wait_condition.to_string()
                    )

            if not bool(conditions_still_open):
                break

            polling_for = time.time() - poll_start
            if polling_for > guarded_wait_time:
                self._logger.warning(
                    "Conditions below still not fulfilled after %f s, estimated execution time was %.2f s. Continuing to the next step.",
                    guarded_wait_time,
                    min_wait_time,
                )
                for condition in conditions_still_open.values():
                    self._logger.warning("  - %s", condition.to_string())
                break

        self._logger.debug("Stopped waiting, conditions fulfilled")

    def _filter_cached_actions(self, daq_actions):
        cached_values = {}
        actions_to_perform = []

        for daq_action in daq_actions:
            if not isinstance(daq_action, DaqNodeGetAction):
                raise LabOneQControllerException(
                    "Elements must be DaqNodeGetAction objects"
                )

            if daq_action.caching_strategy == CachingStrategy.CACHE:
                cached_val = self._node_cache_root.get(daq_action.path)
                if cached_val is not None:
                    cached_values[daq_action.path] = cached_val
                    continue
            actions_to_perform.append(daq_action)

        return cached_values, actions_to_perform

    def _update_cache_with_value_history_dict(self, value_history_dict):
        result = {}

        for path in value_history_dict:
            last_value = value_history_dict[path][-1]
            result[path] = last_value
            self._node_cache_root.force_set(path, last_value)

        return result

    def batch_get(self, daq_actions):
        if not isinstance(daq_actions, list):
            raise LabOneQControllerException("Paths must be a list")

        if len(daq_actions) == 0:
            return {}

        self._log_gets("get", daq_actions)

        cached_values, actions_to_perform = self._filter_cached_actions(daq_actions)

        for daq_action in actions_to_perform:
            self._logger.debug("get not caching: %s", daq_action.path)

        api_input = ",".join([daq_action.path for daq_action in actions_to_perform])

        daq_reply = self._api_wrapper("get", api_input, flat=True)
        current_values = self._api_reply_to_val_history_dict(daq_reply)

        get_values = self._update_cache_with_value_history_dict(current_values)

        return {**cached_values, **get_values}


class DaqWrapperDryRun(DaqWrapper):
    def __init__(self, name, server_qualifier: ServerQualifier = ServerQualifier()):
        assert server_qualifier.dry_run == True
        super().__init__(name, server_qualifier)

    def map_device_type(self, serial: str, type: str):
        assert isinstance(self._zi_api_object, ziDAQServerEmulator)
        self._zi_api_object.map_device_type(serial, type)


class AwgModuleWrapper(ZiApiWrapperBase):
    def __init__(self, name, zi_awg_module):
        super().__init__(name)
        self._zi_api_object = zi_awg_module

    def _api_wrapper(self, method_name, *args, **kwargs):
        api_method = getattr(self._zi_api_object, method_name)
        res = api_method(*args, **kwargs)
        return res

    @property
    def progress(self):
        return self._zi_api_object.progress()

    @property
    def elf_status(self):
        return self._zi_api_object.getInt("elf/status")

    def extract_cache_tree(self):
        self._logger.debug("Extracting cache")
        return CacheTreeNode(self._node_cache_root)

    def inject_cache_tree(self, cache_tree_node):
        self._logger.debug("Injecting cache")
        self._node_cache_root = cache_tree_node.cache

    def _api_reply_to_val_history_dict(self, daq_reply):
        """Converts AWG module reply with flat=True to path-value_history dict
        e.g. { path: [ val1, val2 ] } to { path: [ val1, val2 ] }
        """
        res = {}
        for path in daq_reply:
            res[path] = daq_reply[path]
        return res
