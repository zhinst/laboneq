# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
import re
import sched

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

from numpy import typing as npt
import numpy as np


@dataclass
class NodeBase:
    "A class modeling a single node. Specialized for specific node types."
    read_only: bool = False
    subscribed: bool = False
    handler: Callable = None


@dataclass
class NodeFloat(NodeBase):
    value: float = 0.0


@dataclass
class NodeInt(NodeBase):
    value: int = 0


@dataclass
class NodeStr(NodeBase):
    value: str = ""


@dataclass
class NodeVectorFloat(NodeBase):
    value: npt.ArrayLike = np.array([], dtype=np.float64)


@dataclass
class NodeVectorInt(NodeBase):
    value: npt.ArrayLike = np.array([], dtype=np.int64)


@dataclass
class NodeVectorComplex(NodeBase):
    value: npt.ArrayLike = np.array([], dtype=np.complex128)


class NodeType(Enum):
    FLOAT = NodeFloat
    INT = NodeInt
    STR = NodeStr
    VECTOR_FLOAT = NodeVectorFloat
    VECTOR_INT = NodeVectorInt
    VECTOR_COMPLEX = NodeVectorComplex


@dataclass
class NodeInfo:
    "Node descriptor to use in node definitions."
    type: NodeType = NodeType.FLOAT
    default: Any = 0.0
    read_only: bool = False
    handler: Callable = None


def _node_factory(node_info: NodeInfo) -> NodeBase:
    "Helper function constructing concrete node instance from a NodeInfo descriptor."
    return node_info.type.value(
        read_only=node_info.read_only,
        value=node_info.default,
        handler=node_info.handler,
    )


@dataclass
class PollEvent:
    "A class representing a single poll event"
    path: str = None
    timestamp: int = None
    value: Any = None


class DevEmu(ABC):
    "Base class emulating a device, specialized per device type."

    def __init__(self, scheduler: sched.scheduler):
        self._scheduler = scheduler
        self._node_tree: Dict[str, NodeBase] = {}
        self._poll_queue: List[PollEvent] = []
        self._total_subscribed: int = 0

    @abstractmethod
    def serial(self) -> str:
        ...

    @abstractmethod
    def _node_def(self) -> Dict[str, NodeInfo]:
        ...

    def _full_path(self, dev_path: str) -> str:
        return f"/{self.serial().lower()}/{dev_path}"

    def _make_node(self, dev_path: str):
        node_def = self._node_def()
        new_node_def = node_def.get(dev_path)
        if new_node_def is None:
            new_node_def = NodeInfo(type=NodeType.INT, default=0, read_only=False)
        new_node = _node_factory(new_node_def)
        return new_node

    def _get_node(self, dev_path: str) -> NodeBase:
        node = self._node_tree.get(dev_path)
        if node is None:
            node = self._make_node(dev_path)
            self._node_tree[dev_path] = node
        return node

    def get(self, dev_path: str) -> Any:
        node = self._get_node(dev_path)
        return node.value

    def _set_val(self, dev_path: str, value: Any) -> NodeBase:
        node = self._get_node(dev_path)
        node.value = value
        if node.subscribed:
            self._poll_queue.append(
                PollEvent(path=self._full_path(dev_path), value=value)
            )
        return node

    def set(self, dev_path: str, value: Any):
        node = self._set_val(dev_path, value)
        if node.handler is not None:
            node.handler(self)

    def subscribe(self, dev_path: str):
        node = self._get_node(dev_path)
        node.subscribed = True

    def poll(self) -> List[PollEvent]:
        output = self._poll_queue[:]
        self._poll_queue.clear()
        return output


class DevEmuZI(DevEmu):
    _zi_node_def: Dict[str, NodeInfo] = {
        "about/version": NodeInfo(type=NodeType.STR, default="22.02", read_only=True),
        "about/revision": NodeInfo(type=NodeType.STR, default="99999", read_only=True),
    }

    def serial(self) -> str:
        return "ZI"

    def _node_def(self) -> Dict[str, NodeInfo]:
        return DevEmuZI._zi_node_def


class DevEmuHW(DevEmu):
    def __init__(self, serial: str, scheduler: sched.scheduler):
        super().__init__(scheduler)
        self._serial = serial

    def serial(self) -> str:
        return self._serial


class DevEmuDummy(DevEmuHW):
    def _node_def(self) -> Dict[str, NodeInfo]:
        return {}


class DevEmuHDAWG(DevEmuHW):
    def _awg_stop(self, awg_idx):
        self._set_val(f"awgs/{awg_idx}/enable", 0)

    def _awg_execute(self, awg_idx):
        self._scheduler.enter(
            delay=0.01, priority=0, action=self._awg_stop, argument=(awg_idx,)
        )

    _hdawg_node_def: Dict[str, NodeInfo] = None

    def _node_def(self) -> Dict[str, NodeInfo]:
        if DevEmuHDAWG._hdawg_node_def is None:
            nd = {}
            for awg_idx in range(4):
                nd[f"awgs/{awg_idx}/enable"] = NodeInfo(
                    type=NodeType.INT,
                    default=0,
                    handler=partial(DevEmuHDAWG._awg_execute, awg_idx=awg_idx),
                )
            DevEmuHDAWG._hdawg_node_def = nd
        return DevEmuHDAWG._hdawg_node_def


class DevEmuUHFQA(DevEmuHW):
    def _awg_stop(self):
        self._set_val(f"awgs/0/enable", 0)

    def _awg_execute(self):
        self._scheduler.enter(delay=0.01, priority=0, action=self._awg_stop)

    _uhfqa_node_def: Dict[str, NodeInfo] = {
        "awgs/0/enable": NodeInfo(type=NodeType.INT, default=0, handler=_awg_execute),
    }

    def _node_def(self) -> Dict[str, NodeInfo]:
        return DevEmuUHFQA._uhfqa_node_def


class DevEmuPQSC(DevEmuHW):
    def _trig_stop(self):
        self._set_val(f"execution/enable", 0)

    def _trig_execute(self):
        self._scheduler.enter(delay=0.01, priority=0, action=self._trig_stop)

    _pqsc_node_def: Dict[str, NodeInfo] = {
        "execution/enable": NodeInfo(
            type=NodeType.INT, default=0, handler=_trig_execute
        ),
    }

    def _node_def(self) -> Dict[str, NodeInfo]:
        return DevEmuPQSC._pqsc_node_def


class DevEmuSHFQA(DevEmuHW):
    def _awg_stop(self, channel: int):
        self._set_val(f"qachannels/{channel}/generator/enable", 0)
        self._set_val(f"qachannels/{channel}/readout/result/enable", 0)
        self._set_val(f"qachannels/{channel}/spectroscopy/result/enable", 0)

    def _awg_execute(self, channel: int):
        self._scheduler.enter(
            delay=0.01, priority=0, action=self._awg_stop, argument=(channel,)
        )

    _shfqa_node_def: Dict[str, NodeInfo] = None

    def _node_def(self) -> Dict[str, NodeInfo]:
        if DevEmuSHFQA._shfqa_node_def is None:
            nd = {}
            for channel in range(4):
                nd[f"qachannels/{channel}/generator/enable"] = NodeInfo(
                    type=NodeType.INT,
                    default=0,
                    handler=partial(DevEmuSHFQA._awg_execute, channel=channel),
                )
                # TODO(2K): emulate result logging
                nd[f"qachannels/{channel}/readout/result/enable"] = NodeInfo(
                    type=NodeType.INT, default=0
                )
                nd[f"qachannels/{channel}/spectroscopy/result/enable"] = NodeInfo(
                    type=NodeType.INT, default=0
                )
            DevEmuSHFQA._shfqa_node_def = nd
        return DevEmuSHFQA._shfqa_node_def


class DevEmuSHFSG(DevEmuHW):
    def _awg_stop(self, channel: int):
        self._set_val(f"sgchannels/{channel}/awg/enable", 0)

    def _awg_execute(self, channel: int):
        self._scheduler.enter(
            delay=0.01, priority=0, action=self._awg_stop, argument=(channel,)
        )

    _shfsg_node_def: Dict[str, NodeInfo] = None

    def _node_def(self) -> Dict[str, NodeInfo]:
        if DevEmuSHFSG._shfsg_node_def is None:
            nd = {}
            for channel in range(8):
                nd[f"sgchannels/{channel}/awg/enable"] = NodeInfo(
                    type=NodeType.INT,
                    default=0,
                    handler=partial(DevEmuSHFSG._awg_execute, channel=channel),
                )
            DevEmuSHFSG._shfsg_node_def = nd
        return DevEmuSHFSG._shfsg_node_def


def _serial_to_device_type(dev_id: str):
    m = re.match(pattern="DEV([0-9]+)[0-9]{3}", string=dev_id.upper())
    if m:
        num = int(m.group(1))
        if num < 2:  # HF2 0 ... 1999 - not supported
            return DevEmuDummy
        elif num < 3:  # UHF 2000 ... 2999
            return DevEmuUHFQA
        elif num < 10:  # HD 8000 ... 9999
            return DevEmuHDAWG
        elif num < 12:  # PQSC 10000 ... 11999
            return DevEmuPQSC
        elif num < 13:  # SHF* 12000 ... 12999 # Attention! Only QA supported, no SG
            return DevEmuSHFQA
        else:
            return DevEmuDummy
    else:
        return DevEmuDummy


def _device_factory(
    dev_id: str, scheduler: sched.scheduler, device_type_map: Dict[str, str] = None
) -> DevEmu:
    if device_type_map is None:
        device_type_map = {}
    dev_type_str = device_type_map.get(dev_id.upper())
    if dev_type_str == "HDAWG":
        dev_type = DevEmuHDAWG
    elif dev_type_str == "UHFQA":
        dev_type = DevEmuUHFQA
    elif dev_type_str == "PQSC":
        dev_type = DevEmuPQSC
    elif dev_type_str == "SHFQA":
        dev_type = DevEmuSHFQA
    elif dev_type_str == "SHFSG":
        dev_type = DevEmuSHFSG
    else:
        dev_type = _serial_to_device_type(dev_id)
    return dev_type(dev_id, scheduler)


def _canonical_path_list(path: Union[str, List[str]]) -> List[str]:
    if isinstance(path, list):
        paths = path
    else:
        paths = path.split(",")
    return paths


class ziDAQServerEmulator:
    """ A class replacing the ziPython's 'ziDAQServer', emulating its behavior
        to the extent required by QCCS SW without the real DataServer/HW.
    """

    def __init__(self, host: str, port: int, api_level: int):
        if api_level is None:
            api_level = 6
        assert api_level == 6
        super().__init__()
        self._scheduler = sched.scheduler()
        self._device_type_map: Dict[str, str] = {}
        self._devices: Dict[str, DevEmu] = {"ZI": DevEmuZI(self._scheduler)}

    def map_device_type(self, serial: str, type: str):
        self._device_type_map[serial.upper()] = type.upper()

    def _device_lookup(self, dev_id: str, create: bool = True) -> DevEmu:
        dev_id = dev_id.upper()
        device = self._devices.get(dev_id)
        if device is None and create:
            device = _device_factory(dev_id, self._scheduler, self._device_type_map)
            self._devices[dev_id] = device
        return device

    def _resolve_dev(self, path: str) -> Tuple[DevEmu, str]:
        if path.startswith("/"):
            path = path[1:]
        path_parts = path.split("/")
        device = self._device_lookup(path_parts[0])
        return device, "/".join(path_parts[1:]).lower()

    def _resolve_paths_and_perform(
        self, path: Union[str, List[str]], handler: Callable
    ):
        results = {}
        for p in _canonical_path_list(path):
            device, dev_path = self._resolve_dev(p)
            results[p.lower()] = handler(device, dev_path)
        return results

    def connectDevice(self, dev: str, interface: str, params: str = ""):
        self._progress_scheduler()
        # TODO(2K): model interfaces and parameters
        self._device_lookup(dev)

    def disconnectDevice(self, dev: str):
        self._progress_scheduler()
        device = self._device_lookup(dev, create=False)
        if device is not None:
            del self._devices[dev.upper()]

    def _get(self, device: DevEmu, dev_path: str):
        return device.get(dev_path)

    def get(
        self,
        paths: str,
        *args,
        flat: bool = False,
        all: bool = False,
        settingsonly: bool = True,
    ) -> Any:
        self._progress_scheduler()
        assert len(args) == 0  # Only support new calling signature
        # TODO(2K): handle flags
        raw_results = self._resolve_paths_and_perform(paths, self._get)
        # TODO(2K): reshape results
        assert flat == True
        # TODO(2K): emulate timestamp
        results = {p: {"value": [r]} for p, r in raw_results.items()}
        return results

    def _set(self, device: DevEmu, dev_path: str, value_dict: Dict[str, Any]):
        full_path = device._full_path(dev_path)
        value = value_dict[full_path]
        device.set(dev_path, value)

    @overload
    def set(self, path: str, value: Any):
        ...

    @overload
    def set(self, items: List[List[Any]]):
        ...

    def set(
        self, path_or_items: Union[str, List[List[Any]]], value: Optional[Any] = None
    ):
        self._progress_scheduler()
        if isinstance(path_or_items, str):
            path = path_or_items
            # TODO(2K): stub
        else:
            items = path_or_items
            value_dict = {v[0].lower(): v[1] for v in items}
            paths = list(value_dict.keys())
            self._resolve_paths_and_perform(
                paths, partial(self._set, value_dict=value_dict)
            )

    def _subscribe(self, device: DevEmu, dev_path: str):
        device.subscribe(dev_path)

    def subscribe(self, path: Union[str, List[str]]):
        self._progress_scheduler()
        self._resolve_paths_and_perform(path, self._subscribe)

    def getAsEvent(self, path: str):
        self._progress_scheduler()
        return

    def sync(self):
        # TODO(2K): Do nothing, consider some behavior for testability
        self._progress_scheduler()

    def poll(
        self,
        recording_time_s: float,
        timeout_ms: int,
        flags: int = 0,
        flat: bool = False,
    ) -> Any:
        self._progress_scheduler(max_wait=recording_time_s)
        events: List[PollEvent] = []
        for dev in self._devices.values():
            events.extend(dev.poll())
        result = {}
        # TODO(2K): reshape results
        assert flat == True
        for event in events:
            path_res = result.setdefault(event.path, {"value": []})
            path_res["value"].append(event.value)
        return result

    def awgModule(self) -> AWGModuleEmulator:
        self._progress_scheduler()
        return AWGModuleEmulator(self)

    def _progress_scheduler(self, max_wait: float = 0.0):
        start = time.time()
        while True:
            delay_till_next_event = self._scheduler.run(blocking=False)
            remaining = max_wait - (time.time() - start)
            if delay_till_next_event is None or delay_till_next_event > remaining:
                if remaining > 0:
                    time.sleep(remaining)
                break
            time.sleep(delay_till_next_event)


class AWGModuleEmulator:
    def __init__(self, parent_conn: ziDAQServerEmulator):
        self._parent_conn = parent_conn

    @overload
    def set(self, path: str, value: Any):
        ...

    @overload
    def set(self, items: List[List[Any]]):
        ...

    def set(
        self, path_or_items: Union[str, List[List[Any]]], value: Optional[Any] = None
    ):
        if isinstance(path_or_items, str):
            path = path_or_items
        else:
            items = path_or_items
        # TODO(2K): stub

    def get(self, path: str, flat: bool = False):
        # TODO(2K): stub
        results = {}
        for p in _canonical_path_list(path):
            if p == "/directory":
                val = "/"
            else:
                val = 0
            results[p] = [val]
        # TODO(2K): reshape results
        assert flat == True
        return results

    def getInt(self, path: str) -> int:
        return 0  # TODO(2K): stub

    def execute(self):
        pass  # TODO(2K): stub

    def progress(self) -> float:
        return 1.0  # TODO(2K): stub
