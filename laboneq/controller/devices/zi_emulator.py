# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
import re
import sched
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, overload

import numpy as np
from numpy import typing as npt
from zhinst.core import __version__

_major, _minor, DS_REVISION = __version__.split(".")
DS_VERSION = f"{_major}.{_minor}"


@dataclass
class NodeBase:
    "A class modeling a single node. Specialized for specific node types."
    read_only: bool = False
    subscribed: bool = False
    handler: Callable[[NodeBase], None] = None

    def node_value(self) -> Any:
        return {"value": [self.value]}


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
class NodeVectorBase(NodeBase):
    def node_value(self) -> Any:
        return [{"vector": self.value}]


@dataclass
class NodeVectorFloat(NodeVectorBase):
    value: npt.ArrayLike = field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclass
class NodeVectorInt(NodeVectorBase):
    value: npt.ArrayLike = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class NodeVectorComplex(NodeVectorBase):
    value: npt.ArrayLike = field(
        default_factory=lambda: np.array([], dtype=np.complex128)
    )


@dataclass
class NodeDynamic(NodeBase):
    getter: Callable[[], Any] = None
    setter: Callable[[Any], None] = None

    @property
    def value(self) -> Any:
        return self.getter()

    @value.setter
    def value(self, v: Any):
        self.setter(v)


class NodeType(Enum):
    FLOAT = NodeFloat
    INT = NodeInt
    STR = NodeStr
    VECTOR_FLOAT = NodeVectorFloat
    VECTOR_INT = NodeVectorInt
    VECTOR_COMPLEX = NodeVectorComplex
    DYNAMIC = NodeDynamic


@dataclass
class NodeInfo:
    "Node descriptor to use in node definitions."
    type: NodeType = NodeType.FLOAT
    default: Any | None = None
    read_only: bool = False
    handler: Callable[[NodeBase], None] = None
    # For DYNAMIC nodes
    getter: Callable[[], Any] = None
    setter: Callable[[Any], None] = None

    def make_node(self) -> NodeBase:
        "Constructs concrete node instance from a node descriptor."
        if self.type == NodeType.DYNAMIC:
            node = NodeType.DYNAMIC.value(
                read_only=self.read_only,
                handler=self.handler,
                getter=self.getter,
                setter=self.setter,
            )
            if self.setter is not None and self.default is not None:
                node.value = self.default
            return node

        if self.default is None:
            return self.type.value(
                read_only=self.read_only,
                handler=self.handler,
            )

        return self.type.value(
            read_only=self.read_only,
            value=self.default,
            handler=self.handler,
        )


@dataclass
class PollEvent:
    "A class representing a single poll event"
    path: str = None
    timestamp: int = None
    value: Any = None


class DevEmu(ABC):
    "Base class emulating a device, specialized per device type."

    def __init__(self, scheduler: sched.scheduler, dev_opts: dict[str, Any]):
        self._scheduler = scheduler
        self._dev_opts = dev_opts
        self._node_tree: dict[str, NodeBase] = {}
        self._poll_queue: list[PollEvent] = []
        self._total_subscribed: int = 0
        self._cached_node_def = functools.lru_cache(maxsize=None)(self._node_def)

    @abstractmethod
    def serial(self) -> str:
        ...

    @abstractmethod
    def _node_def(self) -> dict[str, NodeInfo]:
        ...

    def _full_path(self, dev_path: str) -> str:
        return f"/{self.serial().lower()}/{dev_path}"

    def _make_node(self, dev_path: str):
        node_def = self._cached_node_def()
        new_node_def = node_def.get(dev_path)
        if new_node_def is None:
            new_node_def = NodeInfo(type=NodeType.INT, default=0, read_only=False)
        new_node = new_node_def.make_node()
        return new_node

    def _get_node(self, dev_path: str) -> NodeBase:
        node = self._node_tree.get(dev_path)
        if node is None:
            node = self._make_node(dev_path)
            self._node_tree[dev_path] = node
        return node

    def get(self, dev_path: str) -> Any:
        node = self._get_node(dev_path)
        return node.node_value()

    def getString(self, dev_path: str) -> str:
        node = self._get_node(dev_path)
        return str(node.value)

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
            node.handler(self, node)

    def subscribe(self, dev_path: str):
        node = self._get_node(dev_path)
        node.subscribed = True

    def unsubscribe(self, dev_path: str):
        node = self._get_node(dev_path)
        node.subscribed = False

    def getAsEvent(self, dev_path: str):
        node = self._get_node(dev_path)
        self._poll_queue.append(
            PollEvent(path=self._full_path(dev_path), value=node.value)
        )

    def poll(self) -> list[PollEvent]:
        output = self._poll_queue[:]
        self._poll_queue.clear()
        return output


class DevEmuZI(DevEmu):
    def serial(self) -> str:
        return "ZI"

    @property
    def server(self) -> ziDAQServerEmulator:
        return self._dev_opts["emu_server"]

    def _devices_connected(self) -> str:
        devices = [
            d.serial().upper() for d in self.server._devices.values() if d != self
        ]
        return ",".join(devices)

    def _node_def(self) -> dict[str, NodeInfo]:
        return {
            "about/version": NodeInfo(
                type=NodeType.STR, default=DS_VERSION, read_only=True
            ),
            "about/revision": NodeInfo(
                type=NodeType.STR, default=DS_REVISION, read_only=True
            ),
            "about/dataserver": NodeInfo(
                type=NodeType.STR, default="Emulated", read_only=True
            ),
            "devices/connected": NodeInfo(
                type=NodeType.DYNAMIC, read_only=True, getter=self._devices_connected
            ),
        }


class DevEmuHW(DevEmu):
    def __init__(
        self, serial: str, scheduler: sched.scheduler, dev_opts: dict[str, Any]
    ):
        super().__init__(scheduler, dev_opts)
        self._serial = serial

    def serial(self) -> str:
        return self._serial


class DevEmuDummy(DevEmuHW):
    def _node_def(self) -> dict[str, NodeInfo]:
        return {}


class DevEmuHDAWG(DevEmuHW):
    def _awg_stop(self, awg_idx):
        self._set_val(f"awgs/{awg_idx}/enable", 0)

    def _awg_execute(self, node: NodeBase, awg_idx):
        self._scheduler.enter(
            delay=0.001, priority=0, action=self._awg_stop, argument=(awg_idx,)
        )

    def _sample_clock_switched(self):
        self._set_val("system/clocks/sampleclock/status", 0)

    def _sample_clock(self, node: NodeFloat):
        self._set_val("system/clocks/sampleclock/status", 2)
        self._scheduler.enter(
            delay=0.001,
            priority=0,
            action=self._sample_clock_switched,
        )

    def _ref_clock_switched(self, source):
        self._set_val("system/clocks/referenceclock/status", 0)
        # 0 -> internal (freq 100e6)
        # 1 -> external (freq 10e6)
        # 2 -> zsync (freq 100e6)
        target_freq = 10e6 if source == 1 else 100e6
        current_freq = self._get_node("system/clocks/referenceclock/freq").value
        if current_freq != target_freq:
            self._set_val("system/clocks/referenceclock/freq", target_freq)

    def _ref_clock(self, node: NodeInt):
        self._scheduler.enter(
            delay=0.001,
            priority=0,
            action=self._ref_clock_switched,
            argument=(node.value,),
        )

    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "HDAWG8"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", "MF\nME\nSKW"),
            ),
            "system/clocks/sampleclock/status": NodeInfo(type=NodeType.INT, default=0),
            "system/clocks/sampleclock/freq": NodeInfo(
                type=NodeType.FLOAT, default=2.4e9, handler=DevEmuHDAWG._sample_clock
            ),
            "system/clocks/referenceclock/source": NodeInfo(
                type=NodeType.INT, default=0, handler=DevEmuHDAWG._ref_clock
            ),
            "system/clocks/referenceclock/status": NodeInfo(
                type=NodeType.INT, default=0
            ),
            "system/clocks/referenceclock/freq": NodeInfo(
                type=NodeType.FLOAT, default=100e6
            ),
        }
        for awg_idx in range(4):
            nd[f"awgs/{awg_idx}/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuHDAWG._awg_execute, awg_idx=awg_idx),
            )
        return nd


class DevEmuUHFQA(DevEmuHW):
    """Emulated UHFQA.

    Supported emulation options:
        - user_readout_data - callable matching the following signature:
            user_readout_data(
                result_index: int,
                length: int,
                averages: int) -> ArrayLike | list[float]
            The function is called after every AWG execution, once for every integrator with the
            corresponding 'result_index'. It must return the vector of values of size 'length',
            that will be set to the corresponding '<devN>/qas/0/result/data/<result_index>/wave'
            node.
            The argument 'averages' is provided for convenience, as the real device returns the
            sum of all the averaged readouts. The function may also return None, in which case
            the emulator falls back to the default emulated results for this integrator.
    """

    def _awg_stop(self):
        self._set_val("awgs/0/enable", 0)
        result_enable = self._get_node("qas/0/result/enable").value
        monitor_enable = self._get_node("qas/0/monitor/enable").value
        if result_enable != 0:
            length = self._get_node("qas/0/result/length").value
            self._set_val("qas/0/result/acquired", 0)  # Wraps around to 0 on success
            for result_index in range(10):
                user_readout_data = self._dev_opts.get("user_readout_data")
                res = None
                if callable(user_readout_data):
                    averages = self._get_node("qas/0/result/averages").value
                    res = user_readout_data(result_index, length, averages)
                if res is None:
                    integrator = result_index // 2
                    res_c = (42 + integrator + 1j * np.arange(length)).view(float)
                    res = res_c[result_index % 2 :: 2]
                self._set_val(f"qas/0/result/data/{result_index}/wave", np.array(res))
        if monitor_enable != 0:
            length = self._get_node("qas/0/monitor/length").value
            self._set_val("qas/0/monitor/inputs/0/wave", [52] * length)
            self._set_val("qas/0/monitor/inputs/1/wave", [52] * length)

    def _awg_execute(self, node: NodeBase):
        self._scheduler.enter(delay=0.001, priority=0, action=self._awg_stop)

    def _awg_ready(self):
        self._set_val("awgs/0/ready", 1)

    def _elf_upload(self, node: NodeBase):
        self._set_val("awgs/0/ready", 0)
        self._scheduler.enter(delay=0.001, priority=0, action=self._awg_ready)

    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "UHFQA"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", "AWG\nDIG\nQA"),
            ),
            "awgs/0/enable": NodeInfo(
                type=NodeType.INT, default=0, handler=DevEmuUHFQA._awg_execute
            ),
            "awgs/0/elf/data": NodeInfo(
                type=NodeType.VECTOR_INT, default=[], handler=DevEmuUHFQA._elf_upload
            ),
            "awgs/0/ready": NodeInfo(type=NodeType.INT, default=0),
            "qas/0/monitor/inputs/0/wave": NodeInfo(type=NodeType.VECTOR_COMPLEX),
            "qas/0/monitor/inputs/1/wave": NodeInfo(type=NodeType.VECTOR_COMPLEX),
        }
        for result_index in range(10):
            nd[f"qas/0/result/data/{result_index}/wave"] = NodeInfo(
                type=NodeType.VECTOR_COMPLEX
            )
        return nd


class Gen2Base(DevEmuHW):
    def _ref_clock_switched(self, requested_source: int):
        # 0 - INTERNAL
        # 1 - EXTERNAL
        # 2 - ZSYNC
        freq = 10e6 if requested_source == 1 else 100e6
        self._set_val("system/clocks/referenceclock/in/sourceactual", requested_source)
        self._set_val("system/clocks/referenceclock/in/status", 0)
        self._set_val("system/clocks/referenceclock/in/freq", freq)

    def _ref_clock(self, node: NodeBase):
        node_int: NodeInt = node
        self._scheduler.enter(
            delay=0.001,
            priority=0,
            action=self._ref_clock_switched,
            argument=(node_int.value,),
        )

    def _node_def_gen2(self) -> dict[str, NodeInfo]:
        return {
            "system/clocks/referenceclock/in/source": NodeInfo(
                type=NodeType.INT, default=0, handler=DevEmuPQSC._ref_clock
            ),
            "system/clocks/referenceclock/in/sourceactual": NodeInfo(
                type=NodeType.INT, default=0
            ),
            "system/clocks/referenceclock/in/status": NodeInfo(
                type=NodeType.INT, default=0
            ),
            "system/clocks/referenceclock/in/freq": NodeInfo(
                type=NodeType.FLOAT, default=100e6
            ),
        }


class DevEmuPQSC(Gen2Base):
    def _trig_stop(self):
        self._set_val("execution/enable", 0)

    def _trig_execute(self, node: NodeBase):
        self._scheduler.enter(delay=0.001, priority=0, action=self._trig_stop)

    def _node_def(self) -> dict[str, NodeInfo]:
        return {
            **self._node_def_gen2(),
            "execution/enable": NodeInfo(
                type=NodeType.INT, default=0, handler=DevEmuPQSC._trig_execute
            ),
        }


class DevEmuSHFQABase(Gen2Base):
    def _awg_stop_qa(self, channel: int):
        readout_enable = self._get_node(
            f"qachannels/{channel}/readout/result/enable"
        ).value
        spectroscopy_enable = self._get_node(
            f"qachannels/{channel}/spectroscopy/result/enable"
        ).value
        scope_enable = self._get_node("scopes/0/enable").value
        self._set_val(f"qachannels/{channel}/generator/enable", 0)
        self._set_val(f"qachannels/{channel}/readout/result/enable", 0)
        self._set_val(f"qachannels/{channel}/spectroscopy/result/enable", 0)
        if readout_enable != 0:
            length = self._get_node(f"qachannels/{channel}/readout/result/length").value
            averages = self._get_node(
                f"qachannels/{channel}/readout/result/averages"
            ).value
            self._set_val(
                f"qachannels/{channel}/readout/result/acquired", length * averages
            )
            for integrator in range(16):
                self._set_val(
                    f"qachannels/{channel}/readout/result/data/{integrator}/wave",
                    42 + integrator + 1j * np.arange(length),
                )
        if spectroscopy_enable != 0:
            length = self._get_node(
                f"qachannels/{channel}/spectroscopy/result/length"
            ).value
            averages = self._get_node(
                f"qachannels/{channel}/spectroscopy/result/averages"
            ).value
            self._set_val(
                f"qachannels/{channel}/spectroscopy/result/acquired", length * averages
            )
            self._set_val(
                f"qachannels/{channel}/spectroscopy/result/data/wave",
                np.array([(42 + 42j)] * length),
            )
        if scope_enable != 0:
            # Assuming here that the scope was triggered by AWG and channels configured to capture
            # QA channels 1:1. Not emulating various trigger, input source, etc. settings!
            scope_single = self._get_node("scopes/0/single").value
            if scope_single != 0:
                self._set_val("scopes/0/enable", 0)
            length = self._get_node("scopes/0/length").value
            for scope_ch in range(4):
                self._set_val(
                    f"scopes/0/channels/{scope_ch}/wave",
                    np.array([(52 + 52j)] * length),
                )

    def _awg_execute_qa(self, node: NodeBase, channel: int):
        self._scheduler.enter(
            delay=0.001, priority=0, action=self._awg_stop_qa, argument=(channel,)
        )

    def _node_def_qa(self) -> dict[str, NodeInfo]:
        nd = {}
        for channel in range(4):
            nd[f"qachannels/{channel}/generator/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuSHFQABase._awg_execute_qa, channel=channel),
            )
            nd[f"qachannels/{channel}/readout/result/enable"] = NodeInfo(
                type=NodeType.INT, default=0
            )
            nd[f"qachannels/{channel}/spectroscopy/result/enable"] = NodeInfo(
                type=NodeType.INT, default=0
            )
            for integrator in range(16):
                nd[
                    f"qachannels/{channel}/readout/result/data/{integrator}/wave"
                ] = NodeInfo(type=NodeType.VECTOR_COMPLEX)
            nd[f"qachannels/{channel}/spectroscopy/result/data/wave"] = NodeInfo(
                type=NodeType.VECTOR_COMPLEX
            )
            for scope_ch in range(4):
                nd[f"scopes/0/channels/{scope_ch}/wave"] = NodeInfo(
                    type=NodeType.VECTOR_COMPLEX
                )
        return nd


class DevEmuSHFQA(DevEmuSHFQABase):
    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "SHFQA4"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", ""),
            ),
        }
        nd.update(self._node_def_gen2())
        nd.update(self._node_def_qa())
        return nd


class DevEmuSHFSGBase(Gen2Base):
    def _awg_stop_sg(self, channel: int):
        self._set_val(f"sgchannels/{channel}/awg/enable", 0)

    def _awg_execute_sg(self, node: NodeBase, channel: int):
        self._scheduler.enter(
            delay=0.001, priority=0, action=self._awg_stop_sg, argument=(channel,)
        )

    def _pipeliner_committed(self, channel: int):
        avail_slots: int = self._get_node(
            f"sgchannels/{channel}/pipeliner/availableslots"
        ).value
        self._set_val(f"sgchannels/{channel}/pipeliner/availableslots", avail_slots - 1)

    def _pipeliner_commit(self, node: NodeBase, channel: int):
        self._scheduler.enter(
            delay=0.001,
            priority=0,
            action=self._pipeliner_committed,
            argument=(channel,),
        )

    def _pipeliner_reset(self, node: NodeBase, channel: int):
        self._set_val(f"sgchannels/{channel}/pipeliner/status", 0)
        max_slots: int = self._get_node(
            f"sgchannels/{channel}/pipeliner/maxslots"
        ).value
        self._set_val(f"sgchannels/{channel}/pipeliner/availableslots", max_slots)

    def _pipeliner_stop(self, channel: int):
        self._set_val(f"sgchannels/{channel}/pipeliner/status", 3)

    def _pipeliner_enable(self, node: NodeBase, channel: int):
        self._set_val(f"sgchannels/{channel}/pipeliner/status", 1)
        self._scheduler.enter(
            delay=0.001,
            priority=0,
            action=self._pipeliner_stop,
            argument=(channel,),
        )

    def _node_def_sg(self) -> dict[str, NodeInfo]:
        nd = {}
        for channel in range(8):
            nd[f"sgchannels/{channel}/awg/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuSHFSGBase._awg_execute_sg, channel=channel),
            )
            nd[f"sgchannels/{channel}/pipeliner/maxslots"] = NodeInfo(
                type=NodeType.INT,
                default=1024,
                read_only=True,
            )
            nd[f"sgchannels/{channel}/pipeliner/availableslots"] = NodeInfo(
                type=NodeType.INT,
                default=1024,
            )
            nd[f"sgchannels/{channel}/pipeliner/commit"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuSHFSGBase._pipeliner_commit, channel=channel),
            )
            nd[f"sgchannels/{channel}/pipeliner/reset"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuSHFSGBase._pipeliner_reset, channel=channel),
            )
            nd[f"sgchannels/{channel}/pipeliner/status"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                read_only=True,
            )
            nd[f"sgchannels/{channel}/pipeliner/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(DevEmuSHFSGBase._pipeliner_enable, channel=channel),
            )
        return nd


class DevEmuSHFSG(DevEmuSHFSGBase):
    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "SHFSG8"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", ""),
            ),
        }
        nd.update(self._node_def_gen2())
        nd.update(self._node_def_sg())
        return nd


class DevEmuSHFQC(DevEmuSHFQABase, DevEmuSHFSGBase):
    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "SHFQC"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", "QC6CH"),
            ),
        }
        nd.update(self._node_def_gen2())
        nd.update(self._node_def_qa())
        nd.update(self._node_def_sg())
        return nd


class DevEmuNONQC(DevEmuHW):
    def _node_def(self) -> dict[str, NodeInfo]:
        return {}


def _serial_to_device_type(serial: str):
    m = re.match(pattern="DEV([0-9]+)[0-9]{3}", string=serial.upper())
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


def _canonical_path_list(path: str | list[str]) -> list[str]:
    if isinstance(path, list):
        paths = path
    else:
        paths = path.split(",")
    return paths


class ziDAQServerEmulator:
    """A class replacing the 'zhinst.core.ziDAQServer', emulating its behavior
    to the extent required by LabOne Q SW without the real DataServer/HW.
    """

    def __init__(self, host: str, port: int, api_level: int):
        if api_level is None:
            api_level = 6
        assert api_level == 6
        assert isinstance(port, int)
        super().__init__()
        self._scheduler = sched.scheduler()
        self._device_type_map: dict[str, str] = {}
        # TODO(2K): Defer "ZI" device initialization to allow passing options
        self._devices: dict[str, DevEmu] = {
            "ZI": DevEmuZI(self._scheduler, {"emu_server": self})
        }
        self._options: dict[str, dict[str, Any]] = {}

    def map_device_type(self, serial: str, type: str):
        self._device_type_map[serial.upper()] = type.upper()

    def set_option(self, serial: str, option: str, value: Any):
        dev_opts = self._options.setdefault(serial.upper(), {})
        dev_opts[option] = value

    def _device_factory(self, serial: str) -> DevEmu:
        dev_type_str = self._device_type_map.get(serial.upper())
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
        elif dev_type_str == "SHFQC":
            dev_type = DevEmuSHFQC
        elif dev_type_str == "NONQC":
            dev_type = DevEmuNONQC
        else:
            dev_type = _serial_to_device_type(serial)
        dev_opts = self._options.setdefault(serial.upper(), {})
        return dev_type(serial, self._scheduler, dev_opts)

    def _device_lookup(self, serial: str, create: bool = True) -> DevEmu:
        serial = serial.upper()
        device = self._devices.get(serial)
        if device is None and create:
            device = self._device_factory(serial)
            self._devices[serial] = device
        return device

    def _resolve_dev(self, path: str) -> tuple[list[DevEmu], str]:
        if path.startswith("/"):
            path = path[1:]
        path_parts = path.split("/")
        devices = []
        dev_path = ""
        if "*" in path_parts[0]:
            serial_pattern = re.compile(
                path_parts[0].replace("*", ".*"), flags=re.IGNORECASE
            )
            for serial, device in self._devices.items():
                if serial_pattern.match(serial):
                    devices.append(device)
            if len(path_parts) == 1 and path_parts[0] == "*":
                dev_path = "*"
        else:
            devices.append(self._device_lookup(path_parts[0]))
            dev_path = "/".join(path_parts[1:]).lower()
        return devices, dev_path

    def _resolve_paths_and_perform(
        self, path: str | list[str], handler: Callable
    ) -> dict[str, NodeBase]:
        results: dict[str, NodeBase] = {}
        for p in _canonical_path_list(path):
            devices, dev_path = self._resolve_dev(p)
            dev_path_suffix = "" if len(dev_path) == 0 else f"/{dev_path}"
            for device in devices:
                results[f"/{device.serial().lower()}{dev_path_suffix}"] = handler(
                    device, dev_path
                )
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
        assert flat is True
        # TODO(2K): emulate timestamp
        return raw_results

    def _set(self, device: DevEmu, dev_path: str, value: Any):
        device.set(dev_path, value)

    @overload
    def set(self, path: str, value: Any):
        ...

    @overload
    def set(self, items: list[list[Any]]):
        ...

    def set(self, path_or_items: str | list[list[Any]], value: Any | None = None):
        self._progress_scheduler()
        if isinstance(path_or_items, str):
            pass
            # TODO(2K): stub
        else:
            for [path, value] in path_or_items:
                self._resolve_paths_and_perform(
                    path.lower(), partial(self._set, value=value)
                )

    def getString(self, path: str) -> str:
        self._progress_scheduler()
        devices, dev_path = self._resolve_dev(path)
        assert len(devices) == 1
        return devices[0].getString(dev_path)

    def _listNodesJSON(self, device: DevEmu, dev_path: str):
        # Not implemented. Was intended mainly for the toolkit, instead toolkit itself is mocked.
        return {}

    def listNodesJSON(self, path: str, *args, **kwargs) -> str:
        self._progress_scheduler()
        results = self._resolve_paths_and_perform(path, self._listNodesJSON)
        combined_result = {}
        for r in results.values():
            combined_result.update(r)
        return json.dumps(combined_result)

    def _subscribe(self, device: DevEmu, dev_path: str):
        device.subscribe(dev_path)

    def subscribe(self, path: str | list[str]):
        self._progress_scheduler()
        self._resolve_paths_and_perform(path, self._subscribe)

    def _unsubscribe(self, device: DevEmu, dev_path: str):
        device.unsubscribe(dev_path)

    def unsubscribe(self, path: str | list[str]):
        self._progress_scheduler()
        self._resolve_paths_and_perform(path, self._unsubscribe)

    def _getAsEvent(self, device: DevEmu, dev_path: str):
        device.getAsEvent(dev_path)

    def getAsEvent(self, path: str):
        self._progress_scheduler()
        self._resolve_paths_and_perform(path, self._getAsEvent)

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
        self._progress_scheduler(wait_time=recording_time_s)
        events: list[PollEvent] = []
        for dev in self._devices.values():
            events.extend(dev.poll())
        result = {}
        # TODO(2K): reshape results
        assert flat is True
        for event in events:
            path_res = result.setdefault(event.path, {"value": []})
            path_res["value"].append(event.value)
        return result

    def _progress_scheduler(self, wait_time: float = 0.0):
        def _delay(delay: float):
            # time.sleep is not accurate for short waits, skip for delays below 10ms
            if delay > 0.01:
                time.sleep(delay)

        start = time.perf_counter()
        while True:
            delay_till_next_event = self._scheduler.run(blocking=False)
            elapsed = time.perf_counter() - start
            remaining = wait_time - elapsed
            if delay_till_next_event is None or delay_till_next_event > remaining:
                _delay(remaining)
                break
            _delay(delay_till_next_event)
