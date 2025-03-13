# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from collections import defaultdict

import functools
import json
import logging
import re
import sched
import time
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from enum import Enum
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Iterator, cast, overload
from weakref import ReferenceType, ref

import numpy as np
from numpy import typing as npt


_logger = logging.getLogger(__name__)


# Option for the overrange nodes that triggers increment emulation
_INC_ON_RUN = "INC_ON_RUN"


@dataclass
class NodeBase:
    """A class modeling a single node. Specialized for specific node types."""

    read_only: bool = False
    subscribed: bool = False
    handler: Callable[[NodeBase], None] | None = None
    _value: Any = field(init=False, repr=False)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value

    def node_value(self) -> Any:
        return {"value": [self.value]}

    def call_handler(self):
        if self.handler is not None:
            self.handler(self)


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
        return [{"vector": self.value[0], "properties": self.value[1]}]


@dataclass
class NodeVectorFloat(NodeVectorBase):
    value: tuple[npt.NDArray[Any], dict[str, Any]] = field(
        default_factory=lambda: (np.array([], dtype=np.float64), {})
    )


@dataclass
class NodeVectorInt(NodeVectorBase):
    value: tuple[npt.NDArray[Any], dict[str, Any]] = field(
        default_factory=lambda: (np.array([], dtype=np.int64), {})
    )


@dataclass
class NodeVectorStr(NodeVectorBase):
    value: tuple[str, dict[str, Any]] = ("", {})


@dataclass
class NodeVectorComplex(NodeVectorBase):
    value: tuple[npt.NDArray[Any], dict[str, Any]] = field(
        default_factory=lambda: (np.array([], dtype=np.complex128), {})
    )


@dataclass
class NodeDynamic(NodeBase):
    value: InitVar[Any | None] = field(default=None)
    getter: Callable[[], Any] | None = None
    setter: Callable[[Any], None] | None = None

    def __post_init__(self, value: Any | None):
        if self.setter is not None and value is not None:
            self.value = value

    @property  # type: ignore
    def value(self) -> Any:  # noqa: F811
        return None if self.getter is None else self.getter()

    @value.setter
    def value(self, value: Any):
        if self.setter is not None:
            self.setter(value)


class NodeType(Enum):
    FLOAT = NodeFloat
    INT = NodeInt
    STR = NodeStr
    VECTOR_FLOAT = NodeVectorFloat
    VECTOR_INT = NodeVectorInt
    VECTOR_STR = NodeVectorStr
    VECTOR_COMPLEX = NodeVectorComplex
    DYNAMIC = NodeDynamic


@dataclass
class NodeInfo:
    "Node descriptor to use in node definitions."

    type: NodeType = NodeType.FLOAT
    default: Any = None
    read_only: bool = False
    handler: Callable[[NodeBase], Any] | None = None
    # For DYNAMIC nodes
    getter: Callable[[], Any] | None = None
    setter: Callable[[Any], Any] | None = None

    def make_node(self) -> NodeBase:
        "Constructs concrete node instance from a node descriptor."
        if self.type == NodeType.DYNAMIC:
            return NodeType.DYNAMIC.value(
                read_only=self.read_only,
                value=self.default,
                handler=self.handler,
                getter=self.getter,
                setter=self.setter,
            )

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

    path: str
    timestamp: int | None = None
    value: Any = None


class DevEmu(ABC):
    "Base class emulating a device, specialized per device type."

    def __init__(self, serial: str, emulator_state: EmulatorState):
        self._serial = serial
        self._emulator_state = emulator_state
        self._dev_opts = emulator_state.get_options(serial)
        self._node_tree: dict[str, NodeBase] = {}
        self._poll_queue: list[PollEvent] = []
        self._total_subscribed: int = 0
        self._cached_node_def = functools.lru_cache(maxsize=None)(self._node_def)

    def serial(self) -> str:
        return self._serial

    def schedule(self, delay, action, argument=()):
        self._emulator_state.scheduler.enter(
            delay=delay, priority=0, action=action, argument=argument
        )

    def trigger(self):  # noqa: B027
        """No trigger actions by default"""
        pass

    @abstractmethod
    def _node_def(self) -> dict[str, NodeInfo]: ...

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

    def resolve_wildcards(self, dev_path_wildcard: str) -> Iterator[str]:
        if "*" in dev_path_wildcard:
            path_pattern = re.compile(
                dev_path_wildcard.replace("*", "[^/]*"), flags=re.IGNORECASE
            )
            node_def = self._cached_node_def()
            matched = False
            for dev_path in node_def.keys():
                if path_pattern.match(dev_path):
                    matched = True
                    yield dev_path
            if not matched:
                yield dev_path_wildcard
        else:
            yield dev_path_wildcard

    def get(self, dev_path: str) -> Any:
        node = self._get_node(dev_path)
        return node.node_value()

    def getString(self, dev_path: str) -> str:
        node = self._get_node(dev_path)
        return str(node.value)

    def _set_val(
        self,
        dev_path: str,
        value: Any,
        poll_transform: Callable[[Any], Any] | None = None,
    ) -> NodeBase:
        node = self._get_node(dev_path)
        node.value = value
        if node.subscribed:
            poll_data = node.node_value()
            if poll_transform is not None:
                poll_data = poll_transform(poll_data)
            self._poll_queue.append(
                PollEvent(path=self._full_path(dev_path), value=poll_data)
            )
        return node

    def set(self, dev_path: str, value: Any):
        node = self._set_val(dev_path, value)
        node.call_handler()

    def subscribe(self, dev_path: str):
        node = self._get_node(dev_path)
        node.subscribed = True

    def unsubscribe(self, dev_path: str):
        node = self._get_node(dev_path)
        node.subscribed = False

    def getAsEvent(self, dev_path: str):
        node = self._get_node(dev_path)
        self._poll_queue.append(
            PollEvent(path=self._full_path(dev_path), value=node.node_value())
        )

    def poll(self) -> list[PollEvent]:
        output = self._poll_queue[:]
        self._poll_queue.clear()
        return output


class PipelinerEmu:
    def __init__(
        self,
        parent: DevEmu,
        pipeliner_base: str,
        pipeliner_stop_hook: Callable[[int], None] | None = None,
    ):
        self._parent_ref = ref(parent)
        self._pipeliner_base = pipeliner_base
        self._pipeliner_stop_hook = pipeliner_stop_hook
        self._staging_slot: dict[int, int] = defaultdict(lambda: 0)
        # self._pipelined[<channel>][<slot>][<path_part>] = <value>
        self._pipelined: dict[int, list[dict[str, Any]]] = defaultdict(list)

    @property
    def _parent(self) -> DevEmu:
        parent = self._parent_ref()
        assert parent is not None
        return parent

    def is_active(self, channel) -> bool:
        mode: int = self._parent._get_node(
            f"{self._pipeliner_base}/{channel}/pipeliner/mode"
        ).value
        return mode > 0

    def _pipeline(self, node: NodeBase, item: str, channel: int):
        if self.is_active(channel):
            pipelined = self._pipelined[channel]
            staging_slot = self._staging_slot[channel]
            while len(pipelined) <= staging_slot:
                pipelined.append({})
            pipelined[-1][item] = node.value

    def _pipeliner_mode(self, node: NodeBase, channel: int):
        self._staging_slot[channel] = 0
        self._pipelined[channel].clear()

    def _pipeliner_committed(self, channel: int):
        max_slots: int = self._parent._get_node(
            f"{self._pipeliner_base}/{channel}/pipeliner/maxslots"
        ).value
        self._parent._set_val(
            f"{self._pipeliner_base}/{channel}/pipeliner/availableslots",
            max_slots - self._staging_slot[channel],
        )

    def _pipeliner_commit(self, node: NodeBase, channel: int):
        self._staging_slot[channel] += 1
        self._parent.schedule(
            delay=0.001, action=self._pipeliner_committed, argument=(channel,)
        )

    def _pipeliner_reset(self, node: NodeBase, channel: int):
        self._parent._set_val(f"{self._pipeliner_base}/{channel}/pipeliner/status", 0)
        max_slots: int = self._parent._get_node(
            f"{self._pipeliner_base}/{channel}/pipeliner/maxslots"
        ).value
        self._parent._set_val(
            f"{self._pipeliner_base}/{channel}/pipeliner/availableslots", max_slots
        )
        self._staging_slot[channel] = 0
        self._pipelined[channel].clear()

    def _pipeliner_stop(self, channel: int):
        # idle
        self._parent._set_val(f"{self._pipeliner_base}/{channel}/pipeliner/status", 0)
        if self._pipeliner_stop_hook is not None:
            self._pipeliner_stop_hook(channel)

    def _pipeliner_enable(self, node: NodeBase, channel: int):
        # exec
        self._parent._set_val(f"{self._pipeliner_base}/{channel}/pipeliner/status", 1)
        self._parent.schedule(
            delay=0.001, action=self._pipeliner_stop, argument=(channel,)
        )

    def _node_def_pipeliner(self) -> dict[str, NodeInfo]:
        nd = {}
        for channel in range(8):
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/mode"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._pipeliner_mode, channel=channel),
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/maxslots"] = NodeInfo(
                type=NodeType.INT,
                default=1024,
                read_only=True,
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/availableslots"] = NodeInfo(
                type=NodeType.INT,
                default=1024,
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/commit"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._pipeliner_commit, channel=channel),
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/reset"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._pipeliner_reset, channel=channel),
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/status"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                read_only=True,
            )
            nd[f"{self._pipeliner_base}/{channel}/pipeliner/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._pipeliner_enable, channel=channel),
            )
        return nd


class DevEmuZI(DevEmu):
    def _devices_connected(self) -> str:
        server = self._dev_opts.get("emu_server")
        if isinstance(server, ziDAQServerEmulator):
            devices = [
                d.serial().upper() for d in server._devices.values() if d != self
            ]
            return ",".join(devices)
        return ""

    def _node_def(self) -> dict[str, NodeInfo]:
        return {
            "about/fullversion": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("about/fullversion", "99.99.9.9999"),
                read_only=True,
            ),
            "about/dataserver": NodeInfo(
                type=NodeType.STR, default="Emulated", read_only=True
            ),
            "devices/connected": NodeInfo(
                type=NodeType.DYNAMIC, read_only=True, getter=self._devices_connected
            ),
        }


class DevEmuHW(DevEmu):
    def _preset_loaded(self):
        self._set_val("system/preset/busy", 0)

    def _preset_load(self, node: NodeBase):
        self._set_val("system/preset/load", 0)
        self._set_val("system/preset/busy", 1)
        for p in self._node_tree.keys():
            if p not in ["system/preset/load", "system/preset/busy"]:
                node_info = self._cached_node_def().get(p)
                self._set_val(
                    p,
                    0
                    if node_info is None
                    else node_info.type.value()._value
                    if node_info.default is None
                    else node_info.default,
                )
        self.schedule(delay=0.001, action=self._preset_loaded)

    def _node_def_common(self) -> dict[str, NodeInfo]:
        return {
            # TODO(2K): Emulate errors. True response example (without whitespace):
            # {
            #     "sequence_nr": 14,
            #     "new_errors": 0,
            #     "first_timestamp": 26302580645225,
            #     "timestamp": 26302580645225,
            #     "timestamp_utc": "2023-11-07 15:38:51",
            #     "messages": [
            #         {
            #             "code": "AWGOSCMODE",
            #             "attribs": [
            #                 0
            #             ],
            #             "module": "awg",
            #             "severity": 2.0,
            #             "params": [],
            #             "cmd_id": 0,
            #             "count": 1,
            #             "timestamp": 19764071818932,
            #             "message": "The AWG program of AWG 0 tries to control oscillators but the required oscillator mode is not active!"
            #         },
            #         {
            #             "code": "AWGOSCMODE",
            #             "attribs": [
            #                 3
            #             ],
            #             "module": "awg",
            #             "severity": 2.0,
            #             "params": [],
            #             "cmd_id": 0,
            #             "count": 1,
            #             "timestamp": 19764081128705,
            #             "message": "The AWG program of AWG 3 tries to control oscillators but the required oscillator mode is not active!"
            #         },
            #         {
            #             "code": "ELFTRGMODEZS",
            #             "attribs": [
            #                 3
            #             ],
            #             "module": "awg",
            #             "severity": 2.0,
            #             "params": [],
            #             "cmd_id": 0,
            #             "count": 3,
            #             "timestamp": 19764619605502,
            #             "message": "The AWG program (AWG 3) uses one or more ZSync instructions or triggers. Set the DIO mode to 'QCCS'."
            #         }
            #     ]
            # }
            "raw/error/json/errors": NodeInfo(
                type=NodeType.VECTOR_STR,
                read_only=True,
                default=('{"messages":[]}', {}),
            ),
            "system/preset/load": NodeInfo(
                type=NodeType.INT, default=0, handler=self._preset_load
            ),
            "system/preset/busy": NodeInfo(type=NodeType.INT, default=0),
        }


class DevEmuDummy(DevEmuHW):
    def _node_def(self) -> dict[str, NodeInfo]:
        return self._node_def_common()


class DevEmuHDAWG(DevEmuHW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hd_pipeliner = PipelinerEmu(
            parent=self,
            pipeliner_base="awgs",
        )
        self._armed_awgs: set[int] = set()

    def trigger(self):
        super().trigger()
        for awg_idx in self._armed_awgs:
            self._awg_stop(awg_idx)
        self._armed_awgs.clear()

    def _awg_stop(self, awg_idx):
        self._set_val(f"awgs/{awg_idx}/enable", 0)

    def _awg_enable(self, node: NodeBase, awg_idx):
        if node.value != 0:
            ref_clk = self._get_node("system/clocks/referenceclock/source").value
            # TODO(2K): Improve logic by sensing actual AWG code for wait trigger statements
            all_devices = set(self._emulator_state._devices.keys())
            other_devices = all_devices - {"ZI", self.serial().upper()}
            has_other_devices = len(other_devices) > 0
            not_on_zsync = ref_clk != 2
            is_standalone = not_on_zsync and not has_other_devices
            is_leader = not_on_zsync and has_other_devices
            if is_standalone or is_leader:
                self.schedule(delay=0.001, action=self._awg_stop, argument=(awg_idx,))
                if is_leader:
                    self._emulator_state.send_trigger()
            else:
                self._armed_awgs.add(awg_idx)
        elif awg_idx in self._armed_awgs:
            self._armed_awgs.remove(awg_idx)

    def _sample_clock_switched(self):
        self._set_val("system/clocks/sampleclock/status", 0)

    def _sample_clock(self, node: NodeBase):
        self._set_val("system/clocks/sampleclock/status", 2)
        self.schedule(delay=0.001, action=self._sample_clock_switched)

    def _ref_clock_switched(self, source):
        self._set_val("system/clocks/referenceclock/status", 0)
        # 0 -> internal (freq 100e6)
        # 1 -> external (freq 10e6)
        # 2 -> zsync (freq 100e6)
        target_freq = 10e6 if source == 1 else 100e6
        current_freq = self._get_node("system/clocks/referenceclock/freq").value
        if current_freq != target_freq:
            self._set_val("system/clocks/referenceclock/freq", target_freq)

    def _ref_clock(self, node: NodeBase):
        self.schedule(
            delay=0.001,
            action=self._ref_clock_switched,
            argument=(cast(NodeInt, node).value,),
        )

    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            **self._node_def_common(),
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "HDAWG8"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", "MF\nME\nSKW\nPC"),
            ),
            "system/clocks/sampleclock/status": NodeInfo(type=NodeType.INT, default=0),
            "system/clocks/sampleclock/freq": NodeInfo(
                type=NodeType.FLOAT, default=2.4e9, handler=self._sample_clock
            ),
            "system/clocks/referenceclock/source": NodeInfo(
                type=NodeType.INT, default=0, handler=self._ref_clock
            ),
            "system/clocks/referenceclock/status": NodeInfo(
                type=NodeType.INT, default=0
            ),
            "system/clocks/referenceclock/freq": NodeInfo(
                type=NodeType.FLOAT, default=100e6
            ),
        }
        nd.update(self._hd_pipeliner._node_def_pipeliner())
        for awg_idx in range(4):
            nd[f"awgs/{awg_idx}/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._awg_enable, awg_idx=awg_idx),
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._armed_awg: bool = False

    def trigger(self):
        super().trigger()
        if self._armed_awg:
            self._awg_stop()
        self._armed_awg = False

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
                averages = self._get_node("qas/0/result/averages").value
                if callable(user_readout_data):
                    res = user_readout_data(result_index, length, averages)
                if res is None:
                    integrator = result_index // 2
                    res_c = (42 + integrator + 1j * np.arange(length)).view(float)
                    res = res_c[result_index % 2 :: 2]

                def _apply_averages_on_poll(v, averages):
                    return [{"vector": v[0]["vector"] / averages, "properties": {}}]

                self._set_val(
                    f"qas/0/result/data/{result_index}/wave",
                    (np.array(res), {}),
                    poll_transform=partial(_apply_averages_on_poll, averages=averages),
                )
        if monitor_enable != 0:
            length = self._get_node("qas/0/monitor/length").value
            self._set_val(
                "qas/0/monitor/inputs/0/wave", (np.arange(length, dtype=np.float64), {})
            )
            self._set_val(
                "qas/0/monitor/inputs/1/wave",
                (-np.arange(length, dtype=np.float64), {}),
            )

    def _awg_enable(self, node: NodeBase):
        self._armed_awg = node.value != 0

    def _awg_ready(self):
        self._set_val("awgs/0/ready", 1)

    def _elf_upload(self, node: NodeBase):
        self._set_val("awgs/0/ready", 0)
        self.schedule(delay=0.001, action=self._awg_ready)

    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            **self._node_def_common(),
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "UHFQA"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", "AWG\nDIG\nQA"),
            ),
            "awgs/0/enable": NodeInfo(
                type=NodeType.INT, default=0, handler=self._awg_enable
            ),
            "awgs/0/elf/data": NodeInfo(
                type=NodeType.VECTOR_INT, default=([], {}), handler=self._elf_upload
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
        source_actual = self._dev_opts.get(
            "system/clocks/referenceclock/in/sourceactual", requested_source
        )
        freq = 10e6 if source_actual == 1 else 100e6
        self._set_val("system/clocks/referenceclock/in/sourceactual", source_actual)
        self._set_val("system/clocks/referenceclock/in/status", 0)
        self._set_val("system/clocks/referenceclock/in/freq", freq)

    def _ref_clock(self, node: NodeBase):
        self.schedule(
            delay=0.001,
            action=self._ref_clock_switched,
            argument=(cast(NodeInt, node).value,),
        )

    def _node_def_gen2(self) -> dict[str, NodeInfo]:
        return {
            **self._node_def_common(),
            "system/clocks/referenceclock/in/source": NodeInfo(
                type=NodeType.INT, default=0, handler=self._ref_clock
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


class DevEmuLeader(Gen2Base):
    def __init__(self, serial: str, emulator_state: EmulatorState):
        super().__init__(serial=serial, emulator_state=emulator_state)
        self._zsync_ports: int | None = None

    def _trig_execute(self, node: NodeBase):
        self._emulator_state.send_trigger()
        self.schedule(
            delay=0.001, action=self._set_val, argument=("execution/enable", 0)
        )

    def _node_def(self) -> dict[str, NodeInfo]:
        nd = self._node_def_gen2()
        nd["execution/enable"] = NodeInfo(
            type=NodeType.INT, default=0, handler=self._trig_execute
        )
        assert self._zsync_ports is not None
        for zsync in range(self._zsync_ports):
            nd[f"zsyncs/{zsync}/connection/status"] = NodeInfo(
                type=NodeType.INT, default=2
            )
            nd[f"zsyncs/{zsync}/connection/serial"] = NodeInfo(
                type=NodeType.VECTOR_STR,
                default=(
                    self._dev_opts.get(f"zsyncs/{zsync}/connection/serial", ""),
                    {},
                ),
            )
        return nd


class DevEmuPQSC(DevEmuLeader):
    def __init__(self, serial: str, emulator_state: EmulatorState):
        super().__init__(serial=serial, emulator_state=emulator_state)
        self._zsync_ports = 18


class DevEmuQHUB(DevEmuLeader):
    def __init__(self, serial: str, emulator_state: EmulatorState):
        super().__init__(serial=serial, emulator_state=emulator_state)
        self._zsync_ports = 56


class DevEmuSHFBase(Gen2Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _int_trig_execute(self, node: NodeBase):
        if node.value != 0:
            self._emulator_state.send_trigger()
            self.schedule(
                delay=0.001,
                action=self._set_val,
                argument=("system/internaltrigger/enable", 0),
            )

    def _sw_trig_execute(self, node: NodeBase):
        if node.value != 0:
            self._emulator_state.send_trigger()
            self.schedule(
                delay=0.001,
                action=self._set_val,
                argument=("system/swtriggers/0/single", 0),
            )

    def _node_def_shf(self):
        nd = self._node_def_gen2()
        nd["system/internaltrigger/enable"] = NodeInfo(
            type=NodeType.INT, default=0, handler=self._int_trig_execute
        )
        nd["system/swtriggers/0/single"] = NodeInfo(
            type=NodeType.INT, default=0, handler=self._sw_trig_execute
        )
        return nd


class DevEmuSHFQABase(DevEmuSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qa_pipeliner = PipelinerEmu(
            parent=self,
            pipeliner_base="qachannels",
            pipeliner_stop_hook=self._pipeliner_done,
        )
        self._armed_qa_awgs: set[int] = set()
        self._scope_max_segments: int = 1024
        self._scope_memory_size: int = 256 * 1024

    def trigger(self):
        super().trigger()
        for channel in self._armed_qa_awgs:
            self._awg_stop_qa(channel)
        self._armed_qa_awgs.clear()

    def _make_measurement_properties(self, job_id=0):
        return {
            "jobid": job_id,
            # 1 ms per job, in 0.25 ns time base
            "firstSampleTimestamp": 4_000_000 * job_id,
        }

    def _side_effects_qa(self, channel: int):
        out_ovr_node = f"qachannels/{channel}/output/overrangecount"
        out_ovr_opt = self._dev_opts.get(out_ovr_node, None)
        if out_ovr_opt == _INC_ON_RUN:
            out_ovr_count = self._get_node(out_ovr_node).value
            self._set_val(out_ovr_node, out_ovr_count + 1)

        in_ovr_node = f"qachannels/{channel}/input/overrangecount"
        in_ovr_opt = self._dev_opts.get(in_ovr_node, None)
        if in_ovr_opt == _INC_ON_RUN:
            in_ovr_count = self._get_node(in_ovr_node).value
            self._set_val(in_ovr_node, in_ovr_count + 1)

    def _push_readout_result(
        self, channel: int, length: int, averages: int, job_id: int = 0
    ):
        self._set_val(
            f"qachannels/{channel}/readout/result/acquired", length * averages
        )
        for integrator in range(16):
            self._set_val(
                f"qachannels/{channel}/readout/result/data/{integrator}/wave",
                (
                    (42 + integrator + 1j * np.arange(length)) / averages,
                    self._make_measurement_properties(job_id),
                ),
            )

    def _push_spectroscopy_result(
        self, channel: int, length: int, averages: int, job_id: int = 0
    ):
        self._set_val(
            f"qachannels/{channel}/spectroscopy/result/acquired", length * averages
        )
        self._set_val(
            f"qachannels/{channel}/spectroscopy/result/data/wave",
            (
                np.array([(42 + 42j)] * length),
                self._make_measurement_properties(job_id),
            ),
        )

    def _pipeliner_done(self, channel: int):
        self._side_effects_qa(channel)
        pipelined_nodes: dict[str, Any] = {}
        for job_id, slot in enumerate(self._qa_pipeliner._pipelined[channel]):
            pipelined_nodes |= slot

            readout_enable = pipelined_nodes.get("readout/result/enable", 0)
            spectroscopy_enable = pipelined_nodes.get("spectroscopy/result/enable", 0)
            pipelined_nodes["readout/result/enable"] = 0
            pipelined_nodes["spectroscopy/result/enable"] = 0

            if readout_enable != 0:
                length = pipelined_nodes.get("readout/result/length", 0)
                # use default 1 for averages, to avoid division by zero if undefined
                averages = pipelined_nodes.get("readout/result/averages", 1)
                self._push_readout_result(channel, length, averages, job_id)

            if spectroscopy_enable != 0:
                length = pipelined_nodes.get("spectroscopy/result/length", 0)
                # use default 1 for averages, to avoid division by zero if undefined
                averages = pipelined_nodes.get("spectroscopy/result/averages", 1)
                self._push_spectroscopy_result(channel, length, averages, job_id)

    def _measurement_done(self, channel: int):
        readout_enable = self._get_node(
            f"qachannels/{channel}/readout/result/enable"
        ).value
        spectroscopy_enable = self._get_node(
            f"qachannels/{channel}/spectroscopy/result/enable"
        ).value
        self._set_val(f"qachannels/{channel}/readout/result/enable", 0)
        self._set_val(f"qachannels/{channel}/spectroscopy/result/enable", 0)

        if self._qa_pipeliner.is_active(channel):
            return

        if readout_enable != 0:
            length = self._get_node(f"qachannels/{channel}/readout/result/length").value
            averages = self._get_node(
                f"qachannels/{channel}/readout/result/averages"
            ).value
            self._push_readout_result(channel, length, averages)

        if spectroscopy_enable != 0:
            length = self._get_node(
                f"qachannels/{channel}/spectroscopy/result/length"
            ).value
            averages = self._get_node(
                f"qachannels/{channel}/spectroscopy/result/averages"
            ).value
            self._push_spectroscopy_result(channel, length, averages)

        if self._get_node("scopes/0/enable").value != 0:
            # Assuming here that the scope was triggered by AWG and channels configured to capture
            # QA channels 1:1. Not emulating various trigger, input source, etc. settings!
            scope_single = self._get_node("scopes/0/single").value
            if scope_single != 0:
                self._set_val("scopes/0/enable", 0)
            length = self._get_node("scopes/0/length").value
            enabled_channels = self._scope_enabled_channels()
            segments = self._scope_segments()

            # Emulate the scope output using structured data that allows tracking
            # of the results assignment. Each sample value is calculated as
            # <segment> * 10 + <channel> + <sample> * 1j
            segment_samples = np.arange(length) * 1j
            for scope_ch in enabled_channels:
                data = [
                    segment * 10 + scope_ch + segment_samples
                    for segment in range(segments)
                ]
                self._set_val(
                    f"scopes/0/channels/{scope_ch}/wave", (np.concatenate(data), {})
                )

    def _scope_enabled_channels(self) -> list[int]:
        return [
            scope_ch
            for scope_ch in range(4)
            if self._get_node(f"scopes/0/channels/{scope_ch}/enable").value != 0
        ]

    def _scope_segments(self) -> int:
        if self._get_node("scopes/0/segments/enable").value != 0:
            return self._get_node("scopes/0/segments/count").value
        return 1

    def _scope_adjust_segments(self, node: NodeBase):
        if self._scope_segments() > self._scope_max_segments:
            self._set_val("scopes/0/segments/count", self._scope_max_segments)
        self._scope_adjust_length(node)

    def _scope_adjust_length(self, node: NodeBase):
        enabled_channels = self._scope_enabled_channels()
        if len(enabled_channels) < 2:
            ch_split = 1
        elif len(enabled_channels) == 2:
            ch_split = 2
        else:
            ch_split = 4
        segments = self._scope_segments()
        max_length = (self._scope_memory_size // ch_split // segments) & ~0xF
        if self._get_node("scopes/0/length").value > max_length:
            self._set_val("scopes/0/length", max_length)

    def _awg_stop_qa(self, channel: int):
        self._side_effects_qa(channel)
        self._set_val(f"qachannels/{channel}/generator/enable", 0)
        self._measurement_done(channel)

    def _awg_enable_qa(self, node: NodeBase, channel: int):
        if node.value != 0:
            if not self._qa_pipeliner.is_active(channel):
                self._armed_qa_awgs.add(channel)
        elif channel in self._armed_qa_awgs:
            self._armed_qa_awgs.remove(channel)

    def _enable_result_logger(self, node: NodeBase, channel: int, spectroscopy: bool):
        if node.value:
            spectroscopy_mode_enabled = self.get(f"qachannels/{channel}/mode")[
                "value"
            ] == [0]

            if spectroscopy:
                path = f"qachannels/{channel}/spectroscopy/result/enable"
            else:
                path = f"qachannels/{channel}/readout/result/enable"

            if spectroscopy ^ spectroscopy_mode_enabled:
                # the user attempted to enable the spectroscopy result logger while the
                # instrument is in readout mode (or vice-versa).
                current_mode_str = (
                    "spectroscopy" if spectroscopy_mode_enabled else "readout"
                )
                _logger.warning(
                    "Attempting to set /%s/%s to %d while the instrument is in %s mode.",
                    self.serial(),
                    path,
                    node.value,
                    current_mode_str,
                )
                self.set(path, 0)

        # the node is also pipelined
        self._qa_pipeliner._pipeline(
            node,
            item="spectroscopy/result/enable"
            if spectroscopy
            else "readout/result/enable",
            channel=channel,
        )

    def _node_def_qa(self) -> dict[str, NodeInfo]:
        nd = self._qa_pipeliner._node_def_pipeliner()
        for channel in range(4):
            nd[f"qachannels/{channel}/mode"] = NodeInfo(type=NodeType.INT, default=0)
            nd[f"qachannels/{channel}/generator/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._awg_enable_qa, channel=channel),
            )
            for path_part in [
                "readout/result/length",
                "readout/result/averages",
                "spectroscopy/result/length",
                "spectroscopy/result/averages",
            ]:
                nd[f"qachannels/{channel}/{path_part}"] = NodeInfo(
                    type=NodeType.INT,
                    default=0,
                    handler=partial(
                        self._qa_pipeliner._pipeline, item=path_part, channel=channel
                    ),
                )

            nd[f"qachannels/{channel}/spectroscopy/result/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(
                    self._enable_result_logger, channel=channel, spectroscopy=True
                ),
            )
            nd[f"qachannels/{channel}/readout/result/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(
                    self._enable_result_logger, channel=channel, spectroscopy=False
                ),
            )
            nd[f"qachannels/{channel}/output/overrangecount"] = NodeInfo(
                type=NodeType.INT, default=0
            )
            nd[f"qachannels/{channel}/input/overrangecount"] = NodeInfo(
                type=NodeType.INT, default=0
            )

            for integrator in range(16):
                nd[f"qachannels/{channel}/readout/result/data/{integrator}/wave"] = (
                    NodeInfo(type=NodeType.VECTOR_COMPLEX)
                )
            nd[f"qachannels/{channel}/spectroscopy/result/data/wave"] = NodeInfo(
                type=NodeType.VECTOR_COMPLEX
            )
            nd["scopes/0/length"] = NodeInfo(
                type=NodeType.INT, default=4992, handler=self._scope_adjust_length
            )
            nd["scopes/0/segments/enable"] = NodeInfo(
                type=NodeType.INT, default=0, handler=self._scope_adjust_length
            )
            nd["scopes/0/segments/count"] = NodeInfo(
                type=NodeType.INT, default=1, handler=self._scope_adjust_segments
            )
            for scope_ch in range(4):
                nd[f"scopes/0/channels/{scope_ch}/enable"] = NodeInfo(
                    type=NodeType.INT,
                    default=1 if scope_ch == 0 else 0,
                    handler=self._scope_adjust_length,
                )
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
        nd.update(self._node_def_shf())
        nd.update(self._node_def_qa())
        return nd


class DevEmuSHFSGBase(DevEmuSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sg_pipeliner = PipelinerEmu(
            parent=self,
            pipeliner_base="sgchannels",
            pipeliner_stop_hook=self._pipeliner_done,
        )
        self._armed_sg_awgs: set[int] = set()

    def trigger(self):
        super().trigger()
        for channel in self._armed_sg_awgs:
            self._awg_stop_sg(channel)
        self._armed_sg_awgs.clear()

    def _side_effects_sg(self, channel: int):
        out_ovr_node = f"sgchannels/{channel}/output/overrangecount"
        out_ovr_opt = self._dev_opts.get(out_ovr_node, None)
        if out_ovr_opt == _INC_ON_RUN:
            out_ovr_count = self._get_node(out_ovr_node).value
            self._set_val(out_ovr_node, out_ovr_count + 1)

    def _pipeliner_done(self, channel: int):
        self._side_effects_sg(channel)

    def _awg_stop_sg(self, channel: int):
        self._side_effects_sg(channel)
        self._set_val(f"sgchannels/{channel}/awg/enable", 0)

    def _awg_enable_sg(self, node: NodeBase, channel: int):
        if node.value != 0:
            self._armed_sg_awgs.add(channel)
        elif channel in self._armed_sg_awgs:
            self._armed_sg_awgs.remove(channel)

    def _node_def_sg(self) -> dict[str, NodeInfo]:
        nd = self._sg_pipeliner._node_def_pipeliner()
        for channel in range(8):
            nd[f"sgchannels/{channel}/awg/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._awg_enable_sg, channel=channel),
            )
            nd[f"sgchannels/{channel}/output/overrangecount"] = NodeInfo(
                type=NodeType.INT, default=0
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
        nd.update(self._node_def_shf())
        nd.update(self._node_def_sg())
        return nd


class DevEmuSHFQC(DevEmuSHFQABase, DevEmuSHFSGBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scope_memory_size = 64 * 1024

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
        nd.update(self._node_def_shf())
        nd.update(self._node_def_qa())
        nd.update(self._node_def_sg())
        return nd


class DevEmuSHFPPC(DevEmu):
    def _node_def(self) -> dict[str, NodeInfo]:
        nd = {
            "raw/error/json/errors": NodeInfo(
                type=NodeType.VECTOR_STR,
                read_only=True,
                default=('{"messages":[]}', {}),
            ),
            "features/devtype": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/devtype", "SHFPPC4"),
            ),
            "features/options": NodeInfo(
                type=NodeType.STR,
                default=self._dev_opts.get("features/options", ""),
            ),
        }

        for sweeper_idx in range(4):
            nd[f"ppchannels/{sweeper_idx}/sweeper/enable"] = NodeInfo(
                type=NodeType.INT,
                default=0,
                handler=partial(self._sweeper_start, sweeper_idx=sweeper_idx),
            )

        return nd

    def _sweeper_stop(self, sweeper_idx):
        self._set_val(f"ppchannels/{sweeper_idx}/sweeper/enable", 0)

    def _sweeper_start(self, node: NodeBase, sweeper_idx):
        if node.value == 1:
            self.schedule(
                delay=0.001, action=self._sweeper_stop, argument=(sweeper_idx,)
            )


class DevEmuNONQC(DevEmuHW):
    def _node_def(self) -> dict[str, NodeInfo]:
        return self._node_def_common()


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
        elif num >= 24 and num < 25:  # QHUB 24000 ... 24999
            return DevEmuQHUB
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


_dev_type_map: dict[str | None, type[DevEmu]] = {
    "ZI": DevEmuZI,
    "HDAWG": DevEmuHDAWG,
    "UHFQA": DevEmuUHFQA,
    "PQSC": DevEmuPQSC,
    "QHUB": DevEmuQHUB,
    "SHFQA": DevEmuSHFQA,
    "SHFSG": DevEmuSHFSG,
    "SHFQC": DevEmuSHFQC,
    "SHFPPC": DevEmuSHFPPC,
    "NONQC": DevEmuNONQC,
}


class EmulatorState:
    def __init__(self):
        self._dev_type_by_serial: dict[str, str] = {"ZI": "ZI"}
        self._options: dict[str, dict[str, Any]] = defaultdict(dict)
        self._scheduler = sched.scheduler()
        self._devices: dict[str, ReferenceType[DevEmu]] = {}
        self._events: dict[str, dict[str, list[PollEvent]]] = {}

    @property
    def scheduler(self) -> sched.scheduler:
        return self._scheduler

    def make_device(self, serial: str) -> tuple[DevEmu, dict[str, list[PollEvent]]]:
        dev_type = _dev_type_map.get(self.get_device_type(serial), DevEmuNONQC)
        # if dev_type is None:
        #     dev_type = _serial_to_device_type(serial)
        device = self.get_device_by_serial(serial)
        events: dict[str, list[PollEvent]]
        if device is None:
            device = dev_type(serial=serial, emulator_state=self)
            events = defaultdict(list)
            self._devices[serial] = ref(device)
            self._events[serial] = events
        else:
            assert isinstance(device, dev_type)
            assert device.serial() == serial
            events = self._events[serial]
        return device, events

    def get_device_by_serial(self, serial: str) -> DevEmu | None:
        dev_ref = self._devices.get(serial)
        if dev_ref is None:
            return None
        device = dev_ref()
        if device is None:
            self._devices.pop(serial)
        return device

    def send_trigger(self):
        alive = {}
        for serial, dev_ref in self._devices.items():
            dev = dev_ref()
            if dev is None:
                continue
            alive[serial] = dev_ref
            dev.trigger()
        if len(alive) < len(self._devices):
            self._devices = alive

    def map_device_type(self, serial: str, type: str):
        self._dev_type_by_serial[serial.upper()] = type.upper()

    def get_device_type(self, serial: str) -> str | None:
        return self._dev_type_by_serial.get(serial.upper())

    def set_option(self, serial: str, option: str, value: Any):
        self._options[serial.upper()][option] = value

    def get_options(self, serial: str) -> dict[str, Any]:
        return self._options[serial.upper()]


class ziDAQServerEmulator:
    """A class replacing the 'zhinst.core.ziDAQServer', emulating its behavior
    to the extent required by LabOne Q SW without the real DataServer/HW.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_level: int,
        emulator_state: EmulatorState | None = None,
    ):
        if emulator_state is None:
            emulator_state = EmulatorState()
        self._emulator_state = emulator_state
        self._emulator_state.set_option("ZI", "emu_server", self)
        if api_level is None:
            api_level = 6
        assert api_level == 6
        assert isinstance(port, int)
        super().__init__()
        self._devices: dict[str, DevEmu] = {}

    def _device_lookup(self, serial: str, create: bool = True) -> DevEmu | None:
        serial = serial.upper()
        device = self._devices.get(serial)
        if device is None and create:
            device, _ = self._emulator_state.make_device(serial)
            self._devices[serial] = device
        return device

    def _resolve_dev(self, path: str) -> tuple[list[DevEmu], str]:
        if path.startswith("/"):
            path = path[1:]
        path_parts = path.split("/")
        devices = []
        dev_path_wildcard = ""
        if "*" in path_parts[0]:
            serial_pattern = re.compile(
                path_parts[0].replace("*", "[^/]*"), flags=re.IGNORECASE
            )
            for serial, device in self._devices.items():
                if serial_pattern.match(serial):
                    devices.append(device)
            if len(path_parts) == 1 and path_parts[0] == "*":
                dev_path_wildcard = "*"
        else:
            path_dev = self._device_lookup(path_parts[0])
            assert path_dev is not None
            devices.append(path_dev)
            dev_path_wildcard = "/".join(path_parts[1:]).lower()
        return devices, dev_path_wildcard

    def _resolve_paths_and_perform(
        self, path: str | list[str], handler: Callable[[DevEmu, str], Any]
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for p in _canonical_path_list(path):
            devices, dev_path_wildcard = self._resolve_dev(p)
            for device in devices:
                for dev_path in device.resolve_wildcards(dev_path_wildcard):
                    dev_path_suffix = (
                        "" if len(dev_path_wildcard) == 0 else f"/{dev_path}"
                    )
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
    def set(self, path: str, value: Any, /): ...

    @overload
    def set(self, items: list[list[Any]], /): ...

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

    def _listNodesJSON(self, device: DevEmu, dev_path: str) -> dict:
        # Not implemented. Was intended mainly for the toolkit, instead toolkit itself is mocked.
        return {}

    def listNodesJSON(self, path: str, *args, **kwargs) -> str:
        self._progress_scheduler()
        results = self._resolve_paths_and_perform(path, self._listNodesJSON)
        combined_result = {}
        for r in results.values():
            combined_result.update(cast(dict, r))
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
    ) -> dict[str, Any]:
        self._progress_scheduler(wait_time=recording_time_s)
        events: list[PollEvent] = []
        for dev in self._devices.values():
            events.extend(dev.poll())
        result: dict[str, Any] = {}
        # TODO(2K): reshape results
        assert flat is True
        for event in events:
            if "value" in event.value:
                path_res = result.setdefault(event.path, {"value": []})
                path_res["value"].extend(event.value["value"])
            else:
                path_res = result.setdefault(event.path, [])
                path_res.extend(event.value)
        return result

    def _progress_scheduler(self, wait_time: float = 0.0):
        def _delay(delay: float):
            # time.sleep is not accurate for short waits, skip for delays below 10ms
            if delay > 0.01:
                time.sleep(delay)

        start = time.perf_counter()
        while True:
            delay_till_next_event = self._emulator_state.scheduler.run(blocking=False)
            elapsed = time.perf_counter() - start
            remaining = wait_time - elapsed
            if delay_till_next_event is None or delay_till_next_event > remaining:
                _delay(remaining)
                break
            _delay(delay_till_next_event)


_node_logger = logging.getLogger("node.log")


@dataclass
class MockAnnotatedValue:
    path: str
    value: Any
    cache: bool = False
    filename: str | None = None


def make_annotated_value(path: str, value: Any) -> MockAnnotatedValue:
    if isinstance(value, dict) and "value" in value:
        effective_value = value["value"][-1]
    elif isinstance(value, list):
        effective_value = SimpleNamespace(
            vector=value[0]["vector"],
            properties=value[0].get("properties"),
        )
    else:
        effective_value = value
    return MockAnnotatedValue(path=path, value=effective_value)


class MockDataQueue:
    def __init__(self, path: str, kernel_session: KernelSessionEmulator):
        self._path = path
        self._kernel_session = kernel_session
        self._path_events = self._kernel_session._events[self._path]

    def empty(self) -> bool:
        self._kernel_session._poll()
        return len(self._path_events) == 0

    async def get(self) -> MockAnnotatedValue:
        while self.empty():
            await asyncio.sleep(0.01)
        return self._get()

    def get_nowait(self) -> MockAnnotatedValue:
        if self.empty():
            raise asyncio.QueueEmpty
        return self._get()

    def _get(self) -> MockAnnotatedValue:
        value = self._path_events.pop(0)
        return make_annotated_value(path=self._path, value=value)

    def disconnect(self):
        pass


DUMP_TO_NODE_LOGGER_FROM_EMULATOR = False
ASYNC_EMULATE_CACHE = False


class KernelSessionEmulator:
    use_filenames_for_blobs: bool = True

    def __init__(self, serial: str, emulator_state: EmulatorState):
        self._emulator_state = emulator_state
        self._device, self._events = emulator_state.make_device(serial)
        self._cache: dict[str, Any] = {}

    def clear_cache(self):  # TODO(2K): Remove once legacy API is gone
        self._cache.clear()

    def dev_path(self, path: str) -> str:
        if path.startswith("/"):
            path = path[1:]
        path = path.lower()
        serial = self._device.serial().lower()
        if path.startswith(f"{serial}/"):
            return path[len(serial) + 1 :]
        return path

    def _progress_scheduler(self):
        self._emulator_state.scheduler.run(blocking=False)

    async def list_nodes(
        self,
        path: str = "",
        *,
        flags: int = 2,  # ABSOLUTE
    ) -> list[str]:
        self._progress_scheduler()
        return []

    def _log_to_node(self, method: str, path: str, value: Any):
        zi_dev = self._emulator_state.get_device_by_serial("ZI")
        if zi_dev is not None:
            if value is None:
                log_value = ""
            elif isinstance(value, (int, float)):
                log_value = f" value={value}"
            elif isinstance(value, str):
                if path.endswith("/data"):
                    log_value = ' value="vector (?? B)"'
                else:
                    log_value = f" value={value}"
            elif isinstance(value, (np.ndarray, bytes)):
                log_value = ' value="vector (?? B)"'
            else:
                log_value = ' value="<unexpected type>"'
            zi_dev.set(
                "debug/log",
                f'tracer="blocks_out" path="{path}" method="{method}"{log_value}',
            )

    def _log_for_testing(self, value: MockAnnotatedValue):
        def _equal(cached, actual) -> bool:
            if isinstance(actual, np.ndarray):
                return np.array_equal(cached, actual)
            else:
                return cached == actual

        # Log node set for tests, mimic old cache behaviour
        # TODO(2K): Use some better mechanism than logger, refrain from cache mimicking.
        if self.use_filenames_for_blobs or isinstance(value.value, bytes):
            effective_value = value.value if value.filename is None else value.filename
        else:
            effective_value = value.value
        if (
            ASYNC_EMULATE_CACHE
            and value.cache
            and _equal(self._cache.get(value.path), effective_value)
        ):
            effective_value = None

        if effective_value is not None:
            if "*" in value.path:
                pattern = re.compile(value.path.replace("*", "[^/]*"))
                for k in self._cache:
                    if pattern.fullmatch(k):
                        self._cache[k] = effective_value
            else:
                self._cache[value.path] = effective_value

        if isinstance(effective_value, (int, float, complex, str)):
            log_repr = f"{effective_value}"
        elif isinstance(effective_value, np.ndarray):
            array_repr = np.array2string(
                effective_value,
                threshold=30,
                max_line_width=1000,
                floatmode="maxprec",
                precision=3,
                edgeitems=16,
            )
            if "..." in array_repr:
                log_repr = f"array({array_repr}, shape={effective_value.shape})"
            else:
                log_repr = array_repr
        elif effective_value is not None:
            log_repr = f"<value of type {type(effective_value)}>"
        else:
            log_repr = None
        if log_repr is not None:
            if DUMP_TO_NODE_LOGGER_FROM_EMULATOR:
                _node_logger.debug(
                    f"set {value.path} {log_repr}", extra={"node_value": value.value}
                )
            self._log_to_node("set", value.path, value.value)

    async def set(self, value: MockAnnotatedValue) -> MockAnnotatedValue:
        self._progress_scheduler()
        self._log_for_testing(value)
        self._device.set(self.dev_path(value.path), value.value)
        return value

    async def set_with_expression(
        self, value: MockAnnotatedValue
    ) -> list[MockAnnotatedValue]:
        self._progress_scheduler()
        self._log_for_testing(value)
        dev_path_wildcard = self.dev_path(value.path)
        for dev_path in self._device.resolve_wildcards(dev_path_wildcard):
            self._device.set(dev_path, value.value)
        return []

    async def get(
        self,
        path: str,
    ) -> MockAnnotatedValue:
        self._progress_scheduler()
        if DUMP_TO_NODE_LOGGER_FROM_EMULATOR:
            _node_logger.debug(f"get {path} -")
        self._log_to_node("get", path, None)
        value = self._device.get(self.dev_path(path))
        return make_annotated_value(path=path, value=value)

    def _poll(self):
        self._progress_scheduler()
        events = self._device.poll()
        for ev in events:
            self._events[ev.path].append(ev.value)

    async def subscribe(self, path: str, get_initial_value: bool = False, **kwargs):
        self._progress_scheduler()
        dev_path = self.dev_path(path)
        self._device.subscribe(dev_path)
        if get_initial_value:
            self._device.getAsEvent(dev_path)
        return MockDataQueue(path, self)
