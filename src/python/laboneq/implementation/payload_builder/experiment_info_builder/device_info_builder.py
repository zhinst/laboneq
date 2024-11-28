# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Tuple

from laboneq.data.compilation_job import DeviceInfo, DeviceInfoType, FollowerInfo
from laboneq.data.execution_payload import VIRTUAL_SHFSG_UID_SUFFIX
from laboneq.data.setup_description import (
    DeviceType,
    Instrument,
    LogicalSignal,
    PortType,
    Setup,
)
from laboneq.implementation.utils import devices


def _split_shfqc(device: Instrument) -> Tuple[DeviceInfo, DeviceInfo]:
    shfqa = DeviceInfo(
        uid=device.uid,
        device_type=DeviceInfoType.SHFQA,
        reference_clock=device.reference_clock.frequency,
        reference_clock_source=device.reference_clock.source,
        is_qc=True,
    )
    shfsg = DeviceInfo(
        uid=device.uid + VIRTUAL_SHFSG_UID_SUFFIX,
        device_type=DeviceInfoType.SHFSG,
        reference_clock=device.reference_clock.frequency,
        reference_clock_source=device.reference_clock.source,
        is_qc=True,
    )
    return shfqa, shfsg


def _build_non_shfqc(device: Instrument) -> DeviceInfo:
    return DeviceInfo(
        uid=device.uid,
        device_type=DeviceInfoType(device.device_type.name.lower()),
        reference_clock=device.reference_clock.frequency,
        reference_clock_source=device.reference_clock.source,
        is_qc=False,
    )


class DeviceInfoBuilder:
    """Make `DeviceInfo` for each instrument defined in `Setup`.

    Splits SHFQC into individual SHFQA and SHFSG devices.
    """

    def __init__(self, setup: Setup):
        self._setup = setup
        self._device_mapping: dict[str, DeviceInfo] = {}
        self._device_by_ls: dict[LogicalSignal, DeviceInfo] = {}
        self._build_devices_and_connections()
        self._global_leader: DeviceInfo = self._find_global_leader()

    @property
    def device_mapping(self) -> Mapping[str, DeviceInfo]:
        return self._device_mapping

    @property
    def global_leader(self) -> DeviceInfo:
        return self._global_leader

    def device_by_ls(self, ls: LogicalSignal) -> DeviceInfo:
        return self._device_by_ls[ls]

    def _find_global_leader(self) -> DeviceInfo | None:
        for server in self._setup.servers.values():
            if server.leader_uid is not None:
                return (
                    self._device_mapping[server.leader_uid]
                    if server.leader_uid is not None
                    else None
                )

    def _build_devices_and_connections(self):
        """Build devices and assign leader - follower relationships."""
        setup_internal_connections = deepcopy(self._setup.setup_internal_connections)
        for device in self._setup.instruments:
            if device.device_type == DeviceType.UNMANAGED:
                continue
            # Split SHFQC into SHFQA / SHFSG
            if device.device_type == DeviceType.SHFQC:
                shfqa, shfsg = _split_shfqc(device)
                # Check whether Physical channel ports in connection belong to SHFSG or SHFQA
                # to make a LogicalSignal -> SHFQA/SHFSG connection
                for conn in device.connections:
                    pc_ports = [p.path for p in conn.physical_channel.ports]
                    shfqa_ports = [p.path for p in devices.shfqa_ports()]
                    if set(pc_ports).issubset(shfqa_ports):
                        self._device_by_ls[conn.logical_signal] = shfqa
                    else:
                        self._device_by_ls[conn.logical_signal] = shfsg
                if (
                    shfqa in self._device_by_ls.values()
                    or shfsg not in self._device_by_ls.values()
                ):
                    self._device_mapping[shfqa.uid] = shfqa
                else:
                    shfsg.uid = shfsg.uid.removesuffix(VIRTUAL_SHFSG_UID_SUFFIX)
                if shfsg in self._device_by_ls.values():
                    self._device_mapping[shfsg.uid] = shfsg
            else:
                self._device_mapping[device.uid] = _build_non_shfqc(device)
                for conn in device.connections:
                    self._device_by_ls[conn.logical_signal] = self._device_mapping[
                        device.uid
                    ]

        for conn in setup_internal_connections:
            if conn.from_port.type == PortType.RF:
                continue
            leader = self._device_mapping[conn.from_instrument.uid]
            follower = FollowerInfo(
                device=self._device_mapping[conn.to_instrument.uid],
                port=conn.from_port.channel,
            )
            leader.followers.append(follower)
            if conn.to_instrument.device_type == DeviceType.SHFQC:
                sg_uid_candidate = conn.to_instrument.uid + VIRTUAL_SHFSG_UID_SUFFIX
                if sg_uid_candidate not in self._device_mapping:
                    continue
                follower = FollowerInfo(
                    device=self._device_mapping[f"{conn.to_instrument.uid}_sg"],
                    port=conn.from_port.channel,
                )
                leader.followers.append(follower)
