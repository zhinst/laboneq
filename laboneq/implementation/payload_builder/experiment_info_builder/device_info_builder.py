# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from laboneq.data.compilation_job import DeviceInfo, DeviceInfoType, FollowerInfo
from laboneq.data.setup_description import DeviceType, Instrument, LogicalSignal, Setup
from laboneq.implementation.utils import devices


def _split_shfqc(device: Instrument) -> Tuple[Instrument, Instrument]:
    shfqa = DeviceInfo(
        uid=device.uid + "_shfqa",
        device_type=DeviceInfoType.SHFQA,
        reference_clock=device.reference_clock.frequency,
        reference_clock_source=device.reference_clock.source,
        is_qc=True,
    )
    shfsg = DeviceInfo(
        uid=device.uid + "_shfsg",
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
        self._device_mapping: Mapping[str, DeviceInfo] = {}
        self._device_by_ls: Mapping[LogicalSignal, DeviceInfo] = {}
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

    def _find_global_leader(self) -> Optional[DeviceInfo]:
        for server in self._setup.servers.values():
            if server.leader_uid is not None:
                return (
                    self._device_mapping[server.leader_uid]
                    if server.leader_uid is not None
                    else None
                )

    def _build_devices_and_connections(self):
        """Build devices and assign leader - follower relationships."""
        for device in self._setup.instruments:
            # Split SHFQC into SHFQA / SHFSG
            if device.device_type == DeviceType.SHFQC:
                shfqa, shfsg = _split_shfqc(device)
                self._device_mapping[shfqa.uid] = shfqa
                self._device_mapping[shfsg.uid] = shfsg
                # Check whether Physical channel ports in connection belong to SHFSG or SHFQA
                # to make a LogicalSignal -> SHFQA/SHFSG connection
                for conn in device.connections:
                    pc_ports = [p.path for p in conn.physical_channel.ports]
                    shfqa_ports = [p.path for p in devices.shfqa_ports()]
                    if set(pc_ports).issubset(shfqa_ports):
                        self._device_by_ls[conn.logical_signal] = shfqa
                    else:
                        self._device_by_ls[conn.logical_signal] = shfsg
            else:
                self._device_mapping[device.uid] = _build_non_shfqc(device)
                for conn in device.connections:
                    self._device_by_ls[conn.logical_signal] = self._device_mapping[
                        device.uid
                    ]

        for conn in self._setup.setup_internal_connections:
            leader = self._device_mapping[conn.from_instrument.uid]
            follower = FollowerInfo(
                device=self._device_mapping[conn.to_instrument.uid],
                port=conn.to_port.channel,
            )
            leader.followers.append(follower)
