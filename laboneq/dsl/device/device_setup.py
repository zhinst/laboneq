# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Union

from laboneq.core import path as qct_path
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibratable, Calibration, CalibrationItem
from laboneq.dsl.device import _device_setup_modifier as setup_modifier
from laboneq.dsl.device._device_setup_modifier import DeviceSetupInternalException
from laboneq.dsl.device.connection import InternalConnection, SignalConnection
from laboneq.dsl.device.instruments import PQSC
from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
from laboneq.dsl.device.physical_channel_group import PhysicalChannelGroup
from laboneq.dsl.device.servers import DataServer
from laboneq.dsl.serialization import Serializer

from ._device_setup_generator import _DeviceSetupGenerator

if TYPE_CHECKING:
    from laboneq.dsl import quantum
    from laboneq.dsl.device.logical_signal_group import LogicalSignal

    from ._device_setup_generator import (
        ConnectionsType,
        DataServersType,
        InstrumentsType,
    )
    from .instrument import Instrument


@classformatter
@dataclass(init=True, repr=True, order=True)
class DeviceSetup:
    """Data object describing the device setup of a QCCS system.

    Attributes:
        uid (Optional[str]): Unique identifier of the setup.
        servers (Optional[dict[str, DataServer]]): Servers of the device setup.
        instruments (Optional[list[Instrument]]): Instruments of the device setup.
            Instruments must have unique UIDs.
        physical_channel_groups (Optional[dict[str, PhysicalChannelGroup]]): Physical channels of this device setup, by name of the device.
        logical_signal_groups (Optional[dict[str, LogicalSignalGroup]]): Logical signal groups of this device setup, by name of the group.
        qubits (Optional[dict[str, quantum.QuantumElement]]): Experimental: Qubits of this device setup, by the name of the qubit.
            Qubits are generated from the descriptor `qubits` section.

    !!! version-changed "Changed in version 2.19.0"
        `DeviceSetup` can now be created by using the following methods:

            - `DeviceSetup.add_dataserver()`
            - `DeviceSetup.add_instruments()`
            - `DeviceSetup.add_connections()`
    """

    uid: str = field(default="unknown")
    servers: Dict[str, DataServer] = field(default_factory=dict)
    instruments: List[Instrument] = field(default_factory=list)
    physical_channel_groups: Dict[str, PhysicalChannelGroup] = field(
        default_factory=dict
    )
    logical_signal_groups: Dict[str, LogicalSignalGroup] = field(default_factory=dict)
    qubits: Dict[str, "quantum.QuantumElement"] = field(default_factory=dict)

    def add_dataserver(
        self, host: str, port: int | str, uid: str = "zi_server", api_level: int = 6
    ):
        """Add a dataserver to the DeviceSetup.

        Args:
            host: Hostname of the dataserver.
            port: Port of the dataserver.
            uid: UID of the dataserver.
            api_level: API level of the dataserver.

        Raises:
            LabOneQException: Dataserver already exists.

        !!! version-added "Added in version 2.19.0"
        """
        try:
            setup_modifier.add_dataserver(
                self, DataServer(uid=uid, host=host, port=port, api_level=api_level)
            )
        except DeviceSetupInternalException as e:
            raise LabOneQException(str(e)) from e

    def add_instruments(self, *instruments: Instrument):
        """Add instruments to the device setup.

        Instruments must have an unique UID and Zurich Instruments instruments
        must have an address.

        At least one dataserver must be defined and if only one dataserver exists,
        instruments are automatically connected to it, otherwise the respective dataserver UID
        must be defined in the instrument class itself.

        Args:
            instruments: Instruments to add to the setup.

        Raises:
            LabOneQException:
                - If an instrument with the same UID already exists.
                - No dataservers are defined in the setup.
                - Instrument is missing `uid` or `address`.

        !!! version-added "Added in version 2.19.0"
        """
        try:
            for instrument in instruments:
                setup_modifier.add_instrument(self, instrument)
        except DeviceSetupInternalException as e:
            raise LabOneQException(str(e)) from e

    def add_connections(
        self,
        instrument: str,
        *connections: SignalConnection | InternalConnection,
    ):
        """Add connections to the instrument.

        Instrument ports cannot have two different type of signals.
        Signal names must be unique.

        Args:
            instrument: UID of the instrument to add the connections to.
            connections: Connections to add.

        Raises:
            LabOneQException: Connection information is wrong or the given
                instrument does not support the connection.

        !!! version-added "Added in version 2.19.0"
        """
        try:
            for connection in connections:
                setup_modifier.add_connection(self, instrument, connection)
        except DeviceSetupInternalException as e:
            raise LabOneQException(str(e)) from e

    def instrument_by_uid(self, uid: str) -> Instrument | None:
        return next((i for i in self.instruments if i.uid == uid), None)

    def logical_signal_by_uid(self, uid: str) -> LogicalSignal:
        """Get logical signal by uid.

        Args:
            uid: UID of the signal.
        Returns:
            Logical signal with the UID.
        Raises:
            KeyError: Logical signal UID was not found.

        !!! version-added "Added in version 2.5.0"
        """
        for grp in self.logical_signal_groups.values():
            for sig in grp.logical_signals.values():
                if uid == sig.uid:
                    return sig
        raise KeyError(f"Logical signal UID '{uid}' not found.")

    def _set_calibration(
        self,
        calibration_item: CalibrationItem,
        root_collection: Dict[str, Any],
        path_elements: List[str],
        path: str,
    ):
        if calibration_item is None:
            return
        group_name = path_elements.pop(0)
        calibratable_name = path_elements.pop(0)
        group = root_collection[group_name]
        if isinstance(group, LogicalSignalGroup):
            calibratable = group.logical_signals[calibratable_name]
        elif isinstance(group, PhysicalChannelGroup):
            calibratable = group.channels[calibratable_name]
        else:
            raise LabOneQException(f"No calibratable item found at {path}")
        if calibratable is not None:
            calibratable.calibration = calibration_item

    def set_calibration(self, calibration: Calibration):
        """Set the calibration of the device setup.

        Args:
            calibration (Calibration): Calibration object containing the keys of the individual settings.
        """
        for path in calibration.calibration_items.keys():
            path_elements = path.split(qct_path.Separator)
            if len(path_elements) > 0 and path_elements[0] == "":
                path_elements.pop(0)
            top_level_element = path_elements.pop(0)

            if top_level_element == qct_path.Instruments_Path:
                target = self.instruments
            elif top_level_element == qct_path.LogicalSignalGroups_Path:
                target = self.logical_signal_groups
            elif top_level_element == qct_path.PhysicalChannelGroups_Path:
                target = self.physical_channel_groups
            else:
                continue

            self._set_calibration(
                calibration.calibration_items[path], target, path_elements, path
            )

    def _get_instrument_calibration(self, rel_path_stack, orig_path):
        if len(rel_path_stack) >= 1:
            if rel_path_stack[0] not in self.instruments:
                raise LabOneQException(
                    f"No instrument found with id {rel_path_stack[0]}."
                )
            addressed_item = self.instruments[rel_path_stack[0]]
        else:
            raise ValueError(f"Invalid path: {rel_path_stack}")

        if not isinstance(addressed_item, Calibratable):
            raise LabOneQException(f"Not a calibratable item at {orig_path}.")

        return addressed_item.calibration

    def _get_logical_signal_calibration(self, rel_path_stack, orig_path):
        if len(rel_path_stack) != 2:
            raise LabOneQException(f"No calibration found at {orig_path}.")

        ls_group, logical_signal = rel_path_stack

        if ls_group not in self.logical_signal_groups:
            raise LabOneQException(
                f"No logical signal group found with id {logical_signal}."
            )
        if (
            logical_signal
            not in self.logical_signal_groups[ls_group].logical_signals.keys()
        ):
            raise LabOneQException(
                f"No logical signal found with id {logical_signal} in group {ls_group}."
            )
        addressed_item = self.logical_signal_groups[ls_group].logical_signals[
            logical_signal
        ]

        return addressed_item.calibration

    def _get_phyiscal_channel_calibration(self, rel_path_stack, orig_path):
        if len(rel_path_stack) != 2:
            raise LabOneQException(f"No calibration found at {orig_path}.")

        pc_group, physical_channel = rel_path_stack

        if pc_group not in self.physical_channel_groups:
            raise LabOneQException(
                f"No physical channel group found with id {pc_group}."
            )
        if (
            physical_channel
            not in self.physical_channel_groups[pc_group].channels.keys()
        ):
            raise LabOneQException(
                f"No physical channel found with id {physical_channel} in group {pc_group}."
            )
        addressed_item = self.physical_channel_groups[pc_group].channels[
            physical_channel
        ]

        return addressed_item.calibration

    def _get_calibration(self, path):
        path_stack = qct_path.split(path)
        if len(path_stack) == 0:
            raise LabOneQException(f"No calibration found at {path}.")

        root, *rest = path_stack
        if root == qct_path.Instruments_Path:
            return self._get_instrument_calibration(rest, path)
        if root == qct_path.LogicalSignalGroups_Path:
            return self._get_logical_signal_calibration(rest, path)
        if root == qct_path.PhysicalChannelGroups_Path:
            return self._get_phyiscal_channel_calibration(rest, path)
        raise LabOneQException(f"No calibration found at {path}.")

    def get_calibration(self, path: str | None = None) -> Calibration:
        """Retrieve the calibration of a specific path.

        Args:
            path (str):
                Path of the calibration information.
                If `path` is not given, full calibration is returned.

        Returns:
            calibration:
                Calibration object of the device setup.
        """
        if path is not None:
            return self._get_calibration(path)

        lsgs_calibration = dict()
        for lsg in self.logical_signal_groups.values():
            lsgs_calibration = {**lsgs_calibration, **lsg.get_calibration()}

        pcgs_calibration = dict()
        for pcg in self.physical_channel_groups.values():
            pcgs_calibration = {**pcgs_calibration, **pcg.get_calibration()}

        calibration = Calibration(
            calibration_items={**lsgs_calibration, **pcgs_calibration}
        )
        return calibration

    def reset_calibration(self, calibration: Calibration | None = None):
        """Reset the calibration of all logical signals and instruments."""
        for logical_signal_group in self.logical_signal_groups.values():
            logical_signal_group.reset_calibration()
        for physical_channel in self.physical_channel_groups.values():
            physical_channel.reset_calibration()
        if calibration:
            self.set_calibration(calibration)

    def list_calibratables(self):
        calibratables = dict()
        for logical_signal_group in self.logical_signal_groups.values():
            calibratables = {
                **calibratables,
                **logical_signal_group.list_calibratables(
                    qct_path.LogicalSignalGroups_Path_Abs
                ),
            }
        for physical_channel in self.physical_channel_groups.values():
            calibratables = {
                **calibratables,
                **physical_channel.list_calibratables(
                    qct_path.PhysicalChannelGroups_Path_Abs
                ),
            }
        return calibratables

    @classmethod
    def load(cls, filename: str) -> DeviceSetup:
        """Load the device setup from a specified file.

        Args:
            filename (str): Filename.
        """
        # TODO ErC: Error handling
        return Serializer.from_json_file(filename, cls)

    def save(self, filename: str):
        """Save the device setup to a specified file.

        Args:
            filename (str): Filename.
        """
        # TODO ErC: Error handling
        Serializer.to_json_file(self, filename)

    def dumps(self) -> str:
        """Serialize object into a JSON string."""
        # TODO ErC: Error handling
        return Serializer.to_json(self, omit_none_fields=True)

    @classmethod
    def from_descriptor(
        cls,
        yaml_text: str,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ) -> DeviceSetup:
        """Construct the device setup from a YAML descriptor.

        Args:
            yaml_text (str): YAML file containing the device description.
            server_host (str): Server host of the setup that should be created.
            server_port (str): Port of the server that should be created.
            setup_name (str): Name of the setup that should be created.
        """
        return _DeviceSetupGenerator.from_descriptor(
            yaml_text, server_host, server_port, setup_name
        )

    @classmethod
    def from_yaml(
        cls,
        filepath,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ) -> DeviceSetup:
        """Construct the device setup from a YAML file.

        Args:
            filepath (str): Path to the YAML file containing the device description.
            server_host (str): Server host of the setup that should be created.
            server_port (str): Port of the server that should be created.
            setup_name (str): Name of the setup that should be created.
        """
        return _DeviceSetupGenerator.from_yaml(
            filepath, server_host, server_port, setup_name
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        server_host: str | None = None,
        server_port: Union[str, int] | None = None,
        setup_name: str | None = None,
    ) -> DeviceSetup:
        """Construct the device setup from a Python dictionary.

        Args:
            data: Device setup data.
            server_host: Server host of the setup that should be created.
            server_port: Port of the server that should be created.
            setup_name: Name of the setup that should be created.

        !!! version-added "Added in version 2.5.0"
        """
        return _DeviceSetupGenerator.from_dicts(
            instrument_list=data.get("instrument_list"),
            instruments=data.get("instruments"),
            connections=data.get("connections"),
            dataservers=data.get("dataservers"),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_name,
        )

    @classmethod
    def from_dicts(
        cls,
        *,
        instrument_list: InstrumentsType | None = None,
        instruments: InstrumentsType | None = None,
        connections: ConnectionsType | None = None,
        dataservers: DataServersType | None = None,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ) -> DeviceSetup:
        """Construct the device setup from Python dicts, same structure as yaml

        Args:
            instrument_list (dict):
                List of instruments in the setup (deprecated; for
                backwards compatibility)
            instruments (dict):
                List of instruments in the setup
            connections (dict):
                Connections between devices
            server_host:
                Server host of the setup that should be created.
            server_port:
                Port of the server that should be created.
            setup_name:
                Name of the setup that should be created.
        """
        return _DeviceSetupGenerator.from_dicts(
            instrument_list=instrument_list,
            instruments=instruments,
            connections=connections,
            dataservers=dataservers,
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_name,
        )

    def _server_leader_instrument(self, server_uid: str) -> str | None:
        """Return a leader instrument for the given Dataserver UID."""
        for dev in self.instruments:
            if isinstance(dev, PQSC):
                if dev.server_uid == server_uid:
                    return dev.uid
