# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Union

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if TYPE_CHECKING:
    from laboneq.data.setup_description import Setup


InstrumentsType = Dict[str, List[Dict[str, str]]]
ConnectionsType = Dict[str, List[Dict[str, Union[str, List[str]]]]]
DataServersType = Dict[str, Dict[str, Union[str, List[str]]]]


class DeviceSetupGenerator:
    @staticmethod
    def from_descriptor(
        yaml_text: str,
        server_host: str | None = None,
        server_port: int | str | None = None,
        setup_name: str | None = None,
    ) -> Setup:
        setup_desc = load(yaml_text, Loader=Loader)

        return DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_yaml(
        filepath,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ) -> Setup:
        with open(filepath) as fp:
            setup_desc = load(fp, Loader=Loader)

        return DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_dicts(
        instrument_list: InstrumentsType | None = None,
        instruments: InstrumentsType | None = None,
        connections: ConnectionsType | None = None,
        dataservers: DataServersType | None = None,
        server_host: str | None = None,
        server_port: int | str | None = None,
        setup_name: str | None = None,
    ) -> Setup:
        # Go though legacy DeviceSetup for legacy descriptor to avoid duplicate `DeviceSetupGenerator`
        from laboneq.dsl.device import DeviceSetup
        from laboneq.implementation.legacy_adapters.device_setup_converter import (
            convert_device_setup_to_setup,
        )

        legacy = DeviceSetup.from_dicts(
            instrument_list=instrument_list,
            instruments=instruments,
            connections=connections,
            dataservers=dataservers,
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_name,
        )
        return convert_device_setup_to_setup(legacy)
