# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs

from laboneq.data.setup_descriptions import SetupDescription

if TYPE_CHECKING:
    from pathlib import Path


@attrs.define
class ServerEntry:
    uid: str
    host: str | None = None
    port: int | None = None
    url: str | None = None


@attrs.define
class InstrumentEntry:
    uid: str
    device_type: str
    address: str
    options: list[str]
    server_uid: str | None = None
    interface: str | None = None
    reference_clock_source: str | None = None


@attrs.define
class ConnectionEntry:
    instrument_uid: str
    signal_name: str
    primary_port: str
    secondary_port: str | None = None
    signal_type: str | None = None


@attrs.define
class InstrumentTopology:
    servers: list[ServerEntry] = attrs.Factory(list)
    instruments: list[InstrumentEntry] = attrs.Factory(list)
    connections: list[ConnectionEntry] = attrs.Factory(list)
    setup_description: SetupDescription | None = None
    #: Base URL of the LabOne Q controller service that served this topology,
    #: e.g. ``https://lab.example.com/laboneq``. If a topology carries this URL,
    #: a `DeviceSetup` built from it dispatches experiments to that service instead
    #: of driving hardware directly (see `Session.connect`). Set by the service on
    #: the topology it serves; never part of a topology file.
    controller_service_url: str | None = None

    def __attrs_post_init__(self) -> None:
        if len(self.servers) > 1:
            missing = [i.uid for i in self.instruments if i.server_uid is None]
            if missing:
                raise ValueError(
                    f"server_uid is required on every instrument when multiple servers "
                    f"are present; missing on: {missing}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "servers": [attrs.asdict(s) for s in self.servers],
            "instruments": [attrs.asdict(i) for i in self.instruments],
            "connections": [attrs.asdict(c) for c in self.connections],
            "setup_description": (
                self.setup_description.serialize() if self.setup_description else None
            ),
            "controller_service_url": self.controller_service_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentTopology:
        servers = [ServerEntry(**s) for s in data.get("servers", [])]
        instruments = [InstrumentEntry(**i) for i in data.get("instruments", [])]
        connections = [ConnectionEntry(**c) for c in data.get("connections", [])]
        sd = data.get("setup_description")
        return cls(
            servers=servers,
            instruments=instruments,
            connections=connections,
            setup_description=SetupDescription.deserialize(sd) if sd else None,
            controller_service_url=data.get("controller_service_url"),
        )

    @classmethod
    def from_yaml(cls, yaml_text: str) -> InstrumentTopology:
        """Parse a YAML setup-file string into an InstrumentTopology.

        Uses the same YAML shape as ``laboneq-controller --instrumenttopology
        topology.yaml`` (see `laboneq.data.instrument_topology_yaml`), not the
        shape produced by `to_dict`/consumed by `from_dict`.
        """
        # Imported lazily to avoid a circular import: instrument_topology_yaml
        # imports the entry classes from this module.
        from laboneq.data.instrument_topology_yaml import (
            instrument_topology_from_yaml_text,
        )

        return instrument_topology_from_yaml_text(yaml_text)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> InstrumentTopology:
        """Parse a YAML setup file into an InstrumentTopology.

        See `from_yaml` for the expected YAML shape.
        """
        from laboneq.data.instrument_topology_yaml import load_instrument_topology

        return load_instrument_topology(path)
