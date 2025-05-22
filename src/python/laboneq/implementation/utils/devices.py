# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import json
from typing import List

from laboneq.data.execution_payload import TargetSetup
from laboneq.data.setup_description import Port, PortType, Setup
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)


def hdawg_ports() -> List[Port]:
    inputs = [Port(path="ZSYNCS/0", type=PortType.ZSYNC, channel=0)]

    outputs = [Port(path=f"SIGOUTS/{i}", type=PortType.RF, channel=i) for i in range(8)]
    outputs.append(Port(path="DIOS/0", type=PortType.DIO, channel=0))
    return inputs + outputs


def test_device_ports(ports: list[str]) -> List[Port]:
    return [
        Port(path=port, type=PortType.RF, channel=i) for i, port in enumerate(ports)
    ]


def pqsc_ports() -> List[Port]:
    return [Port(path=f"ZSYNCS/{i}", type=PortType.ZSYNC, channel=i) for i in range(18)]


def qhub_ports() -> List[Port]:
    return [Port(path=f"ZSYNCS/{i}", type=PortType.ZSYNC, channel=i) for i in range(56)]


def shfppc_ports() -> List[Port]:
    outputs = [
        Port(path=f"PPCHANNELS/{ch}", type=PortType.RF, channel=ch) for ch in range(4)
    ]
    return outputs


def shfqa_ports() -> List[Port]:
    inputs = [
        Port(path="DIOS/0", type=PortType.DIO, channel=0),
        Port(path="ZSYNCS/0", type=PortType.ZSYNC, channel=0),
    ]
    inputs.extend(
        Port(
            path=f"QACHANNELS/{ch}/INPUT",
            type=PortType.RF,
            channel=ch,
        )
        for ch in range(4)
    )

    outputs = [
        Port(path=f"QACHANNELS/{ch}/OUTPUT", type=PortType.RF, channel=ch)
        for ch in range(4)
    ]
    return inputs + outputs


def shfqc_ports() -> List[Port]:
    inputs = [
        Port(path="DIOS/0", type=PortType.DIO, channel=0),
        Port(path="ZSYNCS/0", type=PortType.ZSYNC, channel=0),
    ]
    inputs.extend(
        Port(
            path=f"QACHANNELS/{ch}/INPUT",
            type=PortType.RF,
            channel=ch,
        )
        for ch in range(4)
    )
    outputs = [
        Port(path=f"QACHANNELS/{ch}/OUTPUT", type=PortType.RF, channel=ch)
        for ch in range(4)
    ]
    outputs.extend(
        Port(path=f"SGCHANNELS/{ch}/OUTPUT", type=PortType.RF, channel=ch)
        for ch in range(8)
    )
    return inputs + outputs


def shfsg_ports() -> List[Port]:
    ports = [
        Port(path="DIOS/0", type=PortType.DIO, channel=0),
        Port(path="ZSYNCS/0", type=PortType.ZSYNC, channel=0),
    ]
    ports.extend(
        Port(path=f"SGCHANNELS/{ch}/OUTPUT", type=PortType.RF, channel=ch)
        for ch in range(8)
    )
    return ports


def uhfqa_ports() -> List[Port]:
    inputs = [Port(path="DIOS/0", type=PortType.DIO, channel=0)]
    outputs = [
        Port(path=f"SIGOUTS/{ch}", type=PortType.RF, channel=ch) for ch in range(2)
    ]
    inputs.extend(
        [Port(path=f"SIGINS/{ch}", type=PortType.RF, channel=ch) for ch in range(2)]
    )

    return inputs + outputs


def nonqc_ports() -> List[Port]:
    return []


def parse_device_options(device_options: str | None) -> tuple[str | None, list[str]]:
    if device_options is None:
        return None, []
    opts = device_options.upper().split("/")
    if len(opts) > 0 and opts[0] == "":
        opts.pop(0)
    dev_type: str | None = None
    if len(opts) > 0:
        dev_type = opts.pop(0)
    dev_opts = opts
    return dev_type, dev_opts


def target_setup_fingerprint(device_setup: TargetSetup) -> str:
    return json.dumps(
        sorted(
            [
                {
                    "uid": device.uid,
                    "type": device.device_type.name,
                    "options": device.device_options,
                }
                for device in device_setup.devices
            ],
            key=lambda x: x["uid"],
        )
    )


def device_setup_fingerprint(device_setup: Setup) -> str:
    return target_setup_fingerprint(TargetSetupGenerator.from_setup(device_setup))
