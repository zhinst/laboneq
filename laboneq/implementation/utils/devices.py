# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from laboneq.data.setup_description import Port, PortType


def hdawg_ports() -> List[Port]:
    inputs = [Port(path="ZSYNCS/0", type=PortType.ZSYNC, channel=0)]

    outputs = [Port(path=f"SIGOUTS/{i}", type=PortType.RF, channel=i) for i in range(8)]
    outputs.append(Port(path="DIOS/0", type=PortType.DIO, channel=0))
    return inputs + outputs


def test_device_ports() -> List[Port]:
    return [Port(path=f"SIGOUTS/{0}", type=PortType.RF, channel=0)]


def pqsc_ports() -> List[Port]:
    return [Port(path=f"ZSYNCS/{i}", type=PortType.ZSYNC, channel=i) for i in range(18)]


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
