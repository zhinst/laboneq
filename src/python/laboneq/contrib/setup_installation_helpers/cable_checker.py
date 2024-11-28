# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.experiment import Experiment


class SignalType(Enum):
    """:meta private:"""

    RF = "rf_signal"
    IQ = "iq_signal"


@dataclass
class DeviceProperties:
    """:meta private:"""

    number_of_sg_ports: int | list[int]
    number_of_channels_per_sg_port: int
    number_of_qa_ports: int
    number_of_channels_per_qa_port: int
    sg_signal_type: SignalType
    channel_per_pulse: bool | list[bool]
    sg_prefix: str | list[str]
    sg_suffix: str
    qa_prefix: str
    qa_suffix: str


@dataclass
class Signal:
    """:meta private:"""

    base_signal_id: str
    signal_ids: dict[str, str]
    device_id: str
    signal_number: int
    port: str
    type: SignalType

    def get_for(self, pulse):
        return self.signal_ids.get(pulse, self.signal_ids.get("", None))


@dataclass
class Device:
    """Device to consider for cable checking."""

    type: str
    zsync_port: int | None = None


DEVICE_PROPERTIES = {
    "PQSC": DeviceProperties(
        number_of_sg_ports=0,
        number_of_channels_per_sg_port=0,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=False,
        sg_prefix="",
        sg_suffix="",
        qa_prefix="",
        qa_suffix="",
    ),
    "HDAWG": DeviceProperties(
        number_of_sg_ports=8,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.RF,
        channel_per_pulse=False,
        sg_prefix="SIGOUTS/",
        sg_suffix="",
        qa_prefix="",
        qa_suffix="",
    ),
    "HDAWG4": DeviceProperties(
        number_of_sg_ports=4,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.RF,
        channel_per_pulse=False,
        sg_prefix="SIGOUTS/",
        sg_suffix="",
        qa_prefix="",
        qa_suffix="",
    ),
    "HDAWG8": DeviceProperties(
        number_of_sg_ports=8,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.RF,
        channel_per_pulse=False,
        sg_prefix="SIGOUTS/",
        sg_suffix="",
        qa_prefix="",
        qa_suffix="",
    ),
    "SHFQA": DeviceProperties(
        number_of_sg_ports=4,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=4,
        number_of_channels_per_qa_port=1,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=True,
        sg_prefix="QACHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="QACHANNELS/",
        qa_suffix="/INPUT",
    ),
    "SHFQA2": DeviceProperties(
        number_of_sg_ports=2,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=2,
        number_of_channels_per_qa_port=1,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=True,
        sg_prefix="QACHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="QACHANNELS/",
        qa_suffix="/INPUT",
    ),
    "SHFQA4": DeviceProperties(
        number_of_sg_ports=4,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=4,
        number_of_channels_per_qa_port=1,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=True,
        sg_prefix="QACHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="QACHANNELS/",
        qa_suffix="/INPUT",
    ),
    "SHFSG": DeviceProperties(
        number_of_sg_ports=8,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=False,
        sg_prefix="SGCHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="",
        qa_suffix="",
    ),
    "SHFSG4": DeviceProperties(
        number_of_sg_ports=4,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=False,
        sg_prefix="SGCHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="",
        qa_suffix="",
    ),
    "SHFSG8": DeviceProperties(
        number_of_sg_ports=8,
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=0,
        number_of_channels_per_qa_port=0,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=False,
        sg_prefix="SGCHANNELS/",
        sg_suffix="/OUTPUT",
        qa_prefix="",
        qa_suffix="",
    ),
    # "UHFQA": DeviceProperties(2, 2, 2, 2, SignalType.IQ, "SIGOUTS/", "", "", ""),
    # QA ports come first!
    "SHFQC": DeviceProperties(
        number_of_sg_ports=[1, 6],
        number_of_channels_per_sg_port=1,
        number_of_qa_ports=1,
        number_of_channels_per_qa_port=1,
        sg_signal_type=SignalType.IQ,
        channel_per_pulse=[True, False],
        sg_prefix=["QACHANNELS/", "SGCHANNELS/"],
        sg_suffix="/OUTPUT",
        qa_prefix="QACHANNELS/",
        qa_suffix="/INPUT",
    ),
}


def create_device_setup(
    devices: dict[str, Device], server_host: str, server_port: int | str
) -> tuple[DeviceSetup, list[Signal], list[Signal]]:
    """:meta private:"""
    dataservers: dict[str, dict[str, str | list[str]]] = {
        "zi_server": {"host": server_host, "port": int(server_port), "instruments": []}
    }
    instruments: dict[str, list[dict[str, str]]] = {}
    connections: dict[str, list[dict[str, str | list[str]]]] = {}
    sg_signals: list[Signal] = []
    qa_signals: list[Signal] = []

    for uid, device in devices.items():
        device_type = device.type.upper()
        name = f"device_{device_type}_{uid}"
        base_type = device_type[:5].upper()
        instruments.setdefault(base_type, []).append({"address": uid, "uid": name})
        dataservers["zi_server"]["instruments"].append(name)  # type: ignore
        conn = []
        devprops = DEVICE_PROPERTIES[device_type]

        n_sg_ports = devprops.number_of_sg_ports
        if not isinstance(n_sg_ports, list):
            n_sg_ports = [n_sg_ports]
            sg_prefix = [devprops.sg_prefix]
            channel_per_pulse = [devprops.channel_per_pulse]
        else:
            sg_prefix = devprops.sg_prefix  # type: ignore
            channel_per_pulse = devprops.channel_per_pulse  # type: ignore

        for k, (nprt, sgprf, cps) in enumerate(
            zip(n_sg_ports, sg_prefix, channel_per_pulse)
        ):
            j = 1
            for i in range(nprt):
                assert (
                    devprops.number_of_channels_per_sg_port == 1
                ), "So far, only one channel per port is supported"
                ls_name = f"{name}_{k}_{j}"
                port = f"{sgprf}{i}{devprops.sg_suffix}"
                pulse_list = ["init", "start", "end", "1", "."] if cps else [""]
                conn += [
                    {
                        devprops.sg_signal_type.value: f"q/{ls_name}{pulse}",
                        "port": port,
                    }
                    for pulse in pulse_list
                ]
                sg_signals.append(
                    Signal(
                        base_signal_id=ls_name,
                        signal_ids={pulse: f"{ls_name}{pulse}" for pulse in pulse_list},
                        device_id=uid,
                        signal_number=j,
                        type=devprops.sg_signal_type,
                        port=port,
                    )
                )
                j += devprops.number_of_channels_per_sg_port
        for i in range(devprops.number_of_qa_ports):
            j = i + 1
            ls_name = f"{name}_0_{j}_acq"
            port = f"{devprops.qa_prefix}{i}{devprops.qa_suffix}"
            conn.append(
                {
                    "acquire_signal": f"q/{ls_name}",
                    "port": port,
                }
            )
            qa_signals.append(
                Signal(
                    base_signal_id=ls_name,
                    signal_ids={"": ls_name},
                    device_id=uid,
                    signal_number=j,
                    type=SignalType.IQ,
                    port=port,
                )
            )
        connections[name] = conn
    if "PQSC" in instruments:
        pqsc_name = instruments["PQSC"][0]["uid"]
        for uid, device in devices.items():
            if device.zsync_port is not None:
                device_type = device.type.upper()
                name = f"device_{device_type}_{uid}"
                connections[pqsc_name].append(
                    {"to": name, "port": f"ZSYNCS/{device.zsync_port}"}
                )

    device_setup = DeviceSetup.from_dicts(
        instruments=instruments, connections=connections, dataservers=dataservers
    )
    return device_setup, sg_signals, qa_signals


def get_matching_acquire(signal_id, qa_signals):
    """:meta private:"""
    return next(
        (qas for qas in qa_signals if qas.base_signal_id == signal_id + "_acq"), None
    )


def check_cable_experiment(
    devices,
    server_host: str,
    server_port: str | int,
    play_parallel: bool = True,
    play_initial_trigger: bool = False,
    bit_pulse_length: float = 256e-9,
    bit_gap: float = 128e-9,
) -> tuple[Experiment, DeviceSetup]:
    """
    Create an experiment to check the cables of the devices in the setup.

    Args:
        devices (dict[str, Device]): A dictionary of devices to check.
        server_host (str): The server host to connect to.
        server_port (int): The server port to connect to.
        play_parallel (bool, optional): Whether to play the pulses in parallel.
        play_initial_trigger (bool, optional): Whether to play a pulse on each
                                               output as initial trigger.

    Returns:
        Tuple[Experiment, DeviceSetup]: The experiment and the device setup to
        be used with a Session object.

    Usage:

    .. code-block :: python

        experiment, device_setup = check_cables(
            devices=devices,
            server_host="11.22.33.44",
            server_port=8004,
            play_parallel=False,
            play_initial_trigger=False,
        )
        session=Session(device_setup)
        session.connect()
        session.run(experiment)
    """
    device_setup, sg_signals, qa_signals = create_device_setup(
        devices, server_host=server_host, server_port=server_port
    )
    init_pulse = pulse_library.const(length=bit_pulse_length)
    start_pulse = pulse_library.const(length=bit_pulse_length, amplitude=0.8)
    end_pulse = pulse_library.const(length=bit_pulse_length, amplitude=0.9)
    on_pulse = pulse_library.const(length=bit_pulse_length, amplitude=0.5)
    off_pulse = pulse_library.const(length=bit_pulse_length, amplitude=0.2)
    acq_pulse = pulse_library.const(length=bit_pulse_length)
    lsg = device_setup.logical_signal_groups["q"].logical_signals
    sg_all_signals = {s: lsg[s] for sig in sg_signals for s in sig.signal_ids.values()}
    qa_all_signals = {s: lsg[s] for sig in qa_signals for s in sig.signal_ids.values()}
    signal_map = {**sg_all_signals, **qa_all_signals}
    exp = Experiment(signals=list(signal_map.keys()))
    exp.set_signal_map(signal_map)
    block_signal = sg_signals[0].get_for("start")
    with exp.acquire_loop_rt(1):
        if play_initial_trigger:
            with exp.section(uid="initial_trigger"):
                for s in sg_signals:
                    exp.play(
                        signal=s.get_for("init"), phase=np.pi / 4, pulse=init_pulse
                    )
                    if qa_sig := get_matching_acquire(s.base_signal_id, qa_signals):
                        exp.acquire(
                            signal=qa_sig.signal_ids[""],
                            kernel=acq_pulse,
                            handle=qa_sig.signal_ids[""],
                        )
                    exp.delay(signal=s.get_for("init"), time=2 * bit_gap)

        for dev_nr, dev in enumerate(devices.keys()):
            print(dev)
            bin_rep_dev = np.binary_repr(dev_nr, width=4).replace("0", ".")
            my_signals = sorted(
                [s for s in sg_signals if s.device_id == dev],
                key=lambda s: s.signal_number,
            )
            for s in my_signals:
                bin_rep_sig = np.binary_repr(s.signal_number, width=4).replace("0", ".")
                pattern = bin_rep_dev + bin_rep_sig
                print(f" - Port: {s.port} ({pattern}) {s.base_signal_id}")
                with exp.section(uid=s.base_signal_id):
                    if qa_sig := get_matching_acquire(s.base_signal_id, qa_signals):
                        # The SHFQA is much more restricted in the way measurement pulses can be played.
                        # We also need at least one acquire to set the acquisition mode to readout.
                        # So let's just measure everything, which can be used later to check the arriving
                        # signal pattern.
                        qa_sig_name = qa_sig.signal_ids[""]
                        with exp.section():
                            with exp.section():
                                if not play_parallel:
                                    exp.reserve(block_signal)
                                exp.measure(
                                    measure_signal=s.get_for("start"),
                                    measure_pulse=start_pulse,
                                    acquire_signal=qa_sig_name,
                                    integration_kernel=acq_pulse,
                                    handle=qa_sig_name,
                                )
                                exp.delay(signal=s.get_for("start"), time=bit_gap)
                            for p in pattern:
                                with exp.section():
                                    if not play_parallel:
                                        exp.reserve(block_signal)
                                    exp.measure(
                                        measure_signal=s.get_for(p),
                                        measure_pulse=on_pulse
                                        if p == "1"
                                        else off_pulse,
                                        acquire_signal=qa_sig_name,
                                        integration_kernel=acq_pulse,
                                        handle=qa_sig_name + "_pattern",
                                    )
                                    exp.delay(signal=s.get_for(p), time=bit_gap)
                            with exp.section():
                                if not play_parallel:
                                    exp.reserve(block_signal)
                                exp.measure(
                                    measure_signal=s.get_for("end"),
                                    measure_pulse=end_pulse,
                                    acquire_signal=qa_sig_name,
                                    integration_kernel=acq_pulse,
                                    handle=qa_sig_name,
                                )
                                exp.delay(signal=s.get_for("end"), time=128e-9)
                    else:
                        signal_name = s.get_for("")
                        play_args = {
                            "signal": signal_name,
                            "phase": np.pi / 4,
                        }
                        if not play_parallel:
                            exp.reserve(block_signal)
                        exp.play(**play_args, pulse=start_pulse)
                        exp.delay(signal=signal_name, time=bit_gap)
                        for p in pattern:
                            exp.play(
                                **play_args, pulse=on_pulse if p == "1" else off_pulse
                            )
                            exp.delay(signal=signal_name, time=bit_gap)
                        exp.play(**play_args, pulse=end_pulse)
                        exp.delay(signal=signal_name, time=128e-9)
    return exp, device_setup
