# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.data.setup_description import Setup


class SetupHelper:
    """
    Helper class for setup related tasks. Main purpose is to provide queries into the setup data structure.
    """

    @classmethod
    def get_instrument_of_logical_signal(cls, setup: Setup, logical_signal_path: str):
        grp, name = logical_signal_path.split("/")
        for i in setup.instruments:
            for c in i.connections:
                if grp == c.logical_signal.group and name == c.logical_signal.name:
                    return i
        raise Exception("No instrument found for logical signal " + logical_signal_path)

    @classmethod
    def get_flat_logcial_signals(cls, setup: Setup):
        logical_signals = []
        for logical_signal_group in setup.logical_signal_groups:
            logical_signals.extend(logical_signal_group.logical_signals)
        return logical_signals

    @classmethod
    def get_ports_of_logical_signal(cls, setup: Setup, logical_signal_path):
        instrument = cls.get_instrument_of_logical_signal(setup, logical_signal_path)
        ports = []
        grp, name = logical_signal_path.split("/")
        for c in instrument.connections:
            if c.logical_signal.name == name and grp == c.logical_signal.group:
                for ch_port in c.physical_channel.ports:
                    if ch_port not in ports:
                        ports.append(ch_port)
        return ports

    @classmethod
    def get_connections_of_logical_signal(cls, setup: Setup, logical_signal_path: str):
        instrument = cls.get_instrument_of_logical_signal(setup, logical_signal_path)
        grp, name = logical_signal_path.split("/")
        connections = []
        for c in instrument.connections:
            if c.logical_signal.name == name and grp == c.logical_signal.group:
                connections.append(c)
        return connections

    @classmethod
    def flat_logical_signals(cls, setup: Setup):
        logical_signals = []
        for logical_signal_group in setup.logical_signal_groups.values():
            logical_signals.extend(
                [
                    (logical_signal_group.uid, logical_signal)
                    for logical_signal in logical_signal_group.logical_signals.values()
                ]
            )
        return logical_signals
