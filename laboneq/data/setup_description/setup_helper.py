# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.data.setup_description import Setup

_logger = logging.getLogger(__name__)


class SetupHelper:
    """
    Helper class for setup related tasks. Main purpose is to provide queries into the setup data structure.
    """

    @classmethod
    def get_instrument_of_logical_signal(cls, setup: Setup, logical_signal_path: str):
        for i in setup.instruments:
            for c in i.connections:
                search_path = logical_signal_path
                # todo: this is a hack to make the path comparison work
                # to fix this, make sure paths are generated correctly
                if not search_path.startswith("/logical_signal_groups/"):
                    search_path = "/logical_signal_groups/" + search_path

                if c.logical_signal.path == search_path:
                    return i
        raise Exception("No instrument found for logical signal " + logical_signal_path)

    @classmethod
    def get_flat_logcial_signals(cls, setup: Setup):
        logical_signals = []
        for logical_signal_group in setup.logical_signal_groups:
            logical_signals.extend(logical_signal_group.logical_signals)
        return logical_signals

    @classmethod
    def get_connections_of_logical_signal(cls, setup: Setup, logical_signal_path: str):
        instrument = cls.get_instrument_of_logical_signal(setup, logical_signal_path)
        connections = []
        for c in instrument.connections:
            if c.logical_signal.path == "/logical_signal_groups/" + logical_signal_path:
                connections.append(c)
        return connections

    @classmethod
    def get_ports_of_logical_signal(cls, setup: Setup, logical_signal_path):
        instrument = cls.get_instrument_of_logical_signal(setup, logical_signal_path)
        ports = []
        physical_channels = []
        search_path = logical_signal_path
        if not search_path.startswith("/logical_signal_groups/"):
            search_path = "/logical_signal_groups/" + search_path

        for c in instrument.connections:
            if c.logical_signal.path == search_path:
                physical_channels.append(c.physical_channel)
        for p in instrument.ports:
            if p.physical_channel in physical_channels:
                ports.append(p)
        return ports

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
