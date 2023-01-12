# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Union

from .. import Server


@dataclass(init=True, repr=True, order=True)
class DataServer(Server):
    """Class representing a LabOne Data Server."""

    #: Unique identifier.
    uid: str = field(default=None)
    #: API level that is used to communicate with the data server.
    api_level: int = field(default=None)
    #: IP address or hostname of the data server.
    host: str = field(default=None)
    #: Unique identifier of the leader device.
    leader_uid: str = field(default=None)
    #: Port number.
    port: Union[str, int] = field(default=None)
