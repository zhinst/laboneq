# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .. import Server


@classformatter
@dataclass(init=True, repr=True, order=True)
class DataServer(Server):
    """Class representing a LabOne Data Server.

    !!! version-changed "Changed in version 2.18"
        Removed `leader_uid` attribute.
    """

    #: Unique identifier.
    uid: str
    #: IP address or hostname of the data server.
    host: str
    #: Port number.
    port: str | int
    #: API level that is used to communicate with the data server.
    api_level: int
