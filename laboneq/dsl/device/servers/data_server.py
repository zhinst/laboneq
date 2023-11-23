# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

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
    uid: str = field(default=None)
    #: API level that is used to communicate with the data server.
    api_level: int = field(default=None)
    #: IP address or hostname of the data server.
    host: str = field(default=None)
    #: Port number.
    port: Union[str, int] = field(default=None)
