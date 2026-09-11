# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol of the near-time callback channel.

The controller and whatever serves its callbacks exchange these messages over
the callback WebSocket. Both directions are named here rather than spelled out
at each end, so that anything written against the channel has one place to read
and the wire strings have one place to change.

The channel carries its own version, separate from the service protocol version
in :mod:`laboneq.controller.constants` and from any LabOne Q release version.
A participant that the controller does not ship, and therefore cannot update,
is compatible by this version alone.
"""

from __future__ import annotations

from enum import Enum

CALLBACK_PROTOCOL_VERSION = "1.0"
"""Version of the message exchange defined in this module.

Bumped when the channel's messages change in a way a participant would notice.
Additions that an older participant can ignore do not need a bump; anything it
cannot ignore does.
"""


class MessageType(str, Enum):
    """Kind of a callback channel message, carried in its ``type`` field.

    The values are the wire format and are compared against messages decoded
    from JSON, so they must keep their exact spelling.
    """

    # Controller to callback provider.
    CALLBACK = "callback"
    DONE = "done"

    # Callback provider to controller.
    RESULT = "result"
    ERROR = "error"
