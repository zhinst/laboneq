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
is compatible by this version alone, admitted over the channel's own
handshake header rather than a LabOne Q version check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np

CALLBACK_PROTOCOL_VERSION = "1.0"
"""Version of the message exchange defined in this module.

Bumped when the channel's messages change in a way a participant would notice.
Additions that an older participant can ignore do not need a bump; anything it
cannot ignore does.
"""

CALLBACK_PROTOCOL_HEADER = "X-LabOneQ-Callback-Protocol-Version"
"""Handshake header carrying the participant's ``CALLBACK_PROTOCOL_VERSION``."""

SUPPORTED_CALLBACK_PROTOCOL_MAJORS = frozenset({1})
"""Major versions this controller admits; the minor part is informational."""

_SUPPORTED_MAJORS_TEXT = ", ".join(
    f"{major}.x" for major in sorted(SUPPORTED_CALLBACK_PROTOCOL_MAJORS)
)

# "major.minor", ASCII digits only.
_VERSION_RE = re.compile(r"(\d{1,4})\.\d{1,4}", re.ASCII)


def json_default(value: Any) -> Any:
    """Encoder hook for values that don't map to JSON's data model, like numpy scalars.

    Pass as ``json.dumps(..., default=json_default)``.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def json_failure(value: Any) -> str | None:
    """Check whether a value can be serialized to JSON.

    Returns:
        None if the value can be serialized to JSON; otherwise, a string explaining
        why *value* cannot be serialized to JSON."""
    try:
        json.dumps(value, default=json_default)
    except (TypeError, ValueError) as exc:
        if not isinstance(value, Mapping):
            return str(exc)
        # Re-encode entry by entry so the message names every offender, not
        # just the first one the encoder hit. If no single entry fails (e.g. a
        # non-string key somewhere), fall back to the encoder's own message.
        offenders = []
        for name, entry in value.items():
            try:
                json.dumps(entry, default=json_default)
            except (TypeError, ValueError) as inner:  # noqa: PERF203
                offenders.append(f"{name!r} ({inner})")
        return ", ".join(offenders) or str(exc)
    return None


class UnsupportedProtocolVersion(ValueError):
    """The participant's callback protocol version cannot be served."""


def check_protocol_version(value: str | None) -> None:
    """Admit a participant by its callback protocol version header.

    Additive changes bump the minor version and are admitted for any major the
    controller supports; incompatible changes bump the major.

    Raises:
        UnsupportedProtocolVersion: Header missing, unparseable or of an
            unsupported major. The message names the supported versions.
    """
    if value is None:
        raise UnsupportedProtocolVersion(
            f"Missing header {CALLBACK_PROTOCOL_HEADER}; supported: {_SUPPORTED_MAJORS_TEXT}"
        )
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None or int(match[1]) not in SUPPORTED_CALLBACK_PROTOCOL_MAJORS:
        # Clipped: the WS close reason is limited to about 120 characters.
        raise UnsupportedProtocolVersion(
            f"Unsupported callback protocol version {value[:40]!r}; supported: {_SUPPORTED_MAJORS_TEXT}"
        )


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
