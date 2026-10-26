# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""HQCS section-level operation IR nodes: _Send, _MarkStale, _Ensure."""

from __future__ import annotations

import typing
from typing import Any

import attrs

if typing.TYPE_CHECKING:
    from laboneq.dsl.coprocessor.predicate import _IsLive
    from laboneq.dsl.coprocessor.stream import _Stream
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.variable import Variable


@attrs.define
class _Send:
    """A `send(stream, **kwargs)` operation attached to a section."""

    stream: _Stream
    literal_kwargs: dict[str, Any] = attrs.field(factory=dict)


@attrs.define
class _MarkStale:
    """An `mark_stale(target)` operation attached to a section.

    Marks the target Variable or Pulse stale.
    """

    target: Variable | Pulse


@attrs.define
class _Ensure:
    """An `ensure(condition)` operation attached to a section.

    Asserts that `condition` holds at this point.
    """

    condition: _IsLive
