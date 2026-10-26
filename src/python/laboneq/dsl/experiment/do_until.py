# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import Any

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.experiment.section import Section
from laboneq.dsl.experiment.section_context import SectionContextManagerBase

if typing.TYPE_CHECKING:
    from laboneq.dsl.coprocessor.predicate import _IsLive, _Predicate


@classformatter
@attrs.define
class DoUntilSection(Section):
    """A do-until loop section.

    Attributes:
        condition: Predicate evaluated after each iteration; loop exits when
            it becomes True. Either a comparison predicate (value-based exit)
            or an `is_live(x)` arrival predicate (arrival-based exit).
        max_count: Worst-case iteration cap. Reaching it without the exit
            condition being satisfied is a fatal runtime error.
    """

    condition: Any | None = attrs.field(default=None)
    max_count: int = attrs.field(default=1)


class DoUntilSectionContextManager(SectionContextManagerBase):
    section_class = DoUntilSection

    def __init__(
        self,
        condition: Any,
        *,
        max_count: int,
        uid: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "condition": condition,
            "max_count": max_count,
        }
        if uid is not None:
            kwargs["uid"] = uid
        super().__init__(kwargs=kwargs)


def do_until(
    condition: _Predicate | _IsLive,
    *,
    max_count: int,
    uid: str | None = None,
) -> DoUntilSectionContextManager:
    """Real-time loop primitive.

    Use as `with do_until(is_live(var), max_count=50): ...` for
    arrival-based exit, or `with do_until(condition=var != 0, max_count=50):
    ...` for value-based exit.

    Args:
        condition: Predicate; loop exits when it evaluates to True.
            `is_live(x)` builds an arrival predicate that becomes True when
            an inbound update flips `x` from stale to live.
        max_count: Hard upper bound on iterations (required).
        uid: Optional explicit UID for the generated section.

    Returns:
        A context manager that auto-attaches the new `DoUntilSection` to
            the enclosing experiment or section on `__exit__`.
    """
    return DoUntilSectionContextManager(condition, max_count=max_count, uid=uid)
