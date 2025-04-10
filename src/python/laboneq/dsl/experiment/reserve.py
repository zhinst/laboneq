# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .operation import Operation


@classformatter
@attrs.define
class Reserve(Operation):
    """Operation to reserve a signal for the active section.
    Reserving an experiment signal in a section means that if there is no
    operation defined on that signal, it is not available for other sections
    as long as the active section is scoped.
    """

    #: Unique identifier of the signal that should be reserved.
    signal: str = attrs.field(default=None)
