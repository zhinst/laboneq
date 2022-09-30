# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from .operation import Operation

from dataclasses import dataclass, field

from typing import Any


@dataclass(init=True, repr=True, order=True)
class Reserve(Operation):
    """Operation to reserve a signal for the active section.
    Reserving an experiment signal in a section means that if there is no
    operation defined on that signal, it is not available for other sections
    as long as the active section is scoped.
    """

    #: Unique identifier of the signal that should be reserved.
    signal: str = field(default=None)

    def get_all_signals(self) -> set:
        """Retrieves a set with all the signals used by this operation.

        Returns:
            Set with the signal of this operation.
        """
        if self.signal is None:
            return set()
        else:
            return {self.signal}
