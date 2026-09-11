# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import ClassVar

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.device import Port

from .zi_standard_instrument import ZIStandardInstrument


@classformatter
@attrs.define
class ZQCS(ZIStandardInstrument):
    """Class representing a ZI ZQCS instrument."""

    backend: ClassVar[str] = "zqcs"

    #: Reached through the controller service at its own address.
    requires_data_server: ClassVar[bool] = False

    def calc_options(self):
        return {
            **super().calc_options(),
        }

    @property
    def ports(self) -> list[Port]:
        return [
            Port(
                uid=c.local_port,
                direction=c.direction,
                signal_type=c.signal_type,
            )
            for c in self.connections
        ]
