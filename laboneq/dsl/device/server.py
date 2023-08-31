# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@dataclass(init=True, repr=True, order=True)
class Server:
    uid: str = field(default=None)
