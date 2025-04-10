# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define(slots=False)
class Server:
    uid: str
