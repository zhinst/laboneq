# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define
class Server:
    """Base class for servers.

    !!! version-changed "Changed in version 2.62.0"

        Changed the class to be slotted, which prevents the accidental creation of new
        attributes.
    """

    uid: str
