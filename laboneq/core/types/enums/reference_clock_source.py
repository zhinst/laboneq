# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


# TODO: Move to laboneq.data. Note that moving the type will cause issues when deserialising
#       objects that referred to the class in its old module. Moving the class is therefore
#       not as straight-forward as one might naively hope.
class ReferenceClockSource(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
