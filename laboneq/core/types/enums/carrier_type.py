# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class CarrierType(Enum):
    IF = "INTERMEDIATE_FREQUENCY"
    RF = "RADIO_FREQUENCY"
