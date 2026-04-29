# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class PhysicalChannelType(Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"
