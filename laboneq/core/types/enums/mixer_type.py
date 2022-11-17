# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum


class MixerType(enum.Enum):
    #: Mixer performs full complex modulation
    IQ = "IQ"

    #: Mixer only performs envelope modulation (UHFQA-style)
    UHFQA_ENVELOPE = "UHFQA_ENVELOPE"
