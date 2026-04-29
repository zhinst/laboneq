# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MixerType(Enum):
    # Rust mirror: src/rust/codegenerator-py/src/common_types.rs MixerTypePy
    #: Mixer performs full complex modulation
    IQ = "IQ"
    #: Mixer only performs envelope modulation (UHFQA-style)
    UHFQA_ENVELOPE = "UHFQA_ENVELOPE"
