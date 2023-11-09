# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class FeedbackRegisterConfig:
    """A dataclass representing the wiring of feedback registers through the PQSC/SHFQC.

    Attributes:
        source_feedback_register:
            The register that the generator draws its feedback data from.
        register_index_select:
            What index (aka 2 bit group) of the feedback register the PQSC transmits to
            the generator. Is None for local feedback.
        codeword_bitshift:
            The shift into the codeword the generator receives from the
            bus for decoding the measurement result.
        codeword_bitmask:
            The bitmask used for decoding the measurement result after applying the
            shift.
        command_table_offset:
            Offset into the command table for mapping
            the measurement result. Corresponds to the command table entry that is
            played if the decoded value is zero.

        target_feedback_register:
            The register that the QA instrument sends its data to.
    """

    # Receiver (SG instruments)
    source_feedback_register: int | Literal["local"] | None = None
    register_index_select: int | None = None
    codeword_bitshift: int | None = None
    codeword_bitmask: int | None = None
    command_table_offset: int | None = None

    # transmitter (QA instruments)
    target_feedback_register: int | None = None
