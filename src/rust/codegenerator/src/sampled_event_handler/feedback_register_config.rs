// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

/// A struct representing the wiring of feedback registers through the PQSC/SHFQC.
///
/// # Attributes
///
/// * `local`: Whether the feedback is local (on the SHFQC) or not (i.e., via the PQSC)
/// * `register_index_select`: The index (aka 2 bit group) of the feedback register that is transmitted
///   by the PQSC to the generator. Is None for local feedback.
/// * `codeword_bitshift`: The shift into the codeword received by the generator from the
///   bus for decoding the measurement result.
/// * `codeword_bitmask`: The bitmask used for decoding the measurement result after applying the
///   shift.
/// * `command_table_offset`: Offset into the command table for mapping
///   the measurement result. Corresponds to the command table entry that is
///   played if the decoded value is zero.
#[derive(Default)]
pub struct FeedbackRegisterConfig {
    pub local: bool,

    // Receiver (SG instruments)
    pub register_index_select: Option<u8>,
    pub codeword_bitshift: Option<u8>,
    pub codeword_bitmask: Option<u16>,
    pub command_table_offset: Option<u32>,
}
