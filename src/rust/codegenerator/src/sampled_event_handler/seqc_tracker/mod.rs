// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod awg;
pub(crate) mod compressor;
mod output_mute;
pub(crate) mod prng_tracker;
pub(crate) mod seqc_generator;
pub(crate) mod seqc_statements;
pub(crate) mod tracker;
pub(crate) mod wave_index_tracker;

pub(crate) type FeedbackRegisterIndex = u32;
