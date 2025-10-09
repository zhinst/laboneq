// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod awg;
pub mod compressor;
mod output_mute;
pub mod prng_tracker;
pub mod seqc_generator;
pub mod seqc_statements;
pub mod tracker;
pub mod wave_index_tracker;

pub type FeedbackRegisterIndex = u32;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}
