// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod compressor;
mod output_mute;
pub mod prng_tracker;
pub mod seqc_generator;
pub mod seqc_statements;
pub mod seqc_tracker;
pub mod wave_index_tracker;

pub type Samples = u64;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
