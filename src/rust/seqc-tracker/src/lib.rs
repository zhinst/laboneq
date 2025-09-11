// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod awg;
pub mod compressor;
mod output_mute;
pub mod prng_tracker;
pub mod seqc_generator;
pub mod seqc_statements;
pub mod seqc_tracker;
pub mod wave_index_tracker;

pub type FeedbackRegisterIndex = u32;
pub type Samples = i64;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl Error {
    pub fn new(msg: &str) -> Self {
        Error::Anyhow(anyhow::anyhow!(msg.to_string()))
    }

    pub fn with_error<E: Into<anyhow::Error>>(err: E) -> Self {
        Error::Anyhow(err.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
