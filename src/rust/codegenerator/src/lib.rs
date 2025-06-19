// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod device_traits;
mod generate_awg_events;
pub mod ir;
pub mod node;
pub(crate) mod passes;
mod sample_waveforms;
pub mod signature;
pub mod tinysample;
pub(crate) mod utils;
pub(crate) mod virtual_signal;

pub use generate_awg_events::transform_ir_to_awg_events;
pub use utils::string_sanitize;
pub type Samples = u64;

pub use sample_waveforms::{
    AwgWaveforms, SampledWaveform, WaveDeclaration, collect_and_finalize_waveforms,
};

pub mod waveform_sampler {
    pub use crate::sample_waveforms::{
        CompressedWaveformPart, SampleWaveforms, SampledWaveformCollection,
        SampledWaveformSignature, WaveformSamplingCandidate,
    };
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    #[error("External")]
    /// External error origination from outside of the library.
    /// This is used to wrap errors from e.g. when calling Python code.
    External(Box<dyn std::any::Any + Sync + Send>),
}

impl Error {
    pub fn new(msg: &str) -> Self {
        Error::Anyhow(anyhow::anyhow!(msg.to_string()))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
