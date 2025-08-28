// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod awg_delays;
pub mod device_traits;
mod generate_awg_events;
pub mod ir;
pub mod node;
pub(crate) mod passes;
mod sample_waveforms;
mod settings;
pub mod signature;
pub mod tinysample;
pub(crate) mod utils;
pub(crate) mod virtual_signal;
pub use awg_delays::{AwgTiming, calculate_awg_delays};
pub use passes::fanout_awg::fanout_for_awg;
pub mod handle_feedback_registers;
pub use generate_awg_events::transform_ir_to_awg_events;
pub use passes::analyze_measurements;
pub use passes::{AwgCompilationInfo, analyze_awg_ir};
pub use settings::CodeGeneratorSettings;
pub type Samples = u64;

pub use sample_waveforms::{
    AwgWaveforms, SampledWaveform, WaveDeclaration, collect_and_finalize_waveforms,
    collect_integration_kernels,
};

pub mod waveform_sampler {
    pub use crate::sample_waveforms::{
        CompressedWaveformPart, IntegrationKernel, SampleWaveforms, SampledWaveformCollection,
        SampledWaveformSignature, WaveformSamplingCandidate,
    };
}

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
