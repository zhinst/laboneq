// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod awg_delays;
mod device_traits;
mod generate_awg_events;
pub(crate) mod handle_feedback_registers;
pub mod ir;
pub mod node;
pub(crate) mod passes;
mod sample_waveforms;
mod sampled_event_handler;
mod settings;
pub mod signature;
pub mod tinysample;
mod utils;
pub(crate) mod virtual_signal;
pub use settings::CodeGeneratorSettings;
mod event_list;
mod generator;
pub mod result;
mod triggers;

pub use generator::generate_code;

// Re-export for easier access
pub use crate::sampled_event_handler::FeedbackRegister;
pub use crate::sampled_event_handler::FeedbackRegisterConfig;
pub use crate::sampled_event_handler::FeedbackRegisterLayout;
pub use crate::sampled_event_handler::SingleFeedbackRegisterLayoutItem;

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
