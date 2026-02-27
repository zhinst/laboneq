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
pub mod utils;
pub(crate) mod virtual_signal;
use laboneq_error::LabOneQError;
pub use settings::CodeGeneratorSettings;
mod awg_processor;
mod context;
mod event_list;
mod generator;
mod integration_units;
mod ir_adapter;
pub mod result;
mod triggers;

pub use generator::generate_code;
// Public for Python layer, not intended for external use
pub use ir_adapter::{AwgInfo, CodegenIr, ir_to_codegen_ir};

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

pub type Result<T, E = LabOneQError> = std::result::Result<T, E>;
