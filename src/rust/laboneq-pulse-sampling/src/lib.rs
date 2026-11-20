// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Evaluates pulse shapes into samples, independently of the backend.
//!
//! [`sample_pulses`] evaluates the envelope of each [`PulseRequest`], applies amplitude,
//! phase and software modulation, and adds each pulse sample directly into its
//! target buffer. Pulses not supported by this crate go to an
//! [`external::ExternalSampler`], such as a bridge to Python. [`builtin`] holds the
//! standard pulses evaluated by this crate as well as their pulse parameters.

pub mod builtin;
mod envelope;
pub mod external;
mod modulate;
mod pulse;
mod samples;

pub use envelope::{EnvelopeShape, FunctionalShape};
pub use pulse::{PulseRequest, sample_pulses};
pub use samples::{Samples, SamplesMut, SamplesRef};
