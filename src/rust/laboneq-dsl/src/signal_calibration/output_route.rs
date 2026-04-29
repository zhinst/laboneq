// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::ValueOrParameter;

/// Output routing for SHFSG outputs.
///
/// The routing must always be within the same device and can either be to another signal
/// defined in the experiment, or to an additional output of the same device that is not
/// defined in the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputRoute {
    /// Source channel on the source signal
    pub source_channel: String,
    pub amplitude_scaling: Option<ValueOrParameter<f64>>,
    pub phase_shift: Option<ValueOrParameter<f64>>,
}
