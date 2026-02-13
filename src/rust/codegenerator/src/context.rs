// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::handle_feedback_registers::FeedbackConfig;
use crate::ir::SignalUid;
use crate::ir::experiment::AcquisitionType;
use crate::result::IntegrationUnitAllocation;
use crate::{CodeGeneratorSettings, FeedbackRegisterLayout};

pub(crate) struct CodeGenContext {
    // Configuration
    pub acquisition_type: AcquisitionType,

    // Resources
    pub feedback_config: FeedbackConfig,
    pub feedback_register_layout: FeedbackRegisterLayout,
    pub integration_unit_allocation: Vec<IntegrationUnitAllocation>,

    // Settings
    pub settings: CodeGeneratorSettings,
}

impl CodeGenContext {
    pub(crate) fn integration_units_for_signal(&self, signal_uid: SignalUid) -> Option<&Vec<u8>> {
        self.integration_unit_allocation
            .iter()
            .find(|alloc| alloc.signal == signal_uid)
            .map(|alloc| &alloc.channels)
    }
}
