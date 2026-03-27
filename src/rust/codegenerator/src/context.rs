// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::handle_feedback_registers::FeedbackConfig;
use crate::integration_units::IntegrationUnitAllocation;
use crate::ir::SignalUid;
use crate::ir::compilation_job::InitialSignalProperties;
use crate::ir::experiment::AcquisitionType;
use crate::{CodeGeneratorSettings, FeedbackRegisterLayout};

pub(crate) struct CodeGenContext {
    // Configuration
    pub acquisition_type: AcquisitionType,

    // Resources
    pub feedback_config: FeedbackConfig,
    pub feedback_register_layout: FeedbackRegisterLayout,
    pub integration_unit_allocation: Vec<IntegrationUnitAllocation>,

    // Initial signal properties (e.g. amplitude).
    // Settings that need to be applied before the execution starts.
    pub initial_signal_properties: Vec<InitialSignalProperties>,

    // Settings
    pub settings: CodeGeneratorSettings,
}

impl CodeGenContext {
    pub(crate) fn integration_units_for_signal(&self, signal_uid: SignalUid) -> Option<&Vec<u8>> {
        self.integration_unit_allocation
            .iter()
            .find(|alloc| alloc.signal == signal_uid)
            .map(|alloc| &alloc.units)
    }

    pub(crate) fn integration_unit_allocation_for_signal(
        &self,
        signal_uid: SignalUid,
    ) -> Option<&IntegrationUnitAllocation> {
        self.integration_unit_allocation
            .iter()
            .find(|alloc| alloc.signal == signal_uid)
    }

    pub(crate) fn signal_properties(
        &self,
        signal_uid: SignalUid,
    ) -> Option<&InitialSignalProperties> {
        self.initial_signal_properties
            .iter()
            .find(|prop| prop.uid == signal_uid)
    }
}
