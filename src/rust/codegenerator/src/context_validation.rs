// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_error::bail;

use crate::CodegenIr;
use crate::Result;
use crate::ir::compilation_job::DeviceKind;
use crate::ir::experiment::AcquisitionType;

pub(crate) fn validate_codegen_ir(codegen_ir: &CodegenIr) -> Result<()> {
    validate_shfqa_configuration(codegen_ir)?;
    Ok(())
}

/// Validate SHFQA configuration.
///
/// In RAW mode, all SHFQA devices must have the same `port_delay` for their acquisition signals.
/// TODO: Could it be per device instead of globally?
fn validate_shfqa_configuration(codegen_ir: &CodegenIr) -> Result<()> {
    if !matches!(codegen_ir.acquisition_type, AcquisitionType::RAW) {
        return Ok(());
    }

    let initial_signal_properties = codegen_ir
        .initial_signal_properties
        .iter()
        .map(|s| (s.uid, s))
        .collect::<HashMap<_, _>>();

    let mut all_shfqa_port_delays = codegen_ir
        .awg_cores
        .iter()
        // Collect SHFQA input signals
        .filter(|awg| matches!(awg.device_kind(), DeviceKind::SHFQA))
        .flat_map(|awg| awg.signals.iter())
        .filter(|s| !s.is_output())
        // Get port delay for each signal, if defined
        .filter_map(|signal| {
            initial_signal_properties
                .get(&signal.uid)
                .and_then(|prop| prop.port_delay.as_ref())
        });

    if let Some(first_delay) = all_shfqa_port_delays.next()
        && !all_shfqa_port_delays.all(|d| d == first_delay)
    {
        bail!(
            "Multiple different port delays defined for SHFQA acquisition signals in 'RAW' acquisition mode. Only 1 supported."
        );
    }

    Ok(())
}
