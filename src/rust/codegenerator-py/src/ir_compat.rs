// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_common::named_id::NamedIdStore;
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use laboneq_py_utils::pulse::{PulseDef, PulseKind};

use codegenerator::ir::compilation_job::{PulseDef as CodePulseDef, PulseDefKind, PulseType};
use codegenerator::{AwgInfo, CodegenIr, Result, ir_to_codegen_ir};

pub(crate) fn ir_to_code_compat(
    ir_experiment: &ExperimentIrPy,
    awg_cores: Vec<AwgInfo>,
) -> Result<CodegenIr> {
    let pulse_defs: Vec<Arc<CodePulseDef>> = ir_experiment
        .pulses
        .iter()
        .map(|pulse| Arc::new(transform_pulse_def(pulse, &ir_experiment.inner.id_store)))
        .collect();
    let codegen_ir =
        ir_to_codegen_ir(&ir_experiment.inner, awg_cores, pulse_defs.iter().collect())?;
    Ok(codegen_ir)
}

fn transform_pulse_def(pulse: &PulseDef, id_store: &NamedIdStore) -> CodePulseDef {
    let uid = id_store.resolve_unchecked(pulse.uid);
    let (kind, pulse_type) = match &pulse.kind {
        PulseKind::MarkerPulse { .. } => (PulseDefKind::Marker, Some(PulseType::Function)),
        PulseKind::Sampled { .. } => (PulseDefKind::Pulse, Some(PulseType::Samples)),
        PulseKind::LengthOnly { .. } => (PulseDefKind::Pulse, None),
        _ => (PulseDefKind::Pulse, Some(PulseType::Function)),
    };

    CodePulseDef {
        uid: uid.to_string(),
        kind,
        pulse_type,
    }
}
