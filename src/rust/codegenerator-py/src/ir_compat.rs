// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use codegenerator::ir::IrNode;
use codegenerator::ir::compilation_job::{
    PulseDef as CodePulseDef, PulseDefKind, PulseType, Signal, SweepParameter as CodeSweepParameter,
};
use codegenerator::ir::experiment::AcquisitionType;
use codegenerator::ir_to_codegen_ir;
use codegenerator_utils::pulse_parameters::PulseParameterDeduplicator;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{AcquisitionType as AcquisitionTypeCommon, SweepParameter};
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use laboneq_py_utils::pulse::{PulseDef, PulseKind};

use crate::Result;

pub(crate) struct IrTransformResult {
    pub root: IrNode,
    pub pulse_parameters: PulseParameterDeduplicator,
    pub acquisition_type: AcquisitionType,
}

pub(crate) fn ir_to_code_compat(
    ir_experiment: &ExperimentIrPy,
    signals: &[&Arc<Signal>],
) -> Result<IrTransformResult> {
    let pulse_defs: Vec<Arc<CodePulseDef>> = ir_experiment
        .pulses
        .iter()
        .map(|pulse| Arc::new(transform_pulse_def(pulse, &ir_experiment.inner.id_store)))
        .collect();
    let sweep_parameters = ir_experiment
        .inner
        .parameters
        .iter()
        .map(|param| {
            Arc::new(transform_parameter_to_code_parameter(
                param,
                &ir_experiment.inner.id_store,
            ))
        })
        .collect::<Vec<Arc<CodeSweepParameter>>>();
    let codegen_ir = ir_to_codegen_ir(
        &ir_experiment.inner.root,
        &ir_experiment.inner.id_store,
        signals.to_vec(),
        pulse_defs.iter().collect(),
        sweep_parameters.iter().collect(),
    )?;
    let result = IrTransformResult {
        root: codegen_ir.root,
        pulse_parameters: codegen_ir.pulse_parameters,
        acquisition_type: convert_acquisition_type(&ir_experiment.inner.acquisition_type),
    };
    Ok(result)
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

fn transform_parameter_to_code_parameter(
    parameter: &SweepParameter,
    id_store: &NamedIdStore,
) -> CodeSweepParameter {
    let uid = id_store.resolve_unchecked(parameter.uid);
    CodeSweepParameter {
        uid: uid.to_string(),
        values: Arc::clone(&parameter.values),
    }
}

pub(crate) fn convert_acquisition_type(
    acquisition_type: &AcquisitionTypeCommon,
) -> AcquisitionType {
    match acquisition_type {
        AcquisitionTypeCommon::Raw => AcquisitionType::RAW,
        AcquisitionTypeCommon::Integration => AcquisitionType::INTEGRATION,
        AcquisitionTypeCommon::SpectroscopyIq => AcquisitionType::SPECTROSCOPY_IQ,
        AcquisitionTypeCommon::SpectroscopyPsd => AcquisitionType::SPECTROSCOPY_PSD,
        AcquisitionTypeCommon::Spectroscopy => AcquisitionType::SPECTROSCOPY_IQ,
        AcquisitionTypeCommon::Discrimination => AcquisitionType::DISCRIMINATION,
    }
}
