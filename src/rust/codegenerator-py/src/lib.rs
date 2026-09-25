// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::CodeGeneratorSettings;
use codegenerator::generate_code;
use codegenerator::result::SeqCGenOutput;
use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_error::LabOneQError;
use laboneq_ir::ExperimentIr;
use laboneq_py_utils::py_object_store::PyObjectStore;
use pyo3::intern;
use pyo3::prelude::*;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod common_types;
mod result;

use codegenerator::ir_to_codegen_ir;

use crate::result::FeedbackRegisterConfigPy;
use crate::result::MeasurementPy;
use crate::result::ResultSourcePy;
use crate::waveform_sampler::WaveformSamplerPy;
use result::{AwgCodeGenerationResultPy, SeqCGenOutputPy};
mod utils;
mod waveform_sampler;

// Re-export types from codegenerator that are needed in the preprocessor and backend for the convenience.
pub use codegenerator::CODE_GENERATOR_SETTINGS;
pub use codegenerator::hardware_setup::{HardwareSetup, SetupDevice, SignalChannelProperties};
pub use codegenerator::ir::compilation_job::DeviceKind;

pub fn generate_code_py(
    py: Python<'_>,
    experiment: ExperimentIr,
    setup_description: &HardwareSetup,
    compiler_settings: &CompilerSettings,
    external_parameters: &PyObjectStore<ExternalParameterUid>,
) -> Result<SeqCGenOutput, LabOneQError> {
    let (codegen_ir, dedup) = ir_to_codegen_ir(&experiment, setup_description)?;
    let sampler = WaveformSamplerPy::new(
        py,
        &experiment.pulses,
        codegen_ir.acquisition_type.clone(),
        &dedup,
        external_parameters,
        experiment.id_store,
    );
    let settings = compiler_setting_to_codegenerator_settings(compiler_settings)?;
    let result = py.detach(|| generate_code(codegen_ir, &sampler, settings))?;
    Ok(result)
}

pub fn artifacts_to_py<'py>(
    py: Python<'py>,
    result: SeqCGenOutput,
    id_store: &NamedIdStore,
    external_parameters: &PyObjectStore<ExternalParameterUid>,
) -> Result<Bound<'py, PyAny>, LabOneQError> {
    let result = SeqCGenOutputPy::new(py, result, id_store, external_parameters)?;
    let parameter_py: Bound<'_, PyAny> = py
        .import(intern!(py, "laboneq.compiler.seqc.code_generator"))?
        .getattr(intern!(py, "generate_output"))?;
    let result_py = parameter_py.call1((result,))?;
    Ok(result_py)
}

fn compiler_setting_to_codegenerator_settings(
    compiler_settings: &CompilerSettings,
) -> Result<CodeGeneratorSettings, LabOneQError> {
    CodeGeneratorSettings::from_settings(
        &compiler_settings.backend,
        compiler_settings.ignore_resource_exhaustion,
    )
}

pub fn create_py_module<'a>(py: Python<'a>, name: &str) -> PyResult<Bound<'a, PyModule>> {
    use crate::result::ChannelPropertiesPy;

    let m = PyModule::new(py, name)?;
    // Common types
    // Move up the compiler stack as we need the common types
    m.add_class::<common_types::SignalTypePy>()?;
    m.add_class::<common_types::DeviceTypePy>()?;
    m.add_class::<common_types::PortModePy>()?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    m.add_class::<FeedbackRegisterConfigPy>()?;
    m.add_class::<MeasurementPy>()?;
    m.add_class::<ResultSourcePy>()?;
    m.add_class::<ChannelPropertiesPy>()?;
    Ok(m)
}
