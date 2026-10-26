// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Cap'n Proto `CompiledExperiment` -> Python deserializer.

use capnp::any_pointer;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use laboneq_capnp::compiled_experiment::v1::{compiled_experiment_capnp, rt_loop_properties_capnp};
use laboneq_error::{WithContext, laboneq_error};

use crate::Result;
use crate::context::DeserializerContext;
use crate::context::{capnp_err, decode_numeric_array, text_to_string};
use crate::execution;
use crate::result_shape;

/// Deserializes a backend-specific `CompiledExperiment.artifacts` pointer into whatever
/// that backend's `ScheduledExperiment.artifacts` should hold; this crate has no opinion
/// on what that is.
pub type ArtifactsDeserializer = for<'py> fn(
    Python<'py>,
    any_pointer::Reader<'_>,
    &DeserializerContext,
) -> Result<Bound<'py, PyAny>>;

/// Deserializes `CompiledExperiment` Cap'n Proto bytes into a
/// `laboneq.data.scheduled_experiment.ScheduledExperiment`.
///
/// `artifacts_from_capnp` must belong to the backend that produced `bytes`; it fills in the
/// `ScheduledExperiment.artifacts` attribute with whatever that backend expects.
pub fn compiled_experiment_from_capnp<'py>(
    py: Python<'py>,
    bytes: &[u8],
    artifacts_from_capnp: ArtifactsDeserializer,
) -> Result<Bound<'py, PyAny>> {
    // Flat framing, matching the writer in `capnp_compiled_experiment`.
    let message = capnp::serialize::read_message_from_flat_slice(
        &mut &*bytes,
        laboneq_capnp::reader_options(),
    )
    .map_err(capnp_err)?;
    let root = message
        .get_root::<compiled_experiment_capnp::compiled_experiment::Reader<'_>>()
        .map_err(capnp_err)?;

    let metadata = root.get_metadata().map_err(capnp_err)?;
    validate_metadata(metadata)?;

    let (parameter_names, parameter_arrays) =
        decode_parameters(py, &root).with_context(|| "Reading `parameters`.")?;
    let handle_names = decode_handles(&root).with_context(|| "Reading `acquisitionHandles`.")?;
    let ctx = DeserializerContext {
        parameter_names,
        parameter_arrays,
        handle_names,
    };

    let instructions = root
        .get_execution()
        .map_err(capnp_err)?
        .get_instructions()
        .map_err(capnp_err)?;
    let execution_py = execution::build_execution(py, instructions, &ctx)
        .with_context(|| "Reading `execution`.")?;

    let rt_loop_properties_py =
        build_rt_loop_properties(py, root.get_real_time_properties().map_err(capnp_err)?)
            .with_context(|| "Reading `realTimeProperties`.")?;
    let result_shape_info_py = result_shape::build_result_shape_info(
        py,
        root.get_result_shapes().map_err(capnp_err)?,
        &ctx,
    )
    .with_context(|| "Reading `resultShapes`.")?;
    let versions_py = build_software_versions(py, metadata)?;
    let schedule_py = build_pulse_sheet(py, root.get_pulse_sheet().map_err(capnp_err)?)?;
    let artifacts_py = artifacts_from_capnp(py, root.get_artifacts(), &ctx)?;
    let (coprocessor_payloads_py, coprocessor_mappings_py) =
        build_coprocessors(py, &root).with_context(|| "Reading `coprocessors`.")?;
    let timing = root.get_timing().map_err(capnp_err)?;

    let scheduled_experiment_class = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "ScheduledExperiment"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "device_setup_fingerprint",
        text_to_string(root.get_hardware_fingerprint().map_err(capnp_err)?)?,
    )?;
    kwargs.set_item("artifacts", artifacts_py)?;
    kwargs.set_item("schedule", schedule_py)?;
    kwargs.set_item("execution", execution_py)?;
    kwargs.set_item("rt_loop_properties", rt_loop_properties_py)?;
    kwargs.set_item("result_shape_info", result_shape_info_py)?;
    kwargs.set_item("total_execution_time", timing.get_total_duration())?;
    kwargs.set_item("max_step_execution_time", timing.get_max_step_duration())?;
    kwargs.set_item("versions", versions_py)?;
    kwargs.set_item("coprocessor_payloads", coprocessor_payloads_py)?;
    kwargs.set_item("coprocessor_mappings", coprocessor_mappings_py)?;
    Ok(scheduled_experiment_class.call((), Some(&kwargs))?)
}

/// Reads just the producing backend's identifier out of a serialized message, without
/// decoding anything else.
pub fn peek_backend_name(bytes: &[u8]) -> Result<String> {
    let backend = laboneq_capnp::metadata::peek_backend_name(bytes)
        .map_err(|e| laboneq_error!("Could not read the producing backend name: {e}."))?;
    reject_empty_producer(&backend)?;
    Ok(backend)
}

/// Rejects metadata that names no producer.
fn reject_empty_producer(backend: &str) -> Result<()> {
    if backend.is_empty() {
        return Err(laboneq_error!(
            "CompiledExperiment metadata names no producing backend: `createdBy` is empty."
        ));
    }
    Ok(())
}

/// Checks the metadata this deserializer relies on, before decoding any of the message.
fn validate_metadata(metadata: compiled_experiment_capnp::metadata::Reader<'_>) -> Result<()> {
    let version = text_to_string(metadata.get_schema_version().map_err(capnp_err)?)?;
    if version != compiled_experiment_capnp::SCHEMA_VERSION {
        return Err(laboneq_error!(
            "CompiledExperiment schema version mismatch: message has '{version}', this \
             deserializer expects '{}'.",
            compiled_experiment_capnp::SCHEMA_VERSION
        ));
    }
    let created_by = text_to_string(metadata.get_created_by().map_err(capnp_err)?)?;
    reject_empty_producer(laboneq_capnp::metadata::backend_name(created_by))
}

fn decode_parameters(
    py: Python<'_>,
    root: &compiled_experiment_capnp::compiled_experiment::Reader<'_>,
) -> Result<(Vec<String>, Vec<Py<PyAny>>)> {
    let params = root.get_parameters().map_err(capnp_err)?;
    let mut names = Vec::with_capacity(params.len() as usize);
    let mut arrays = Vec::with_capacity(params.len() as usize);
    for p in params.iter() {
        names.push(text_to_string(p.get_uid().map_err(capnp_err)?)?.to_owned());
        arrays.push(decode_numeric_array(py, p.get_values().map_err(capnp_err)?)?.unbind());
    }
    Ok((names, arrays))
}

fn decode_handles(
    root: &compiled_experiment_capnp::compiled_experiment::Reader<'_>,
) -> Result<Vec<String>> {
    root.get_acquisition_handles()
        .map_err(capnp_err)?
        .iter()
        .map(|a| Ok(text_to_string(a.get_uid().map_err(capnp_err)?)?.to_owned()))
        .collect()
}

/// Splits `CompiledExperiment.coprocessors` into `ScheduledExperiment.coprocessor_payloads`
/// and `ScheduledExperiment.coprocessor_mapping`.
fn build_coprocessors<'py>(
    py: Python<'py>,
    root: &compiled_experiment_capnp::compiled_experiment::Reader<'_>,
) -> Result<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let payloads = PyDict::new(py);
    let mappings = PyDict::new(py);
    for binding in root.get_coprocessors().map_err(capnp_err)?.iter() {
        let label = text_to_string(binding.get_label().map_err(capnp_err)?)?;
        let inventory_key = text_to_string(binding.get_inventory_key().map_err(capnp_err)?)?;
        mappings.set_item(label, inventory_key)?;

        let payload = binding.get_payload().map_err(capnp_err)?;
        if !payload.is_empty() {
            // `&[u8]` would convert to `list[int]`; the payload must stay `bytes`.
            payloads.set_item(inventory_key, PyBytes::new(py, payload))?;
        }
    }
    Ok((payloads, mappings))
}

fn build_rt_loop_properties<'py>(
    py: Python<'py>,
    reader: rt_loop_properties_capnp::rt_loop_properties::Reader<'_>,
) -> Result<Bound<'py, PyAny>> {
    let acquisition_type_name = match reader.get_acquisition_type().map_err(capnp_err)? {
        rt_loop_properties_capnp::AcquisitionType::Integration => "INTEGRATION",
        rt_loop_properties_capnp::AcquisitionType::SpectroscopyIq => "SPECTROSCOPY_IQ",
        rt_loop_properties_capnp::AcquisitionType::SpectroscopyPsd => "SPECTROSCOPY_PSD",
        rt_loop_properties_capnp::AcquisitionType::Discrimination => "DISCRIMINATION",
        rt_loop_properties_capnp::AcquisitionType::Raw => "RAW",
    };
    let averaging_mode_name = match reader.get_averaging_mode().map_err(capnp_err)? {
        rt_loop_properties_capnp::AveragingMode::Sequential => "SEQUENTIAL",
        rt_loop_properties_capnp::AveragingMode::Cyclic => "CYCLIC",
        rt_loop_properties_capnp::AveragingMode::SingleShot => "SINGLE_SHOT",
    };

    use rt_loop_properties_capnp::rt_loop_properties::chunk_count::Which as ChunkWhich;
    let chunk_count: Option<u32> = match reader.get_chunk_count().which().map_err(capnp_err)? {
        ChunkWhich::None(()) => None,
        ChunkWhich::Value(v) => Some(v),
    };

    let acquisition_type = py
        .import(intern!(py, "laboneq.core.types.enums.acquisition_type"))?
        .getattr(intern!(py, "AcquisitionType"))?
        .getattr(acquisition_type_name)?;
    let averaging_mode = py
        .import(intern!(py, "laboneq.core.types.enums.averaging_mode"))?
        .getattr(intern!(py, "AveragingMode"))?
        .getattr(averaging_mode_name)?;

    let rt_loop_properties_class = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "RtLoopProperties"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("acquisition_type", acquisition_type)?;
    kwargs.set_item("averaging_mode", averaging_mode)?;
    kwargs.set_item("shots", reader.get_shots())?;
    kwargs.set_item("chunk_count", chunk_count)?;
    Ok(rt_loop_properties_class.call((), Some(&kwargs))?)
}

fn build_software_versions<'py>(
    py: Python<'py>,
    metadata: compiled_experiment_capnp::metadata::Reader<'_>,
) -> Result<Bound<'py, PyAny>> {
    // Only the version half of `createdBy` survives into `SoftwareVersions`.
    // A field without a separator yields "", matching `SoftwareVersions`' own default
    // for an unknown version. An entirely empty field cannot reach here:
    // `validate_metadata` rejects it.
    let created_by = text_to_string(metadata.get_created_by().map_err(capnp_err)?)?;
    let (_, version) = laboneq_capnp::metadata::split_created_by(created_by);

    let versions_class = py
        .import(intern!(py, "laboneq.data.scheduled_experiment"))?
        .getattr(intern!(py, "SoftwareVersions"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("laboneq", version)?;
    Ok(versions_class.call((), Some(&kwargs))?)
}

fn build_pulse_sheet(
    py: Python<'_>,
    reader: laboneq_capnp::compiled_experiment::v1::pulse_sheet_capnp::pulse_sheet::Reader<'_>,
) -> Result<Py<PyAny>> {
    let json_bytes = reader.get_json().map_err(capnp_err)?;
    if json_bytes.is_empty() {
        return Ok(py.None());
    }
    let value: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| laboneq_error!("Invalid pulseSheet JSON: {e}"))?;
    Ok(json_to_py(py, &value)?.unbind())
}

/// Helper function to convert serde_json::Value to Python objects.
///
/// Stays in `PyResult`: every failure here comes from Python object construction, and the
/// caller lifts it into a [`crate::Result`].
fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObject;
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyDict, PyList};
    use serde_json::Value;

    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(b) => {
            let py_bool = b
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(format!("Failed to convert bool: {}", e)))?;
            Ok(py_bool.to_owned().into_any())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let py_int = i
                    .into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(format!("Failed to convert i64: {}", e)))?;
                Ok(py_int.to_owned().into_any())
            } else if let Some(u) = n.as_u64() {
                let py_int = u
                    .into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(format!("Failed to convert u64: {}", e)))?;
                Ok(py_int.to_owned().into_any())
            } else if let Some(f) = n.as_f64() {
                let py_float = f
                    .into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(format!("Failed to convert f64: {}", e)))?;
                Ok(py_float.to_owned().into_any())
            } else {
                Err(PyValueError::new_err("Invalid JSON number"))
            }
        }
        Value::String(s) => {
            let py_str = s
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(format!("Failed to convert string: {}", e)))?;
            Ok(py_str.to_owned().into_any())
        }
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(json_to_py(py, item)?)?;
            }
            Ok(py_list.into_any())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                py_dict.set_item(key, json_to_py(py, val)?)?;
            }
            Ok(py_dict.into_any())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::peek_backend_name;
    use laboneq_capnp::compiled_experiment::v1::compiled_experiment_capnp::compiled_experiment;

    /// A message carrying nothing but `metadata.createdBy` -- the only field that `peek_backend_name` reads.
    fn message_created_by(created_by: &str) -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        message
            .init_root::<compiled_experiment::Builder<'_>>()
            .init_metadata()
            .set_created_by(created_by);
        capnp::serialize::write_message_to_words(&message)
    }

    #[test]
    fn reads_the_backend_name() {
        assert_eq!(
            peek_backend_name(&message_created_by("backend/26.10.0")).unwrap(),
            "backend"
        );
    }

    #[test]
    fn accepts_a_producer_that_omits_its_version() {
        assert_eq!(
            peek_backend_name(&message_created_by("backend")).unwrap(),
            "backend"
        );
    }

    #[test]
    fn rejects_metadata_that_names_no_producer() {
        // An unset Text field also reads as empty, so this covers a `createdBy` that was
        // never written.
        let err = peek_backend_name(&message_created_by(""))
            .unwrap_err()
            .to_string();
        assert!(err.contains("names no producing backend"), "{err}");
    }

    #[test]
    fn rejects_bytes_that_are_not_a_compiled_experiment() {
        let err = peek_backend_name(b"not-capnp-at-all")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Could not read the producing backend"),
            "{err}"
        );
    }
}
