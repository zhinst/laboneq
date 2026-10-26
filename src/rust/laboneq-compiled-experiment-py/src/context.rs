// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Shared decoding logic for compiler output and backend-specific artifacts.

use num_complex::Complex64;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use laboneq_capnp::compiled_experiment::v1::{common_capnp, values_capnp};
use laboneq_error::{LabOneQError, laboneq_error};

use crate::Result;

/// Converts a capnp error (`capnp::Error` from field accessors, or `capnp::NotInSchema` from
/// `.which()` on a union) into the "compiled experiment bytes are malformed" failure mode.
///
/// Callers add context describing *what* they were reading; this only says why it failed.
pub(crate) fn capnp_err(err: impl std::fmt::Display) -> LabOneQError {
    laboneq_error!("Malformed CompiledExperiment bytes: {err}")
}

pub(crate) fn text_to_string(reader: capnp::text::Reader<'_>) -> Result<&str> {
    reader
        .to_str()
        .map_err(|e| laboneq_error!("Invalid UTF-8: {e}"))
}

/// The `CompiledExperiment.parameters`/`acquisitionHandles` tables, decoded once up front:
/// each entry's name plus (for parameters) its full values array, indexed by capnp position
/// -- the same indices that `execution` and result shapes reference.
///
/// Passed to the backend's [`crate::ArtifactsDeserializer`] so backend-specific artifacts
/// can resolve the same parameter/handle references as `execution` and result shapes do.
pub struct DeserializerContext {
    /// The name of each parameter, indexed by capnp position.
    pub parameter_names: Vec<String>,
    /// Pre-built NumPy array for each parameter, shared between `execution`'s
    /// `SetParameter` and result-shape `parameterRef` axis values (mirrors the wire
    /// format, which serializes each parameter's values exactly once for both uses).
    pub parameter_arrays: Vec<Py<PyAny>>,
    /// The name of each acquisition handle, indexed by capnp position.
    pub handle_names: Vec<String>,
}

impl DeserializerContext {
    /// Returns the name of the parameter at the given index, or fails if the index is out
    /// of bounds.
    pub fn parameter_name(&self, idx: u32) -> Result<&str> {
        self.parameter_names
            .get(idx as usize)
            .map(String::as_str)
            .ok_or_else(|| {
                laboneq_error!("CompiledExperiment.parameters has no entry at index {idx}")
            })
    }

    /// Returns the NumPy array of values for the parameter at the given index, or fails if
    /// the index is out of bounds.
    pub fn parameter_array(&self, py: Python<'_>, idx: u32) -> Result<Py<PyAny>> {
        self.parameter_arrays
            .get(idx as usize)
            .map(|arr| arr.clone_ref(py))
            .ok_or_else(|| {
                laboneq_error!("CompiledExperiment.parameters has no entry at index {idx}")
            })
    }

    /// Returns the name of the acquisition handle at the given index, or fails if the index
    /// is out of bounds.
    pub fn handle_name(&self, idx: u32) -> Result<&str> {
        self.handle_names
            .get(idx as usize)
            .map(String::as_str)
            .ok_or_else(|| {
                laboneq_error!("CompiledExperiment.acquisitionHandles has no entry at index {idx}")
            })
    }
}

/// The decoded value of a `Common.Value` reader.
pub enum DecodedValue {
    None,
    Constant(Py<PyAny>),
    ParameterRef(u32),
}

/// Decodes a `Common.Value` reader. The generated types are the same
/// across every schema that imports `compiled_experiment/v1/common.capnp`,
/// so a backend's own artifacts deserializer can reuse this directly rather than
/// re-implementing the same match.
pub fn decode_value(
    py: Python<'_>,
    reader: common_capnp::value::Reader<'_>,
) -> Result<DecodedValue> {
    use common_capnp::value::Which;
    match reader.which().map_err(capnp_err)? {
        Which::None(()) => Ok(DecodedValue::None),
        Which::Constant(c) => Ok(DecodedValue::Constant(decode_constant(
            py,
            c.map_err(capnp_err)?,
        )?)),
        Which::ParameterRef(idx) => Ok(DecodedValue::ParameterRef(idx)),
    }
}

/// Decodes a `Common.Constant` into a Python object.
pub fn decode_constant(
    py: Python<'_>,
    reader: common_capnp::constant::Reader<'_>,
) -> Result<Py<PyAny>> {
    use common_capnp::constant::Which;
    match reader.which().map_err(capnp_err)? {
        Which::Real(v) => Ok(v.into_py_any(py)?),
        Which::Integer(v) => Ok(v.into_py_any(py)?),
        Which::Complex(c) => {
            let c = c.map_err(capnp_err)?;
            Ok(Complex64::new(c.get_real(), c.get_imag()).into_py_any(py)?)
        }
        Which::StringValue(s) => Ok(text_to_string(s.map_err(capnp_err)?)?.into_py_any(py)?),
        Which::RawBytesValue(b) => {
            let b = b.map_err(capnp_err)?;
            Ok(PyBytes::new(py, b).into_py_any(py)?)
        }
        Which::PythonValue(b) => {
            let b = b.map_err(capnp_err)?;
            Ok(laboneq_py_utils::constant_serializer::deserialize_json(py, b)?.unbind())
        }
    }
}

/// Decodes a `Values.NumericArray` into a NumPy array.
pub fn decode_numeric_array<'py>(
    py: Python<'py>,
    reader: values_capnp::numeric_array::Reader<'_>,
) -> Result<Bound<'py, PyAny>> {
    let data = reader.get_data().map_err(capnp_err)?;
    let dtype = reader.get_dtype().map_err(capnp_err)?;

    // `chunks_exact` below drops a trailing partial element, which would truncate the array
    // rather than report the malformed payload that produced it.
    let stride = match dtype {
        values_capnp::NumericDType::Float64 | values_capnp::NumericDType::Int64 => 8,
        values_capnp::NumericDType::Complex128 => 16,
        values_capnp::NumericDType::Unspecified => {
            return Err(laboneq_error!("NumericArray has unspecified dtype"));
        }
    };
    if !data.len().is_multiple_of(stride) {
        return Err(laboneq_error!(
            "NumericArray of {dtype:?} has {} bytes of data, not a multiple of {stride}",
            data.len()
        ));
    }

    match dtype {
        values_capnp::NumericDType::Float64 => {
            let (words, _) = data.as_chunks::<8>();
            let vals: Vec<f64> = words.iter().map(|&w| f64::from_le_bytes(w)).collect();
            Ok(PyArray1::from_vec(py, vals).into_bound_py_any(py)?)
        }
        values_capnp::NumericDType::Int64 => {
            let (words, _) = data.as_chunks::<8>();
            let vals: Vec<i64> = words.iter().map(|&w| i64::from_le_bytes(w)).collect();
            Ok(PyArray1::from_vec(py, vals).into_bound_py_any(py)?)
        }
        values_capnp::NumericDType::Complex128 => {
            let (pairs, _) = data.as_chunks::<8>().0.as_chunks::<2>();
            let vals: Vec<Complex64> = pairs
                .iter()
                .map(|&[re, im]| Complex64::new(f64::from_le_bytes(re), f64::from_le_bytes(im)))
                .collect();
            Ok(PyArray1::from_vec(py, vals).into_bound_py_any(py)?)
        }
        // Already rejected above, when the stride was resolved.
        values_capnp::NumericDType::Unspecified => {
            Err(laboneq_error!("NumericArray has unspecified dtype"))
        }
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::Builder;

    use super::*;

    /// Builds a `NumericArray` with the given raw `data`, bypassing any writer that would
    /// keep the length consistent with the dtype -- which is the point: these bytes stand in
    /// for a malformed or foreign-produced payload.
    fn numeric_array(
        dtype: values_capnp::NumericDType,
        data: &[u8],
    ) -> Builder<capnp::message::HeapAllocator> {
        let mut message = Builder::new_default();
        let mut builder = message.init_root::<values_capnp::numeric_array::Builder<'_>>();
        builder.set_dtype(dtype);
        builder.set_data(data);
        message
    }

    #[test]
    fn decodes_whole_elements() {
        let message = numeric_array(values_capnp::NumericDType::Float64, &[0u8; 16]);
        let reader = message.get_root_as_reader().unwrap();
        Python::attach(|py| {
            let array = decode_numeric_array(py, reader).expect("two whole float64 elements");
            assert_eq!(array.len().unwrap(), 2);
        });
    }

    /// A trailing partial element means the payload disagrees with its own dtype. Decoding
    /// it would silently drop the remainder, so it has to fail instead.
    #[test]
    fn rejects_data_that_is_not_a_whole_number_of_elements() {
        for (dtype, len) in [
            (values_capnp::NumericDType::Float64, 12),
            (values_capnp::NumericDType::Int64, 4),
            (values_capnp::NumericDType::Complex128, 24),
        ] {
            let message = numeric_array(dtype, &vec![0u8; len]);
            let reader = message.get_root_as_reader().unwrap();
            Python::attach(|py| {
                let err = decode_numeric_array(py, reader)
                    .expect_err("partial trailing element must not be dropped");
                assert!(
                    err.to_string().contains("not a multiple of"),
                    "unexpected error: {err}"
                );
            });
        }
    }
}
