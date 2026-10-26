// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Shared Cap'n Proto serialization primitives for `CompiledExperiment.artifacts` writers.
use std::fmt::Debug;

use rustc_hash::FxHashMap;

use laboneq_capnp::compiled_experiment::v1::{common_capnp, values_capnp};
use laboneq_capnp::convert::narrow;
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_common::types::Literal;
use laboneq_dsl::operation::ExternalOrValue;
use laboneq_dsl::types::{
    ExternalParameterUid, HandleUid, NumericLiteral, ParameterUid, ValueOrParameter,
};
use laboneq_py_utils::py_object_store::{OwnedPyObjectPayload, PyObjectStore};
use numeric_array::NumericArray;

/// Bundles the state that a [`CompilerArtifact`](crate::compiler_backend::CompilerArtifact)
/// needs to serialize its own capnp artifacts consistently with the rest of
/// `CompiledExperiment`: the symbol table, and the `parameters`/`acquisitionHandles`
/// index tables built while serializing those shared tables.
pub struct ArtifactSerializerContext<'a> {
    id_store: &'a NamedIdStore,
    external_parameters: &'a PyObjectStore<ExternalParameterUid>,
    parameter_indices: &'a FxHashMap<ParameterUid, usize>,
    handle_indices: &'a FxHashMap<HandleUid, u32>,
}

impl<'a> ArtifactSerializerContext<'a> {
    pub(crate) fn new(
        id_store: &'a NamedIdStore,
        external_parameters: &'a PyObjectStore<ExternalParameterUid>,
        parameter_indices: &'a FxHashMap<ParameterUid, usize>,
        handle_indices: &'a FxHashMap<HandleUid, u32>,
    ) -> Self {
        Self {
            id_store,
            external_parameters,
            parameter_indices,
            handle_indices,
        }
    }

    /// Resolves an `uid` into the string.
    pub fn resolve_symbol(&self, uid: impl Into<NamedId> + Debug + Copy) -> capnp::Result<&str> {
        resolve_symbol(self.id_store, uid)
    }

    /// `CompiledExperiment.parameters` index for a named sweep parameter.
    ///
    /// Fails only if `uid` was never indexed, i.e. never if `parameter_indices` is built
    /// correctly.
    pub fn parameter_ref(&self, uid: ParameterUid) -> capnp::Result<u32> {
        let index = self.parameter_indices.get(&uid).ok_or_else(|| {
            capnp::Error::failed(format!(
                "Internal error: Parameter {uid:?} is missing from `parameters`."
            ))
        })?;
        Ok(narrow(*index, "parameterRef")?)
    }

    /// `CompiledExperiment.acquisitionHandles` index for a pre-declared handle.
    ///
    /// Fails only if `handle` was never pre-declared, i.e. never if `handle_indices` is built
    /// correctly.
    pub fn handle_ref(&self, handle: HandleUid) -> capnp::Result<u32> {
        self.handle_indices.get(&handle).copied().ok_or_else(|| {
            capnp::Error::failed(format!(
                "Internal error: {handle:?} references an acquisition handle that was not pre-declared."
            ))
        })
    }

    /// Serializes a [`ValueOrParameter`] of [`Literal`] into a Cap'n Proto `Value` builder.
    pub fn serialize_value_or_parameter_literal(
        &self,
        value: ValueOrParameter<Literal>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> capnp::Result<()> {
        match value {
            ValueOrParameter::Value(v) => self.serialize_literal(v, builder),
            ValueOrParameter::Parameter(uid) => {
                builder
                    .reborrow()
                    .set_parameter_ref(self.parameter_ref(uid)?);
            }
            ValueOrParameter::ResolvedParameter { value, .. } => {
                self.serialize_literal(value, builder);
            }
        }
        Ok(())
    }

    /// Serializes an [`ExternalOrValue`] into a Cap'n Proto `Value` builder.
    pub fn serialize_external_or_value(
        &self,
        value: ExternalOrValue,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> capnp::Result<()> {
        match value {
            ExternalOrValue::ValueOrParameter(vop) => {
                self.serialize_value_or_parameter_numeric(vop, builder)?;
            }
            ExternalOrValue::ExternalParameter(val) => {
                let payload = self.external_parameters.resolve(&val).ok_or_else(|| {
                    capnp::Error::failed(format!(
                        "Internal error: External parameter {val:?} is missing from the object store."
                    ))
                })?;
                let mut constant = builder.reborrow().init_constant();
                match payload {
                    OwnedPyObjectPayload::Str(value) => constant.set_string_value(value.as_str()),
                    OwnedPyObjectPayload::Json(data) => constant.set_python_value(data),
                    OwnedPyObjectPayload::RawBytes(data) => constant.set_raw_bytes_value(data),
                }
            }
            ExternalOrValue::None => {
                builder.reborrow().set_none(());
            }
        }
        Ok(())
    }

    /// Serializes a [`ValueOrParameter`] of [`NumericLiteral`] into a Cap'n Proto `Value` builder.
    pub fn serialize_value_or_parameter_numeric(
        &self,
        value: ValueOrParameter<NumericLiteral>,
        builder: &mut common_capnp::value::Builder<'_>,
    ) -> capnp::Result<()> {
        match value {
            ValueOrParameter::Value(v) => self.serialize_numeric_literal(v, builder),
            ValueOrParameter::Parameter(uid) => {
                builder
                    .reborrow()
                    .set_parameter_ref(self.parameter_ref(uid)?);
            }
            ValueOrParameter::ResolvedParameter { value, .. } => {
                self.serialize_numeric_literal(value, builder);
            }
        }
        Ok(())
    }

    /// Serializes a [`NumericLiteral`] into a Cap'n Proto `Value` builder.
    pub fn serialize_numeric_literal(
        &self,
        value: NumericLiteral,
        builder: &mut common_capnp::value::Builder<'_>,
    ) {
        let mut constant = builder.reborrow().init_constant();
        match value {
            NumericLiteral::Int(v) => constant.set_integer(v),
            NumericLiteral::Float(v) => constant.set_real(v),
            NumericLiteral::Complex(v) => {
                let mut c = constant.init_complex();
                c.set_real(v.re);
                c.set_imag(v.im);
            }
        }
    }

    /// Serializes a [`Literal`] into a Cap'n Proto `Value` builder.
    pub fn serialize_literal(
        &self,
        value: Literal,
        builder: &mut common_capnp::value::Builder<'_>,
    ) {
        let mut constant = builder.reborrow().init_constant();
        match value {
            Literal::Integer(v) => constant.set_integer(v),
            Literal::Real(v) => constant.set_real(v),
            Literal::Complex(v) => {
                let mut c = constant.init_complex();
                c.set_real(v.re);
                c.set_imag(v.im);
            }
            Literal::Text(s) => constant.set_string_value(&s),
        }
    }
}

/// Writes a [`NumericArray`] into a `Values.NumericArray` builder.
pub fn write_numeric_array(
    mut builder: values_capnp::numeric_array::Builder<'_>,
    array: &NumericArray,
) -> capnp::Result<()> {
    match array {
        NumericArray::Float64(vals) => {
            builder.set_length(narrow(vals.len(), "length")?);
            builder.set_dtype(values_capnp::NumericDType::Float64);
            write_elements(builder, vals, |v| v.to_le_bytes())?;
        }
        NumericArray::Integer64(vals) => {
            builder.set_length(narrow(vals.len(), "length")?);
            builder.set_dtype(values_capnp::NumericDType::Int64);
            write_elements(builder, vals, |v| v.to_le_bytes())?;
        }
        NumericArray::Complex64(vals) => {
            builder.set_length(narrow(vals.len(), "length")?);
            builder.set_dtype(values_capnp::NumericDType::Complex128);
            write_elements(builder, vals, |v| {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&v.re.to_le_bytes());
                bytes[8..].copy_from_slice(&v.im.to_le_bytes());
                bytes
            })?;
        }
    }
    Ok(())
}

/// Encodes each of `vals` as `N` bytes via `encode`, into the array's `data` field.
fn write_elements<T, const N: usize>(
    builder: values_capnp::numeric_array::Builder<'_>,
    vals: &[T],
    encode: impl Fn(&T) -> [u8; N],
) -> capnp::Result<()> {
    // A capnp Data field holds at most 4 GiB.
    let data = builder.init_data(narrow(vals.len() * N, "data")?);
    for (slot, value) in data.as_chunks_mut::<N>().0.iter_mut().zip(vals) {
        *slot = encode(value);
    }
    Ok(())
}

/// Resolves an interned symbol into the string.
///
/// Fails only if `uid` is not found in the store, i.e. never if the
/// `NamedIdStore` is built correctly.
pub(crate) fn resolve_symbol(
    id_store: &NamedIdStore,
    uid: impl Into<NamedId> + Debug + Copy,
) -> capnp::Result<&str> {
    // `ok_or_else`, so the message is only formatted on the (never taken) error path.
    id_store
        .resolve(uid.into())
        .ok_or_else(|| capnp::Error::failed(format!("Internal error: Failed to resolve {uid:?}.")))
}

#[cfg(test)]
mod tests {
    use capnp::message::{Builder, HeapAllocator};
    use laboneq_py_utils::py_object_store::PyObjectPayload;
    use pyo3::Python;

    use super::*;

    fn write_array(array: &NumericArray) -> Builder<HeapAllocator> {
        let mut message = Builder::new_default();
        write_numeric_array(
            message.init_root::<values_capnp::numeric_array::Builder<'_>>(),
            array,
        )
        .expect("array fits the capnp field widths");
        message
    }

    /// Asserts the on-wire encoding element by element, since `data` is opaque bytes, given
    /// meaning only by the deserializer's dtype-driven chunking -- a wrong element
    /// width or byte order would decode into plausible-looking garbage rather than fail.
    #[track_caller]
    fn assert_encodes_to(
        array: &NumericArray,
        dtype: values_capnp::NumericDType,
        expected: &[[u8; 8]],
    ) {
        let message = write_array(array);
        let reader = message
            .get_root_as_reader::<values_capnp::numeric_array::Reader<'_>>()
            .unwrap();

        assert_eq!(reader.get_dtype().unwrap(), dtype);
        let data = reader.get_data().unwrap();
        let words = data.as_chunks::<8>().0.to_vec();
        assert_eq!(words, expected);
        // `length` counts elements, not bytes: complex packs two words per element.
        assert_eq!(
            data.len(),
            reader.get_length() as usize * element_width(dtype)
        );
    }

    fn element_width(dtype: values_capnp::NumericDType) -> usize {
        match dtype {
            values_capnp::NumericDType::Complex128 => 16,
            _ => 8,
        }
    }

    #[test]
    fn numeric_arrays_encode_as_little_endian_elements() {
        use values_capnp::NumericDType;

        assert_encodes_to(
            &NumericArray::Integer64(vec![1, -2]),
            NumericDType::Int64,
            &[1i64.to_le_bytes(), (-2i64).to_le_bytes()],
        );
        assert_encodes_to(
            &NumericArray::Float64(vec![1.5, -0.25]),
            NumericDType::Float64,
            &[1.5f64.to_le_bytes(), (-0.25f64).to_le_bytes()],
        );
        // Real and imaginary parts must stay in that order within each element.
        assert_encodes_to(
            &NumericArray::Complex64(vec![num_complex::Complex::new(1.5, -0.25)]),
            NumericDType::Complex128,
            &[1.5f64.to_le_bytes(), (-0.25f64).to_le_bytes()],
        );
    }

    #[test]
    fn empty_numeric_array_writes_a_zero_length_data_field() {
        let message = write_array(&NumericArray::Float64(vec![]));
        let reader = message
            .get_root_as_reader::<values_capnp::numeric_array::Reader<'_>>()
            .unwrap();

        assert_eq!(reader.get_length(), 0);
        assert!(reader.get_data().unwrap().is_empty());
    }

    /// Stores an opaque payload, then serializes a reference to it back out to capnp.
    fn write_external(payload: PyObjectPayload<'_>) -> Builder<HeapAllocator> {
        Python::attach(|py| {
            let mut external_parameters = PyObjectStore::new();
            let uid = external_parameters.store(py, payload).unwrap();
            let id_store = NamedIdStore::new();
            let (parameter_indices, handle_indices) = (Default::default(), Default::default());
            let ctx = ArtifactSerializerContext::new(
                &id_store,
                &external_parameters,
                &parameter_indices,
                &handle_indices,
            );

            let mut message = Builder::new_default();
            let mut builder = message.init_root::<common_capnp::value::Builder<'_>>();
            ctx.serialize_external_or_value(ExternalOrValue::ExternalParameter(uid), &mut builder)
                .unwrap();
            message
        })
    }

    fn constant_of(message: &Builder<HeapAllocator>) -> common_capnp::constant::Reader<'_> {
        let reader = message
            .get_root_as_reader::<common_capnp::value::Reader<'_>>()
            .unwrap();
        match reader.which().unwrap() {
            common_capnp::value::Which::Constant(constant) => constant.unwrap(),
            _ => panic!("expected a constant"),
        }
    }

    /// `Json` is not covered here: deserializing it imports the `laboneq` Python package,
    /// which deadlocks against this crate's other Python tests under cargo's parallel
    /// harness. The payload it stores is covered in `laboneq_py_utils::py_object_store`.
    #[test]
    fn opaque_values_are_written_back_as_the_payload_they_arrived_as() {
        // Without the stored payload, the `Str` case would be indistinguishable from a
        // JSON-encoded string once deserialized into Python.
        use common_capnp::constant::Which;

        let message = write_external(PyObjectPayload::Str("x"));
        match constant_of(&message).which().unwrap() {
            Which::StringValue(value) => assert_eq!(value.unwrap().to_str().unwrap(), "x"),
            _ => panic!("expected stringValue"),
        }

        let message = write_external(PyObjectPayload::RawBytes(b"\x00\xff"));
        match constant_of(&message).which().unwrap() {
            Which::RawBytesValue(data) => assert_eq!(data.unwrap(), b"\x00\xff"),
            _ => panic!("expected rawBytesValue"),
        }
    }
}
