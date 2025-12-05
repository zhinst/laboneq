// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use pyo3::{intern, prelude::*, types::PyBytes};

pub(super) type ObjectUid = u64;

/// Python object interner.
///
/// The interner assigns a unique UID to each distinct Python object based on its pickled representation.
///
/// The UID is not stable across different runs.
/// It is only stable within a single run of the program.
pub(crate) struct PyObjectInterner<K: Copy + Eq + std::hash::Hash + From<ObjectUid>> {
    values: HashMap<K, Py<PyAny>>,
}

impl<K: Copy + Eq + std::hash::Hash + From<ObjectUid>> PyObjectInterner<K> {
    pub(crate) fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Get the UID for the given Python object, interning it if necessary.
    pub(crate) fn get_or_intern(&mut self, value: &Bound<'_, PyAny>) -> PyResult<K> {
        let uid = intern_py_object(value)?.into();
        self.values
            .entry(uid)
            .or_insert_with(|| value.clone().unbind());
        Ok(uid)
    }

    /// Resolve the Python object associated with the given UID.
    pub(crate) fn resolve(&self, key: &K) -> Option<&Py<PyAny>> {
        self.values.get(key)
    }
}

fn intern_py_object(value: &Bound<'_, PyAny>) -> PyResult<ObjectUid> {
    let py: Python<'_> = value.py();
    let pickled_bytes = value
        .py()
        .import(intern!(py, "pickle"))
        .and_then(|pickle_module| pickle_module.getattr(intern!(py, "dumps")))
        .and_then(|dumps_func| dumps_func.call1((value,)))?;
    let hashlib = value
        .py()
        .import(intern!(py, "hashlib"))
        .and_then(|hashlib_module| hashlib_module.getattr(intern!(py, "sha256")))?;
    let sha256 = hashlib.call1((pickled_bytes,))?;
    let digest = sha256.call_method0(intern!(py, "digest"))?;
    let digest: &[u8] = digest.cast::<PyBytes>()?.as_bytes();
    let id = bytes_to_u64_be(digest);
    Ok(id)
}

fn bytes_to_u64_be(digest: &[u8]) -> u64 {
    // If digest is less than 8 bytes, pad with zeros at the end.
    // Should not happen with sha256, but just in case.
    u64::from_be_bytes(digest[..digest.len().min(8)].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use pyo3::{ffi::c_str, prelude::*};

    use super::PyObjectInterner;

    fn setup_py_objects<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let code = c_str!(
            r#"
integer_positive = 42
integer_negative = -42
complex_object_0 = {"a": [1, 2]}
complex_object_1 = {"a": [1, 3]}
none = None
zero = 0
false = False
true = True
"#
        );
        let test_module: Py<PyAny> = PyModule::from_code(py, code, c_str!(""), c_str!(""))
            .unwrap()
            .into();
        let globals = test_module.bind(py);
        globals.clone()
    }

    #[test]
    fn test_interning_roundtrip() {
        Python::attach(|py| {
            let py_objects = setup_py_objects(py);
            let mut interner: PyObjectInterner<u64> = PyObjectInterner::new();

            let integer_positive = py_objects.getattr("integer_positive").unwrap();
            let uid1 = interner.get_or_intern(&integer_positive).unwrap();
            assert!(
                integer_positive
                    .eq(interner.resolve(&uid1).unwrap())
                    .unwrap()
            );

            let complex_object_0 = py_objects.getattr("complex_object_0").unwrap();
            let uid2 = interner.get_or_intern(&complex_object_0).unwrap();
            assert!(
                complex_object_0
                    .eq(interner.resolve(&uid2).unwrap())
                    .unwrap()
            );
        });
    }

    #[test]
    fn test_interning_uniqueness() {
        Python::attach(|py| {
            let py_objects = setup_py_objects(py);
            let mut interner: PyObjectInterner<u64> = PyObjectInterner::new();
            let integer_positive = py_objects.getattr("integer_positive").unwrap();
            let integer_negative = py_objects.getattr("integer_negative").unwrap();
            let uid1 = interner.get_or_intern(&integer_positive).unwrap();
            let uid2 = interner.get_or_intern(&integer_negative).unwrap();
            assert_ne!(uid1, uid2);

            let complex_object_0 = py_objects.getattr("complex_object_0").unwrap();
            let complex_object_1 = py_objects.getattr("complex_object_1").unwrap();
            let uid3 = interner.get_or_intern(&complex_object_0).unwrap();
            let uid4 = interner.get_or_intern(&complex_object_1).unwrap();
            assert_ne!(uid3, uid4);
        });
    }

    #[test]
    fn test_same_object_returns_same_uid() {
        Python::attach(|py| {
            let py_objects = setup_py_objects(py);
            let mut interner: PyObjectInterner<u64> = PyObjectInterner::new();
            let py_object = py_objects.getattr("complex_object_0").unwrap();
            let uid1 = interner.get_or_intern(&py_object).unwrap();
            let uid2 = interner.get_or_intern(&py_object).unwrap();
            let uid3 = interner.get_or_intern(&py_object).unwrap();
            assert_eq!(uid1, uid2);
            assert_eq!(uid2, uid3);
        });
    }

    #[test]
    fn test_edge_cases() {
        Python::attach(|py| {
            let py_objects = setup_py_objects(py);
            let mut interner: PyObjectInterner<u64> = PyObjectInterner::new();

            let none = py_objects.getattr("none").unwrap();
            let uid_none = interner.get_or_intern(&none).unwrap();
            assert!(none.eq(interner.resolve(&uid_none).unwrap()).unwrap());

            let zero = py_objects.getattr("zero").unwrap();
            let uid_zero = interner.get_or_intern(&zero).unwrap();
            assert!(zero.eq(interner.resolve(&uid_zero).unwrap()).unwrap());

            let false_value = py_objects.getattr("false").unwrap();
            let uid_false = interner.get_or_intern(&false_value).unwrap();
            assert!(
                false_value
                    .eq(interner.resolve(&uid_false).unwrap())
                    .unwrap()
            );

            let true_value = py_objects.getattr("true").unwrap();
            let uid_true = interner.get_or_intern(&true_value).unwrap();
            assert!(true_value.eq(interner.resolve(&uid_true).unwrap()).unwrap());

            assert_ne!(uid_false, uid_true);
            assert_ne!(uid_none, uid_false);
            assert_ne!(uid_none, uid_true);
            assert_ne!(uid_false, uid_true);
            assert_ne!(uid_none, uid_zero);
            assert_ne!(uid_false, uid_zero);
            assert_ne!(uid_true, uid_zero);
        });
    }
}
