// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use twox_hash::XxHash3_64;

use crate::constant_serializer::deserialize_json;

/// Serialized payload of a value the compiler carries through opaquely.
#[derive(Hash)]
pub enum PyObjectPayload<'a> {
    /// A Python `str`.
    Str(&'a str),
    /// JSON written by [`crate::constant_serializer::serialize_json`].
    Json(&'a [u8]),
    /// Opaque `bytes` supplied by the user.
    RawBytes(&'a [u8]),
}

/// Owned counterpart of [`PyObjectPayload`], as retained by [`PyObjectStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnedPyObjectPayload {
    Str(String),
    Json(Vec<u8>),
    RawBytes(Vec<u8>),
}

impl From<&PyObjectPayload<'_>> for OwnedPyObjectPayload {
    fn from(payload: &PyObjectPayload<'_>) -> Self {
        match *payload {
            PyObjectPayload::Str(value) => Self::Str(value.to_owned()),
            PyObjectPayload::Json(data) => Self::Json(data.to_vec()),
            PyObjectPayload::RawBytes(data) => Self::RawBytes(data.to_vec()),
        }
    }
}

struct StoredPyObject {
    payload: OwnedPyObjectPayload,
    value: Py<PyAny>,
}

/// Content-addressed store of Python objects the compiler carries through
/// opaquely.
///
/// Callers submit a serialized payload and get back a UID.
/// The store deserializes the payload into a Python object only once, and returns the same UID for
/// repeated submissions of the same payload. A UID resolves back to either the original payload
/// ([`resolve`]) or the Python object ([`resolve_to_python`]). Keeping the payload lets a
/// consumer re-emit the value in exactly the form it arrived in, without a Python round trip.
///
/// UIDs are process-local -- never serialized, valid only within the run that
/// produced them -- so the hash needs no cross-version stability.
///
/// [`resolve`]: PyObjectStore::resolve
/// [`resolve_to_python`]: PyObjectStore::resolve_to_python
#[derive(Default)]
pub struct PyObjectStore<K: Copy + Eq + Hash + From<u64>> {
    values: HashMap<K, StoredPyObject>,
}

impl<K: Copy + Eq + Hash + From<u64>> PyObjectStore<K> {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Store the Python value denoted by `payload` and return its UID.
    ///
    /// The value is only reconstructed if the payload is not already stored, so a
    /// payload referenced from many places is deserialized once.
    pub fn store(&mut self, py: Python<'_>, payload: PyObjectPayload<'_>) -> PyResult<K> {
        let uid = K::from(uid_of(&payload));
        if let Entry::Vacant(entry) = self.values.entry(uid) {
            let owned = OwnedPyObjectPayload::from(&payload);
            entry.insert(StoredPyObject {
                payload: owned,
                value: payload.into_py_object(py)?,
            });
        }
        Ok(uid)
    }

    /// Resolve the payload from which the value associated with the given UID was stored.
    pub fn resolve(&self, key: &K) -> Option<&OwnedPyObjectPayload> {
        self.values.get(key).map(|stored| &stored.payload)
    }

    /// Resolve the Python object associated with the given UID.
    pub fn resolve_to_python(&self, key: &K) -> Option<&Py<PyAny>> {
        self.values.get(key).map(|stored| &stored.value)
    }
}

fn uid_of(payload: &PyObjectPayload<'_>) -> u64 {
    let mut hasher = XxHash3_64::new();
    payload.hash(&mut hasher);
    hasher.finish()
}

impl PyObjectPayload<'_> {
    fn into_py_object(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Str(value) => value.into_py_any(py),
            Self::Json(bytes) => Ok(deserialize_json(py, bytes)?.unbind()),
            Self::RawBytes(bytes) => Ok(PyBytes::new(py, bytes).into_any().unbind()),
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    use super::{OwnedPyObjectPayload, PyObjectPayload, PyObjectStore};

    const JSON: &[u8] = b"[1, 2]";

    fn new_store() -> PyObjectStore<u64> {
        PyObjectStore::new()
    }

    #[test]
    fn stores_and_resolves_each_payload_kind() {
        Python::attach(|py| {
            let mut store = new_store();

            let str_uid = store.store(py, PyObjectPayload::Str("x")).unwrap();
            let json_uid = store.store(py, PyObjectPayload::Json(JSON)).unwrap();
            let bytes_uid = store.store(py, PyObjectPayload::RawBytes(b"x")).unwrap();

            let resolved_str = store.resolve_to_python(&str_uid).unwrap();
            assert!(resolved_str.bind(py).eq("x").unwrap());
            let resolved_json = store.resolve_to_python(&json_uid).unwrap();
            assert!(resolved_json.bind(py).eq(vec![1, 2]).unwrap());
            let resolved_bytes = store.resolve_to_python(&bytes_uid).unwrap();
            assert!(resolved_bytes.bind(py).eq(b"x".as_slice()).unwrap());

            assert_eq!(
                store.resolve(&str_uid),
                Some(&OwnedPyObjectPayload::Str("x".to_owned()))
            );
            assert_eq!(
                store.resolve(&json_uid),
                Some(&OwnedPyObjectPayload::Json(JSON.to_vec()))
            );
            assert_eq!(
                store.resolve(&bytes_uid),
                Some(&OwnedPyObjectPayload::RawBytes(b"x".to_vec()))
            );
        });
    }

    #[test]
    fn payload_distinguishes_kinds_yielding_equal_objects() {
        // The reason the payload is kept: both deserialize to the same Python `str`, so
        // only the stored payload tells a serializer which form to write back.
        Python::attach(|py| {
            let mut store = new_store();

            let str_uid = store.store(py, PyObjectPayload::Str("x")).unwrap();
            let json_uid = store.store(py, PyObjectPayload::Json(b"\"x\"")).unwrap();

            let from_str = store.resolve_to_python(&str_uid).unwrap().bind(py);
            let from_json = store.resolve_to_python(&json_uid).unwrap().bind(py);
            assert!(from_str.eq(from_json).unwrap());

            assert_eq!(
                store.resolve(&str_uid),
                Some(&OwnedPyObjectPayload::Str("x".to_owned()))
            );
            assert_eq!(
                store.resolve(&json_uid),
                Some(&OwnedPyObjectPayload::Json(b"\"x\"".to_vec()))
            );
        });
    }

    #[test]
    fn separates_payload_kinds_sharing_bytes() {
        // `Json` and `RawBytes` are both `&[u8]`, so without the variant
        // discriminant these would share a UID and one would shadow the other.
        // Reachable whenever a user's raw bytes happen to spell valid JSON.
        Python::attach(|py| {
            let mut store = new_store();

            let json_uid = store.store(py, PyObjectPayload::Json(JSON)).unwrap();
            let bytes_uid = store.store(py, PyObjectPayload::RawBytes(JSON)).unwrap();
            assert_ne!(json_uid, bytes_uid);

            let str_uid = store.store(py, PyObjectPayload::Str("x")).unwrap();
            let raw_uid = store.store(py, PyObjectPayload::RawBytes(b"x")).unwrap();
            assert_ne!(str_uid, raw_uid);
        });
    }

    #[test]
    fn equal_payloads_share_one_entry() {
        // The point of the store: repeated references resolve to one object.
        Python::attach(|py| {
            let mut store = new_store();

            let first = store.store(py, PyObjectPayload::Json(JSON)).unwrap();
            let again = store.store(py, PyObjectPayload::Json(JSON)).unwrap();
            let other = store.store(py, PyObjectPayload::Json(b"[1, 3]")).unwrap();

            assert_eq!(first, again);
            assert_ne!(first, other);
        });
    }
}
