// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic handling of `DeviceSetup.setupDescription`.
//!
//! `setupDescription` is an `AnyPointer`: its concrete schema belongs to the backend that
//! targets the setup, named by `DeviceSetup.backend`. The core compiler never decodes it. On
//! the way out it forwards the backend's own pointer straight into the schema; on the way in
//! it hands the backend an opaque [`SetupDescriptionPayload`], which the backend decodes with
//! its own schema in [`preprocess_experiment`].
//!
//! [`preprocess_experiment`]: crate::compiler_backend::CompilerBackend::preprocess_experiment

use capnp::any_pointer;
use capnp::message::{HeapAllocator, TypedBuilder};
use capnp::traits::Owned;

use crate::error::{Error, Result};

/// An undecoded `DeviceSetup.setupDescription`, tagged with the backend that owns its schema.
///
/// The pointer is held as the root of a message of its own, because it has to outlive the
/// experiment message it was read from.
pub struct SetupDescriptionPayload {
    backend: String,
    message: TypedBuilder<any_pointer::Owned, HeapAllocator>,
}

impl SetupDescriptionPayload {
    /// Copies the `AnyPointer` written by `backend` out of an experiment message.
    pub fn from_pointer(backend: &str, pointer: any_pointer::Reader<'_>) -> Result<Self> {
        let mut message = TypedBuilder::<any_pointer::Owned>::new_default();
        message
            .set_root(pointer)
            .map_err(|e| Error::new(format!("Failed to copy the setup description: {e}")))?;
        Ok(Self {
            backend: backend.to_owned(),
            message,
        })
    }

    /// Decodes the payload as `T`, the root struct of `backend`'s setup-description schema.
    ///
    /// # Errors
    /// Returns an error if the payload was written for a different backend, or if it does not
    /// hold a struct of `T`'s shape.
    pub fn read_as<T: Owned>(&self, backend: &str) -> Result<T::Reader<'_>> {
        if self.backend != backend {
            return Err(Error::new(format!(
                "Device setup targets the '{}' backend, but the compiler was invoked for the \
                 '{backend}' backend",
                self.backend
            )));
        }
        self.message
            .get_root_as_reader()
            .and_then(|pointer| pointer.get_as::<T::Reader<'_>>())
            .map_err(|e| Error::new(format!("Failed to read the setup description: {e}")))
    }

    /// Backend identifier recorded in `DeviceSetup.backend`.
    pub fn backend(&self) -> &str {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use laboneq_capnp::pulse::v1::common_capnp::string_entry;

    use super::*;

    /// Stands in for a backend's setup-description message: the pointer is opaque to the core,
    /// so any struct type exercises the same code path.
    fn setup_description() -> TypedBuilder<any_pointer::Owned> {
        let mut message = capnp::message::Builder::new_default();
        let mut root = message.init_root::<string_entry::Builder<'_>>();
        root.set_key("uid");
        root.set_value("device_0");
        message.into_typed()
    }

    fn payload(backend: &str) -> SetupDescriptionPayload {
        let message = setup_description();
        SetupDescriptionPayload::from_pointer(backend, message.get_root_as_reader().unwrap())
            .unwrap()
    }

    /// The pointer survives the trip through an experiment message untouched, so that what a
    /// backend writes is what it reads back.
    #[test]
    fn setup_description_round_trips_through_an_experiment_message() {
        let message = setup_description();
        let mut experiment = capnp::message::Builder::new_default();
        experiment
            .init_root::<any_pointer::Builder<'_>>()
            .set_as::<any_pointer::Owned>(message.get_root_as_reader().unwrap())
            .unwrap();

        let pointer = experiment
            .get_root_as_reader::<any_pointer::Reader<'_>>()
            .unwrap();
        let restored = SetupDescriptionPayload::from_pointer("qccs", pointer).unwrap();

        let reader = restored.read_as::<string_entry::Owned>("qccs").unwrap();
        assert_eq!(reader.get_key().unwrap().to_str().unwrap(), "uid");
        assert_eq!(reader.get_value().unwrap().to_str().unwrap(), "device_0");
    }

    /// An `AnyPointer` carries no type information, so decoding a foreign payload would
    /// silently yield garbage. The backend tag is what stops it.
    #[test]
    fn decoding_another_backends_payload_is_refused() {
        let err = payload("zqcs")
            .read_as::<string_entry::Owned>("qccs")
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Device setup targets the 'zqcs' backend"),
            "unexpected error: {err}"
        );
    }
}
