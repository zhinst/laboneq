// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Helpers for `CompiledExperiment.metadata`.

use crate::compiled_experiment::v1::compiled_experiment_capnp::compiled_experiment;

/// Splits `Metadata.createdBy` ("<backend>/<version>") into its two halves.
///
/// The *first* separator is the boundary: a backend identifier should never contain '/', while a
/// version may.
///
/// A string without a separator is taken as a bare backend name of unknown version.
pub fn split_created_by(created_by: &str) -> (&str, &str) {
    created_by.split_once('/').unwrap_or((created_by, ""))
}

/// The backend identifier half of `Metadata.createdBy`.
pub fn backend_name(created_by: &str) -> &str {
    split_created_by(created_by).0
}

/// Reads just the producing backend's identifier out of a serialized message, without
/// decoding anything else.
pub fn peek_backend_name(bytes: &[u8]) -> capnp::Result<String> {
    let reader =
        capnp::serialize::read_message_from_flat_slice(&mut &*bytes, crate::reader_options())?;
    let root = reader.get_root::<compiled_experiment::Reader<'_>>()?;
    let created_by = root.get_metadata()?.get_created_by()?.to_str()?;
    Ok(backend_name(created_by).to_owned())
}

#[cfg(test)]
mod tests {
    use super::{backend_name, peek_backend_name, split_created_by};
    use crate::compiled_experiment::v1::compiled_experiment_capnp::compiled_experiment;

    #[test]
    fn backend_name_strips_the_version() {
        assert_eq!(backend_name("backend/2.60.0"), "backend");
        assert_eq!(backend_name("backend/2.60.0.dev1+g0/deadbeef"), "backend");
        assert_eq!(backend_name("backend"), "backend");
        assert_eq!(backend_name(""), "");
    }

    #[test]
    fn created_by_splits_at_the_first_separator() {
        assert_eq!(split_created_by("backend/2.60.0"), ("backend", "2.60.0"));
        // A version may carry further separators; they belong to the version, not the name.
        assert_eq!(
            split_created_by("backend/2.60.0.dev1+g0/deadbeef"),
            ("backend", "2.60.0.dev1+g0/deadbeef")
        );
        assert_eq!(split_created_by("backend"), ("backend", ""));
        assert_eq!(split_created_by(""), ("", ""));
    }

    #[test]
    fn peek_reads_the_backend_out_of_a_serialized_message() {
        let mut message = capnp::message::Builder::new_default();
        message
            .init_root::<compiled_experiment::Builder<'_>>()
            .init_metadata()
            .set_created_by("backend/26.10.0-dev");
        let bytes = capnp::serialize::write_message_to_words(&message);

        assert_eq!(peek_backend_name(&bytes).unwrap(), "backend");
    }
}
