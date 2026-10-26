// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Checked conversions for capnp field types.
//!
//! Capnp fields are fixed width (`UInt32`, `Int32`, ...) while the compiler works in
//! `usize`/`i64`/`f64`. A plain `as` cast silently wraps or truncates, producing artifacts
//! that only fail much later on the instrument. Blobs (`Data`, `Text`) hold at most
//! 512 MiB, and capnp panics beyond that. These helpers reject the value instead.

use std::any::type_name;
use std::error::Error;
use std::fmt::{self, Display};

/// Largest byte count of a capnp `Data` field: capnp lists hold fewer than 2^29 elements.
const MAX_DATA_LEN: usize = (1 << 29) - 1;
/// Largest byte count of a capnp `Text` field, which also stores a NUL terminator.
const MAX_TEXT_LEN: usize = MAX_DATA_LEN - 1;

/// A value that cannot be represented by the capnp field it was being written to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NarrowError {
    field: &'static str,
    reason: String,
}

impl NarrowError {
    /// The capnp field that rejected the value.
    pub fn field(&self) -> &'static str {
        self.field
    }

    fn out_of_range<U>(value: impl Display, field: &'static str) -> Self {
        Self {
            field,
            reason: format!("{value} does not fit into {}", type_name::<U>()),
        }
    }
}

impl Display for NarrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capnp field `{}`: {}", self.field, self.reason)
    }
}

impl Error for NarrowError {}

impl From<NarrowError> for capnp::Error {
    fn from(error: NarrowError) -> Self {
        capnp::Error::failed(error.to_string())
    }
}

/// Narrows `value` to a capnp field's fixed-width type.
///
/// `field` names the schema slot, so the error identifies it on its own. It is `'static` so
/// that a caller cannot pass a freshly formatted `String` into a hot serialization loop.
pub fn narrow<T: Copy + Display, U: TryFrom<T>>(
    value: T,
    field: &'static str,
) -> Result<U, NarrowError> {
    U::try_from(value).map_err(|_| NarrowError::out_of_range::<U>(value, field))
}

/// Narrows a whole-number `f64` to a capnp field's integer type, rejecting values that are
/// fractional, non-finite, or out of range.
pub fn narrow_f64<U: TryFrom<i128>>(value: f64, field: &'static str) -> Result<U, NarrowError> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(NarrowError {
            field,
            reason: format!("{value} is not a whole number"),
        });
    }
    // `as` saturates at the `i128` bounds, which the conversion then rejects.
    U::try_from(value as i128).map_err(|_| NarrowError::out_of_range::<U>(value, field))
}

/// Returns `value`, or an error when it is too large for a capnp `Data` field.
pub fn checked_data<'a>(value: &'a [u8], field: &'static str) -> Result<&'a [u8], NarrowError> {
    check_blob_len(value.len(), MAX_DATA_LEN, field).map(|()| value)
}

/// Returns `value`, or an error when it is too large for a capnp `Text` field.
pub fn checked_text<'a>(value: &'a str, field: &'static str) -> Result<&'a str, NarrowError> {
    check_blob_len(value.len(), MAX_TEXT_LEN, field).map(|()| value)
}

fn check_blob_len(len: usize, max: usize, field: &'static str) -> Result<(), NarrowError> {
    if len <= max {
        Ok(())
    } else {
        Err(NarrowError {
            field,
            reason: format!("{len} bytes exceed the limit of {max}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_TEXT_LEN, NarrowError, check_blob_len, checked_data, checked_text, narrow, narrow_f64,
    };

    #[test]
    fn narrows_values_in_range() {
        assert_eq!(narrow::<usize, u32>(u32::MAX as usize, "f"), Ok(u32::MAX));
        assert_eq!(narrow::<i64, u32>(0, "f"), Ok(0));
        assert_eq!(narrow_f64::<u32>(2e9, "f"), Ok(2_000_000_000));
    }

    #[test]
    fn rejects_values_out_of_range() {
        let cases = [
            narrow::<i64, u32>(-1, "acquisitionLength").unwrap_err(),
            narrow::<i64, u32>(i64::from(u32::MAX) + 1, "acquisitionLength").unwrap_err(),
            narrow_f64::<u32>(1e30, "acquisitionLength").unwrap_err(),
        ];
        for error in cases {
            assert_eq!(error.field(), "acquisitionLength");
            assert!(
                error.to_string().contains("does not fit into u32"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn rejects_f64_values_that_are_not_whole_numbers() {
        for value in [2.5, f64::NAN, f64::INFINITY] {
            let error = narrow_f64::<u32>(value, "samplingRate").unwrap_err();
            assert!(
                error.to_string().ends_with("is not a whole number"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn passes_blobs_within_the_limit() {
        assert_eq!(checked_data(b"elf", "bin"), Ok(&b"elf"[..]));
        assert_eq!(checked_text("seqc", "src"), Ok("seqc"));
        assert_eq!(check_blob_len(MAX_TEXT_LEN, MAX_TEXT_LEN, "src"), Ok(()));
    }

    #[test]
    fn rejects_blobs_over_the_limit() {
        let error = check_blob_len(MAX_TEXT_LEN + 1, MAX_TEXT_LEN, "src").unwrap_err();
        assert_eq!(
            error.to_string(),
            "capnp field `src`: 536870911 bytes exceed the limit of 536870910"
        );
    }

    #[test]
    fn error_message_names_the_field() {
        let error = narrow::<i64, u32>(-1, "acquisitionLength").unwrap_err();
        assert_eq!(
            error.to_string(),
            "capnp field `acquisitionLength`: -1 does not fit into u32"
        );
    }

    /// Backend writers return `capnp::Result`, so `?` must carry the message across.
    #[test]
    fn converts_into_a_capnp_error() {
        let error: capnp::Error = narrow::<i64, u32>(-1, "acquisitionLength")
            .unwrap_err()
            .into();
        assert_eq!(error.kind, capnp::ErrorKind::Failed);
        assert!(error.to_string().contains("acquisitionLength"));
    }

    #[test]
    fn narrow_error_is_a_std_error() {
        fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
        assert_error::<NarrowError>();
    }
}
