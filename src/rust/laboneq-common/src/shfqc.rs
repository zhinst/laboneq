// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub const VIRTUAL_SHFSG_UID_SUFFIX: &str = "_sg";

/// Format the UID for the virtual SHFSG device created from splitting an SHFQC device.
///
/// The UID is derived from the original SHFQC instrument UID by appending a suffix to indicate it's the SG part of the split device.
/// This is necessary to ensure the new SHFSG device has a unique UID.
pub fn to_sg_uid(instrument_uid: &str) -> String {
    format!("{}{}", instrument_uid, VIRTUAL_SHFSG_UID_SUFFIX)
}

/// Remove the suffix from a virtual SHFSG UID to retrieve the base UID of the original SHFQC device.
///
/// Only call on UIDs that were produced by [`to_sg_uid`].
pub fn to_base_uid(instrument_uid: &str) -> String {
    instrument_uid
        .strip_suffix(VIRTUAL_SHFSG_UID_SUFFIX)
        .unwrap_or(instrument_uid)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_sg_uid() {
        let base_uid = "SHFQC_1234";
        let uid_round_trip = to_base_uid(&to_sg_uid(base_uid));
        assert_eq!(uid_round_trip, base_uid);

        let base_uid = format!("SHFQC_1234{}", VIRTUAL_SHFSG_UID_SUFFIX);
        let uid_round_trip = to_base_uid(&to_sg_uid(&base_uid));
        assert_eq!(uid_round_trip, base_uid);
    }
}
