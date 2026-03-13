// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

/// Represents the feature options for a device.
///
/// The options are normalized to uppercase.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceOptions {
    options: Arc<Vec<String>>,
}

impl std::ops::Deref for DeviceOptions {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl DeviceOptions {
    pub fn new(mut options: Vec<String>) -> Self {
        Self::normalize_options(&mut options);
        DeviceOptions {
            options: Arc::new(options.into_iter().collect()),
        }
    }

    /// Checks if a specific option is present in the device options.
    ///
    /// The check is case-sensitive, use uppercase.
    pub fn contains<S: AsRef<str>>(&self, option: S) -> bool {
        let option_str = option.as_ref();
        self.options.iter().any(|s| s == option_str)
    }

    fn normalize_options(options: &mut Vec<String>) {
        options.retain(|opt| !opt.is_empty());
        options
            .iter_mut()
            .for_each(|opt| opt.make_ascii_uppercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_options_new() {
        let opts = DeviceOptions::new(vec!["qc".to_string()]);
        assert_eq!(&*opts, &["QC"]);
        assert!(opts.contains("QC"));
        assert!(!opts.contains("qc")); // Case-sensitive check
        assert!(!opts.contains("Invalid"));
    }

    #[test]
    fn test_normalize_options() {
        let opts = DeviceOptions::new(vec![
            "hdawg4".to_string(),
            "qc".to_string(),
            "".to_string(),
            "16w".to_string(),
        ]);
        assert_eq!(&*opts, &["HDAWG4", "QC", "16W"]);
    }
}
