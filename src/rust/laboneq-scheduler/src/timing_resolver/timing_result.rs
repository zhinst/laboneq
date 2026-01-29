// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::SectionUid;
use laboneq_units::duration::{Duration, Second};

/// Different types of warnings that can occur during timing calculation.
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord)]
pub(super) enum TimingWarning {
    MatchStartShifted {
        section_uid: SectionUid,
        delay: Duration<Second>,
    },
}

/// Result of the timing calculation, including any warnings encountered.
///
/// The warnings can be optionally deduplicated using `deduplicate_warnings`.
pub(crate) struct TimingResult {
    warnings: Vec<TimingWarning>,
}

impl TimingResult {
    pub(super) fn new() -> Self {
        TimingResult {
            warnings: Vec::new(),
        }
    }

    /// Add a warning to the timing result.
    pub(super) fn add_warning(&mut self, warning: TimingWarning) {
        self.warnings.push(warning);
    }

    /// Check if there are any warnings in the timing result.
    pub(crate) fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Deduplicate warnings in the timing result.
    pub(crate) fn deduplicate_warnings(&mut self) {
        self.warnings.sort();
        self.warnings.dedup();
    }
}

impl Eq for TimingWarning {}

impl std::fmt::Display for TimingWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingWarning::MatchStartShifted { section_uid, delay } => {
                let msg = format!(
                    "Match section '{}' shifted by ({:.2} ns) due to feedback latency constraints.",
                    section_uid.0,
                    delay.value() * 1e9_f64
                );
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::fmt::Display for TimingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut feedback_timing_shifted = vec![];
        for warning in &self.warnings {
            match warning {
                TimingWarning::MatchStartShifted { .. } => {
                    feedback_timing_shifted.push(format!("  - {}", warning));
                }
            }
        }
        if !feedback_timing_shifted.is_empty() {
            let mut msg =
                "Due to feedback latency constraints, the following match sections were shifted:\n"
                    .to_string();
            msg += &feedback_timing_shifted.join("\n");
            write!(f, "{}", msg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use laboneq_common::named_id::NamedId;

    use super::*;

    #[test]
    fn test_timing_result_display() {
        let mut timing_result = TimingResult::new();
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(0).into(),
            delay: 0.1.into(),
        });
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(1).into(),
            delay: 0.01.into(),
        });
        let display = format!("{}", timing_result);
        let expected = "\
Due to feedback latency constraints, the following match sections were shifted:
  - Match section 'NamedId(0)' shifted by (100000000.00 ns) due to feedback latency constraints.
  - Match section 'NamedId(1)' shifted by (10000000.00 ns) due to feedback latency constraints.";
        assert_eq!(display, expected);
    }

    #[test]
    fn test_deduplicate_warnings() {
        let mut timing_result = TimingResult::new();
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(0).into(),
            delay: 0.1.into(),
        });
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(0).into(),
            delay: 0.1.into(),
        });
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(0).into(),
            delay: 0.01.into(),
        });
        timing_result.deduplicate_warnings();
        assert_eq!(timing_result.warnings.len(), 2);
    }

    #[test]
    fn test_has_warnings() {
        let mut timing_result = TimingResult::new();
        assert!(!timing_result.has_warnings());
        timing_result.add_warning(TimingWarning::MatchStartShifted {
            section_uid: NamedId::debug_id(0).into(),
            delay: 0.1.into(),
        });
        assert!(timing_result.has_warnings());
    }
}
