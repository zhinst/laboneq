// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use laboneq_common::types::SignalKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{OscillatorKind, SignalUid};
use laboneq_ir::signal::Signal;

/// Resolves phase resets without assigned signals by assigning them
/// to the signals present in their scope: Signals in the same section.
///
/// Errors if a phase reset is assigned to an RF signal with hardware oscillator.
pub(super) fn resolve_phase_reset(
    node: &mut ExperimentNode,
    signals: &HashMap<SignalUid, &Signal>,
) -> Result<()> {
    let mut has_unassigned_reset = false;
    for child in &mut node.children {
        match &child.kind {
            Operation::ResetOscillatorPhase(obj) => {
                has_unassigned_reset |= obj.signals.is_empty();
                for signal in &obj.signals {
                    let signal = signals
                        .get(signal)
                        .expect("Internal Error: Expected signal to be present");
                    if let Some(oscillator) = &signal.oscillator
                        && oscillator.kind == OscillatorKind::Hardware
                        && signal.kind == SignalKind::Rf
                    {
                        return Err(Error::new(format!(
                            "Phase reset on hardware modulated RF signal '{}' is not supported.",
                            signal.uid.0
                        )));
                    }
                }
            }
            _ => resolve_phase_reset(child, signals)?,
        }
    }
    if has_unassigned_reset {
        assign_scope_signals(node)?;
    }
    Ok(())
}

/// Assigns the signals in scope of `node`'s children to its phase resets that have none.
fn assign_scope_signals(node: &mut ExperimentNode) -> Result<()> {
    let mut scope_signals = HashSet::new();
    for child in &node.children {
        collect_signals(child, &mut scope_signals);
    }
    if scope_signals.is_empty() {
        return Err(Error::new(
            "No signals available to assign for oscillator phase reset.",
        ));
    }
    let mut scope_signals: Vec<_> = scope_signals.into_iter().collect();
    scope_signals.sort();
    for child in &mut node.children {
        if let Operation::ResetOscillatorPhase(obj) = &mut child.kind
            && obj.signals.is_empty()
        {
            obj.signals = scope_signals.clone();
        }
    }
    Ok(())
}

/// Collects the signals of `node` and its descendants, excluding phase resets.
fn collect_signals(node: &ExperimentNode, signals: &mut HashSet<SignalUid>) {
    if matches!(node.kind, Operation::ResetOscillatorPhase(_)) {
        return;
    }
    signals.extend(node.kind.signals());
    for child in &node.children {
        collect_signals(child, signals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laboneq_common::device_traits::TEST_DEVICE_TRAITS;
    use laboneq_common::named_id::NamedId;
    use laboneq_common::types::{AwgKey, PhysicalDeviceUid};
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::{Reserve, ResetOscillatorPhase};
    use laboneq_dsl::types::Oscillator;
    use laboneq_ir::signal::builder::SignalBuilder;

    fn make_iq_signal(uid: u32) -> Signal {
        SignalBuilder::new(
            NamedId::debug_id(uid).into(),
            2e9,
            AwgKey(0),
            NamedId::debug_id(0).into(),
            PhysicalDeviceUid(0),
            &TEST_DEVICE_TRAITS,
            SignalKind::Iq,
        )
        .build()
    }

    #[test]
    fn test_resolve_phase_reset_fill_scoped_signals() {
        let signals = [make_iq_signal(0), make_iq_signal(1)];
        let mut experiment = node_structure!(
            Operation::RealTimeBoundary,
            [
                (Operation::Reserve(Reserve::new(signals[0].uid)), []),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::default()),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[1].uid)), []),
            ]
        );
        resolve_phase_reset(
            &mut experiment,
            &signals.iter().map(|s| (s.uid, s)).collect(),
        )
        .unwrap();
        let experiment_expected = node_structure!(
            Operation::RealTimeBoundary,
            [
                (Operation::Reserve(Reserve::new(signals[0].uid)), []),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[0].uid,
                        signals[1].uid
                    ])),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[1].uid)), []),
            ]
        );
        assert_eq!(experiment, experiment_expected);
    }

    #[test]
    fn test_resolve_phase_reset_fill_nested_scopes() {
        let signals = [make_iq_signal(0), make_iq_signal(1)];
        let mut experiment = node_structure!(
            Operation::Root,
            [
                (
                    Operation::RealTimeBoundary,
                    [
                        (Operation::Reserve(Reserve::new(signals[1].uid)), []),
                        (
                            Operation::ResetOscillatorPhase(ResetOscillatorPhase::default()),
                            []
                        ),
                    ]
                ),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::default()),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[0].uid)), []),
            ]
        );
        resolve_phase_reset(
            &mut experiment,
            &signals.iter().map(|s| (s.uid, s)).collect(),
        )
        .unwrap();
        let experiment_expected = node_structure!(
            Operation::Root,
            [
                (
                    Operation::RealTimeBoundary,
                    [
                        (Operation::Reserve(Reserve::new(signals[1].uid)), []),
                        (
                            Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                                signals[1].uid
                            ])),
                            []
                        ),
                    ]
                ),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[0].uid,
                        signals[1].uid
                    ])),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[0].uid)), []),
            ]
        );
        assert_eq!(experiment, experiment_expected);
    }

    #[test]
    fn test_resolve_phase_reset_fill_ignores_signals_of_other_resets() {
        let signals = [make_iq_signal(0), make_iq_signal(1)];
        let mut experiment = node_structure!(
            Operation::RealTimeBoundary,
            [
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[0].uid
                    ])),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[1].uid)), []),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::default()),
                    []
                ),
            ]
        );
        resolve_phase_reset(
            &mut experiment,
            &signals.iter().map(|s| (s.uid, s)).collect(),
        )
        .unwrap();
        let experiment_expected = node_structure!(
            Operation::RealTimeBoundary,
            [
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[0].uid
                    ])),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[1].uid)), []),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[1].uid
                    ])),
                    []
                ),
            ]
        );
        assert_eq!(experiment, experiment_expected);
    }

    #[test]
    fn test_resolve_phase_reset_no_signals_in_scope() {
        let mut experiment = node_structure!(
            Operation::RealTimeBoundary,
            [(
                Operation::ResetOscillatorPhase(ResetOscillatorPhase::default()),
                []
            ),]
        );
        let result = resolve_phase_reset(&mut experiment, &HashMap::new());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No signals available to assign for oscillator phase reset.")
        );
    }

    #[test]
    fn test_resolve_phase_reset_signals_defined() {
        let signals = [make_iq_signal(0), make_iq_signal(1)];
        let mut experiment = node_structure!(
            Operation::RealTimeBoundary,
            [
                (Operation::Reserve(Reserve::new(signals[0].uid)), []),
                (
                    Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                        signals[1].uid
                    ])),
                    []
                ),
                (Operation::Reserve(Reserve::new(signals[1].uid)), []),
            ]
        );
        let experiment_expected = experiment.clone();
        resolve_phase_reset(
            &mut experiment,
            &signals.iter().map(|s| (s.uid, s)).collect(),
        )
        .unwrap();
        assert_eq!(experiment, experiment_expected);
    }

    #[test]
    fn test_resolve_phase_reset_rf_signal() {
        let signal = SignalBuilder::new(
            NamedId::debug_id(0).into(),
            2e9,
            AwgKey(0),
            NamedId::debug_id(0).into(),
            PhysicalDeviceUid(0),
            &TEST_DEVICE_TRAITS,
            SignalKind::Rf,
        )
        .oscillator(Oscillator {
            uid: NamedId::debug_id(0).into(),
            frequency: 5e9.into(),
            kind: OscillatorKind::Hardware,
        })
        .build();

        let mut experiment = node_structure!(
            Operation::RealTimeBoundary,
            [(
                Operation::ResetOscillatorPhase(ResetOscillatorPhase::new(vec![
                    NamedId::debug_id(0).into()
                ])),
                []
            ),]
        );
        let result = resolve_phase_reset(
            &mut experiment,
            &[signal].iter().map(|s| (s.uid, s)).collect(),
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Phase reset on hardware modulated RF signal")
        );
    }
}
