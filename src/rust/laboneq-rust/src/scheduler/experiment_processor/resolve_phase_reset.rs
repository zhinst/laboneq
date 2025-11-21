// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::scheduler::experiment::{Signal, SignalKind};
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{Operation, OscillatorKind, SignalUid};

/// Resolves phase resets without assigned signals by assigning them
/// to the signals present in their scope: Signals in the same section.
///
/// Errors if a phase reset is assigned to an RF signal with hardware oscillator.
pub fn resolve_phase_reset(
    node: &mut ExperimentNode,
    signals: &HashMap<SignalUid, Signal>,
) -> Result<()> {
    let mut seen_signals = HashSet::new();
    resolve_phase_reset_impl(node, &mut seen_signals, signals)
}

fn resolve_phase_reset_impl(
    node: &mut ExperimentNode,
    seen_signals: &mut HashSet<SignalUid>,
    signals: &HashMap<SignalUid, Signal>,
) -> Result<()> {
    let mut to_fill = vec![];
    let mut this_signals = HashSet::new();
    for (idx, child) in node.children.iter_mut().enumerate() {
        match &child.kind {
            Operation::ResetOscillatorPhase(obj) => {
                if obj.signals.is_empty() {
                    to_fill.push(idx);
                }
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
            _ => {
                this_signals.extend(child.kind.signals().iter().cloned());
                resolve_phase_reset_impl(child.make_mut(), &mut this_signals, signals)?;
            }
        }
    }
    if !to_fill.is_empty() {
        if this_signals.is_empty() {
            return Err(Error::new(
                "No signals available to assign for oscillator phase reset.",
            ));
        }
        for obj in to_fill {
            let node = node.children[obj].make_mut();
            if let Operation::ResetOscillatorPhase(obj) = &mut node.kind {
                obj.signals = this_signals.iter().cloned().collect();
                obj.signals.sort();
            }
        }
    }
    seen_signals.extend(this_signals);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::scheduler::experiment::builders::SignalBuilder;

    use super::*;
    use laboneq_common::named_id::NamedId;
    use laboneq_common::types::{AwgKey, DeviceKind};
    use laboneq_scheduler::experiment::types::{
        Oscillator, OscillatorKind, Reserve, ResetOscillatorPhase,
    };
    use laboneq_scheduler::node_structure;

    fn make_iq_signal(uid: u32) -> Signal {
        SignalBuilder::new(
            NamedId::debug_id(uid).into(),
            2e9,
            AwgKey(0),
            DeviceKind::Hdawg,
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
            &signals.iter().map(|s| (s.uid, s.clone())).collect(),
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
            &signals.iter().map(|s| (s.uid, s.clone())).collect(),
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
            DeviceKind::Hdawg,
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
        let result = resolve_phase_reset(&mut experiment, &HashMap::from([(signal.uid, signal)]));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Phase reset on hardware modulated RF signal")
        );
    }
}
