// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_common::types::PhysicalDeviceUid;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{DeviceUid, HandleUid, MatchTarget, SignalUid};

use crate::error::{Error, Result};
use crate::signal_view::SignalView;

/// Resolves [`Operation::Match`] nodes in the experiment tree.
///
/// It ensures that the associated acquisition signals are valid and
/// determines if local feedback is possible.
pub(super) fn resolve_match(
    node: &mut ExperimentNode,
    signals: &HashMap<SignalUid, SignalView>,
) -> Result<()> {
    let mut seen_acquisitions = HashMap::new();
    resolve_match_impl(node, signals, &mut seen_acquisitions)
}

fn resolve_match_impl(
    node: &mut ExperimentNode,
    signals: &HashMap<SignalUid, SignalView>,
    seen_acquisitions: &mut HashMap<HandleUid, SignalUid>,
) -> Result<()> {
    let Operation::Match(obj) = &mut node.kind else {
        for child in node.children.iter_mut() {
            if let Operation::Acquire(acq) = &child.kind {
                seen_acquisitions.insert(acq.handle, acq.signal);
            }
            resolve_match_impl(child, signals, seen_acquisitions)?;
        }
        return Ok(());
    };
    if node.children.is_empty() {
        return Err(Error::new(
            "Match operation must have at least one branch.".to_string(),
        ));
    }
    let MatchTarget::Handle(handle) = obj.target else {
        return Ok(());
    };
    if !seen_acquisitions.contains_key(&handle) {
        return Err(Error::new(format!(
            "No acquisition found for handle '{}' used in match operation.",
            handle.0
        )));
    }
    // Collect all signals used in the match branches to determine devices involved
    let mut contained_signals: HashSet<SignalUid> = HashSet::new();
    for child in node.children.iter() {
        collect_signals_in_node(child, &mut contained_signals);
    }
    let match_devices = collect_devices(contained_signals.iter().map(|s| signals.get(s).unwrap()));
    let feedback_signal = signals
        .get(seen_acquisitions.get(&handle).unwrap())
        .unwrap();
    let local_feedback_allowed = local_feedback_possible(feedback_signal, &match_devices);
    if let Some(local) = obj.local {
        if local && !local_feedback_allowed {
            let msg = format!(
                "Local feedback not possible across devices: '{}' and '{}'.",
                feedback_signal.device_uid().0,
                match_devices
                    .iter()
                    .map(|d| d.uid.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Err(Error::new(msg));
        }
    } else {
        // Set local feedback based on device capabilities if left unspecified
        obj.local = Some(local_feedback_allowed);
    }
    Ok(())
}

fn collect_signals_in_node(node: &ExperimentNode, signals: &mut HashSet<SignalUid>) {
    signals.extend(node.kind.signals());
    for child in &node.children {
        collect_signals_in_node(child, signals);
    }
}

/// A device that is contained in the match operation.
struct MatchDevice {
    uid: DeviceUid,
    physical_uid: PhysicalDeviceUid,
}

/// Whether the match can take feedback without leaving the physical device.
///
/// It can when the acquiring device supports local feedback and every signal the
/// match plays on sits on that same physical device.
fn local_feedback_possible(feedback_signal: &SignalView, match_devices: &[MatchDevice]) -> bool {
    feedback_signal.device_traits().supports_local_feedback
        && match_devices.len() == 1
        && match_devices[0].physical_uid == feedback_signal.physical_device_uid()
}

fn collect_devices<'a>(signals: impl Iterator<Item = &'a SignalView<'a>>) -> Vec<MatchDevice> {
    let mut device_uids = HashSet::new();
    signals
        .filter(|s| device_uids.insert(s.device_uid()))
        .map(|s| MatchDevice {
            uid: s.device_uid(),
            physical_uid: s.physical_device_uid(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use laboneq_common::device_traits::{DeviceTraits, TEST_DEVICE_TRAITS};
    use laboneq_common::named_id::NamedId;
    use laboneq_common::types::{AwgKey, SignalKind};
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::{Acquire, Match, Reserve};
    use laboneq_ir::signal::Signal;
    use laboneq_ir::signal::builder::SignalBuilder;
    use std::collections::HashMap;

    const LOCAL_FEEDBACK_TRAITS: DeviceTraits = DeviceTraits {
        supports_local_feedback: true,
        ..TEST_DEVICE_TRAITS
    };

    struct TestSignalFixture {
        /// On the SHFQA half of an SHFQC.
        signal1: Signal,
        /// On the SHFSG half of the same SHFQC.
        signal2: Signal,
        /// On a separate HDAWG.
        signal3: Signal,
    }

    impl TestSignalFixture {
        fn new() -> Self {
            let signal1 = SignalBuilder::new(
                0.into(),
                2.0e9,
                AwgKey(0),
                0.into(),
                PhysicalDeviceUid(0),
                &LOCAL_FEEDBACK_TRAITS,
                SignalKind::Integration,
            )
            .build();

            let signal2 = SignalBuilder::new(
                1.into(),
                2.0e9,
                AwgKey(1),
                1.into(),
                PhysicalDeviceUid(0),
                &LOCAL_FEEDBACK_TRAITS,
                SignalKind::Iq,
            )
            .build();

            let signal3 = SignalBuilder::new(
                2.into(),
                2.0e9,
                AwgKey(2),
                2.into(),
                PhysicalDeviceUid(1),
                &TEST_DEVICE_TRAITS,
                SignalKind::Iq,
            )
            .build();

            Self {
                signal1,
                signal2,
                signal3,
            }
        }

        fn signal_views(&self) -> HashMap<SignalUid, SignalView<'_>> {
            HashMap::from_iter([
                (self.signal1.uid, SignalView::new(&self.signal1)),
                (self.signal2.uid, SignalView::new(&self.signal2)),
                (self.signal3.uid, SignalView::new(&self.signal3)),
            ])
        }

        fn signal_uids(&self) -> (SignalUid, SignalUid, SignalUid) {
            (self.signal1.uid, self.signal2.uid, self.signal3.uid)
        }
    }

    fn create_feedback_experiment(
        acquire_signal: SignalUid,
        measure_signal: SignalUid,
        handle: HandleUid,
        local: Option<bool>,
    ) -> ExperimentNode {
        node_structure!(
            Operation::Root,
            [
                (
                    Operation::Acquire(Acquire {
                        signal: acquire_signal,
                        handle,
                        length: None,
                        kernel: vec![],
                        parameters: vec![],
                        pulse_parameters: vec![],
                    }),
                    []
                ),
                (
                    Operation::Match(Match {
                        uid: NamedId::debug_id(123).into(),
                        target: MatchTarget::Handle(handle),
                        local,
                        play_after: vec![],
                    }),
                    [(
                        Operation::Reserve(Reserve {
                            signal: measure_signal,
                        }),
                        []
                    ),]
                )
            ]
        )
    }

    /// Test that local feedback is allowed when devices are the same
    /// in the match operation and acquisition.
    #[test]
    fn test_local_feedback_allowed_same_device() {
        let fixture = TestSignalFixture::new();
        let (qa_acquire_uid, qa_measure_uid, _) = fixture.signal_uids();

        // Create experiment tree with match operation
        // Acquire on handle 1 (SHFQC), then match with operations on same device
        let mut node = create_feedback_experiment(qa_acquire_uid, qa_measure_uid, 1.into(), None);
        let result = resolve_match(&mut node, &fixture.signal_views());
        assert!(result.is_ok());

        // Check that local feedback was automatically enabled
        if let Operation::Match(match_op) = &node.children[1].kind {
            assert_eq!(match_op.local, Some(true));
        } else {
            panic!("Expected Match operation");
        }
    }

    /// Test that local feedback is rejected when devices differ
    /// in the match operation and acquisition.
    #[test]
    fn test_local_feedback_not_allowed() {
        let fixture = TestSignalFixture::new();
        let (qa_acquire_uid, _, hdawg_uid) = fixture.signal_uids();

        // Match with local feedback on different devices should error
        let mut root = create_feedback_experiment(
            qa_acquire_uid,
            hdawg_uid,
            1.into(),
            Some(true), // local feedback explicitly enabled, should error
        );

        let result = resolve_match(&mut root, &fixture.signal_views());

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Local feedback not possible across devices"));
    }

    /// Test that local feedback is automatically disabled when not possible
    #[test]
    fn test_not_local_feedback_resolved() {
        let fixture = TestSignalFixture::new();
        let (qa_acquire_uid, _, hdawg_uid) = fixture.signal_uids();

        let mut root = create_feedback_experiment(
            qa_acquire_uid,
            hdawg_uid,
            1.into(),
            None, // local feedback unspecified, should be resolved to false
        );

        resolve_match(&mut root, &fixture.signal_views()).unwrap();

        // Check that local feedback is disabled
        if let Operation::Match(match_op) = &root.children[1].kind {
            assert_eq!(match_op.local, Some(false));
        } else {
            panic!("Expected Match operation");
        }
    }
}
