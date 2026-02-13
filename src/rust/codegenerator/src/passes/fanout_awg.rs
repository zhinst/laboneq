// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use crate::ir::builders::delay;
use crate::ir::compilation_job::{AwgCore, DeviceKind, Signal};
use crate::ir::{
    Case, InitialOscillatorFrequency, IrNode, NodeKind, PhaseReset, Samples,
    SetOscillatorFrequency, SignalUid, TriggerBitData,
};

struct Context<'a> {
    signals: HashMap<SignalUid, &'a Arc<Signal>>,
    inline_sections: bool,
}

impl<'a> Context<'a> {
    fn get_signal(&self, uid: SignalUid) -> &'a Arc<Signal> {
        self.signals
            .get(&uid)
            .expect("Signal must exist in context")
    }
}

fn contains_signal(ctx: &Context, signal_uid: SignalUid) -> bool {
    ctx.signals.contains_key(&signal_uid)
}

fn filter_signals(awg: &Context, signals: &[Arc<Signal>]) -> Vec<Arc<Signal>> {
    signals
        .iter()
        .filter_map(|s| contains_signal(awg, s.uid).then_some(Arc::clone(s)))
        .collect()
}

fn sort_nodes(nodes: &mut [IrNode]) {
    nodes.sort_by_key(|n| *n.offset());
}

/// Recursively gets the first signal uid found in the node or its children.
fn get_first_signal(node: &IrNode) -> Option<SignalUid> {
    if let Some(sig) = node.data().signals().first() {
        return Some(*sig);
    }
    for child in node.iter_children() {
        if let Some(sig) = get_first_signal(child) {
            return Some(sig);
        }
    }
    None
}

/// Builds the AWG IR for the given node.
///
/// The function traverses the node and its children, filtering out nodes that are not relevant to the AWG
/// and collecting the relevant nodes into `nodes`.
///
/// The filtering logic:
///
/// - Leaf nodes: Node properties are checked for the relevancy for the AWG.
/// - Nodes with children: The node is relevant only if at least one of its children is relevant.
fn build_awg_ir(node: &IrNode, parent_offset: Samples, ctx: &Context<'_>, nodes: &mut Vec<IrNode>) {
    match node.data() {
        NodeKind::PlayPulse(ob) => {
            if !contains_signal(ctx, ob.signal.uid) {
                return;
            }
            let new_node = IrNode::new(
                NodeKind::PlayPulse(ob.clone()),
                *node.offset() + parent_offset,
            );
            nodes.push(new_node);
        }
        NodeKind::AcquirePulse(ob) => {
            if !contains_signal(ctx, ob.signal.uid) {
                return;
            }
            let kind = NodeKind::AcquirePulse(ob.clone());
            let new_node = IrNode::new(kind, *node.offset() + parent_offset);
            nodes.push(new_node);
        }
        NodeKind::InitialOscillatorFrequency(ob) => {
            let mut out = vec![];
            for freq in ob.iter() {
                if !contains_signal(ctx, freq.signal.uid) {
                    continue;
                } else {
                    out.push(freq.clone());
                }
            }
            if out.is_empty() {
                return;
            }
            let data = NodeKind::InitialOscillatorFrequency(InitialOscillatorFrequency::new(out));
            let new_node = IrNode::new(data, *node.offset() + parent_offset);
            nodes.push(new_node);
        }
        NodeKind::SetOscillatorFrequency(ob) => {
            let mut out = vec![];
            for freq in ob.iter() {
                if !contains_signal(ctx, freq.signal.uid) {
                    continue;
                } else {
                    out.push(freq.clone());
                }
            }
            if out.is_empty() {
                return;
            }
            let data = NodeKind::SetOscillatorFrequency(SetOscillatorFrequency::new(out));
            let new_node = IrNode::new(data, *node.offset() + parent_offset);
            nodes.push(new_node);
        }
        NodeKind::Match(obj) => {
            // For Match nodes, we need to process branches first to determine
            // if any of them are relevant to the AWG.
            //
            // If any of the branches are relevant, we need to ensure that
            // all branches have a length equal to the Match node length and the
            // zero-length and empty branches are filled with placeholder pulses to ensure
            // correct timing.
            let length = obj.length;
            let mut children = vec![];
            for child in node.iter_children() {
                build_awg_ir(child, 0, ctx, &mut children);
            }

            if children.iter().all(|child| !child.has_children()) {
                // If all children are empty, we can skip the entire Match block
                return;
            }

            // We will get the first signal used in the non-empty children.
            // In the case of multiple signals, it does not matter which one we pick,
            // as the purpose is only to create a placeholder pulse.
            let used_signal = ctx.get_signal(
                children
                    .iter()
                    .find_map(get_first_signal)
                    .expect("Internal error: No relevant signals found for match branches."),
            );

            for child in children.iter_mut() {
                // For zero-length cases, we need to create a waveform placeholder for
                // the length of the parent match-block.
                // The case-block may be empty (no children) or contain only nodes which
                // have no length (e.g. phase increments).
                if !child.has_children() || child.data().length() == 0 {
                    let delay = delay(Arc::clone(used_signal), length);
                    child.add_child(0, NodeKind::PlayPulse(delay));
                }
            }

            let mut new_node = IrNode::new(node.data().clone(), *node.offset() + parent_offset);
            new_node.add_child_nodes(children);
            nodes.push(new_node);
        }
        NodeKind::Case(ob) => {
            let ob = Case {
                length: ob.length,
                state: ob.state,
                section_info: Arc::clone(&ob.section_info),
            };
            let mut new_node = IrNode::new(NodeKind::Case(ob), *node.offset() + parent_offset);
            let mut children = vec![];
            for child in node.iter_children() {
                build_awg_ir(child, *node.offset() + parent_offset, ctx, &mut children);
            }
            new_node.add_child_nodes(children);
            nodes.push(new_node);
        }
        NodeKind::PpcSweepStep(ob) => {
            if !contains_signal(ctx, ob.signal.uid) {
                return;
            }
            let new_node = IrNode::new(
                NodeKind::PpcSweepStep(ob.clone()),
                *node.offset() + parent_offset,
            );
            nodes.push(new_node);
        }
        NodeKind::PhaseReset(ob) => {
            let signals = filter_signals(ctx, &ob.signals);
            if signals.is_empty() {
                return;
            }
            let ob = PhaseReset { signals };
            let new_node = IrNode::new(NodeKind::PhaseReset(ob), *node.offset() + parent_offset);
            nodes.push(new_node);
        }
        NodeKind::PrecompensationFilterReset { signal } => {
            if !contains_signal(ctx, signal.uid) {
                return;
            }
            let new_node = IrNode::new(
                NodeKind::PrecompensationFilterReset {
                    signal: Arc::clone(signal),
                },
                *node.offset() + parent_offset,
            );
            nodes.push(new_node);
        }
        NodeKind::Section(ob) => {
            // When inlining sections, we must extend the children offsets
            // with previous parent offsets
            let children_offset = if ctx.inline_sections {
                node.offset() + parent_offset
            } else {
                0
            };
            let mut children = vec![];
            for child in node.iter_children() {
                build_awg_ir(child, children_offset, ctx, &mut children);
            }
            if children.is_empty() && ob.trigger_output.is_empty() {
                // If there are no children and no trigger output, we can skip this section
                return;
            }
            let mut start_nodes = Vec::with_capacity(1 + ob.trigger_output.len());
            let mut end_nodes = Vec::with_capacity(1 + ob.trigger_output.len());
            // Setup PRNG only if Section contains AWG related nodes
            if let Some(prng_setup) = &ob.prng_setup
                && !children.is_empty()
            {
                let prng_setup = NodeKind::SetupPrng(prng_setup.clone());
                let prng_setup_node = IrNode::new(prng_setup, *node.offset() + parent_offset);
                start_nodes.push(prng_setup_node);
                let prng_drop = IrNode::new(
                    NodeKind::DropPrngSetup,
                    *node.offset() + node.data().length() + parent_offset,
                );
                end_nodes.push(prng_drop);
            }
            // Trigger is always set whether or not the Section contains AWG related nodes.
            for (signal, bits) in &ob.trigger_output {
                if !contains_signal(ctx, signal.uid) {
                    continue;
                }
                let set_data = TriggerBitData {
                    signal: Arc::clone(signal),
                    bits: *bits,
                    set: true,
                    section_info: Arc::clone(&ob.section_info),
                };
                let new_node = IrNode::new(
                    NodeKind::TriggerSet(set_data),
                    *node.offset() + parent_offset,
                );
                start_nodes.push(new_node);
                let unset_data = TriggerBitData {
                    signal: Arc::clone(signal),
                    bits: *bits,
                    set: false,
                    section_info: Arc::clone(&ob.section_info),
                };
                let new_node = IrNode::new(
                    NodeKind::TriggerSet(unset_data),
                    *node.offset() + node.data().length() + parent_offset,
                );
                end_nodes.push(new_node);
            }
            nodes.extend(start_nodes);
            if ctx.inline_sections {
                // If we are inlining sections, we do not add the section node itself
                nodes.extend(children);
            } else {
                let mut new_node = IrNode::new(node.data().clone(), *node.offset() + parent_offset);
                new_node.add_child_nodes(children);
                nodes.push(new_node);
            }
            nodes.extend(end_nodes);
        }
        _ => {
            if node.has_children() {
                let mut children = vec![];
                for child in node.iter_children() {
                    build_awg_ir(child, 0, ctx, &mut children);
                }
                if !children.is_empty() {
                    sort_nodes(&mut children);
                    let mut new_node =
                        IrNode::new(node.data().clone(), *node.offset() + parent_offset);
                    new_node.add_child_nodes(children);
                    nodes.push(new_node);
                }
            }
        }
    }
}

/// Generates a fanout for the given AWG.
///
/// The function collects all nodes that are relevant for the AWG.
/// The function also inlines all the [`NodeKind::Section`] nodes.
/// The pass expects the node timing to be relative to the parent node.
///
/// # Returns
///
/// A new [`IrNode`] that contains the filtered nodes for the AWG.
/// If no relevant nodes are found, a Nop node is returned with the length of the original node.
pub(crate) fn fanout_for_awg(node: &IrNode, awg: &AwgCore) -> IrNode {
    let signals: HashMap<SignalUid, &Arc<Signal>> =
        awg.signals.iter().map(|s| (s.uid, s)).collect();
    let ctx = Context {
        signals,
        inline_sections: !matches!(awg.device_kind(), &DeviceKind::SHFQA | &DeviceKind::UHFQA),
    };
    let mut children = vec![];
    build_awg_ir(node, 0, &ctx, &mut children);
    children.pop().unwrap_or(IrNode::new(
        NodeKind::Nop {
            length: node.data().length(),
        },
        0,
    ))
}

#[cfg(test)]
mod tests {
    use laboneq_common::named_id::NamedId;

    use super::*;
    use crate::ir::compilation_job::{AwgKind, Device, DeviceKind, Signal, SignalKind};
    use crate::ir::{IrNode, Loop, NodeKind, Section, SectionInfo};

    struct IrBuilder {
        node_stack: Vec<IrNode>,
    }

    impl IrBuilder {
        fn new() -> Self {
            Self {
                node_stack: vec![IrNode::new(NodeKind::Nop { length: 0 }, 0)],
            }
        }

        pub(crate) fn with<F>(&mut self, f: F)
        where
            F: FnOnce(&mut Self),
        {
            f(self);
        }

        fn enter_stack<F>(&mut self, node: IrNode, f: F)
        where
            F: FnOnce(&mut Self),
        {
            self.node_stack.push(node);
            f(self);
            let parent = self.node_stack.pop();
            if let Some(parent) = parent {
                self.node_stack.last_mut().unwrap().add_child_node(parent);
            }
        }

        pub(crate) fn section<F>(&mut self, uid: &str, length: Samples, offset: Samples, f: F)
        where
            F: FnOnce(&mut Self),
        {
            let section = NodeKind::Section(Section {
                length,
                trigger_output: vec![],
                prng_setup: None,
                section_info: Arc::new(SectionInfo {
                    name: uid.to_string(),
                    id: 0,
                }),
            });
            self.enter_stack(IrNode::new(section, offset), f);
        }

        pub(crate) fn sweep<F>(&mut self, length: Samples, f: F)
        where
            F: FnOnce(&mut Self),
        {
            let root = NodeKind::Loop(Loop {
                length,
                compressed: false,
                section_info: Arc::new(SectionInfo {
                    name: "".to_string(),
                    id: 0,
                }),
                count: 1,
                prng_sample: None,
                parameters: vec![],
            });
            self.enter_stack(IrNode::new(root, 0), f);
        }

        pub(crate) fn reset_precompensation(&mut self, offset: Samples, signal: Arc<Signal>) {
            let node = NodeKind::PrecompensationFilterReset { signal };
            let ir_node = IrNode::new(node, offset);
            self.node_stack.last_mut().unwrap().add_child_node(ir_node);
        }

        fn build(&mut self) -> IrNode {
            self.node_stack.pop().unwrap()
        }
    }

    fn create_awg_core(signals: Vec<Signal>, device_kind: DeviceKind) -> AwgCore {
        AwgCore::new(
            0,
            AwgKind::IQ,
            signals.iter().map(|s| Arc::new(s.clone())).collect(),
            2e9,
            Arc::new(Device::new("test_device".to_string().into(), device_kind)),
            HashMap::new(),
            None,
            false,
        )
    }

    fn create_signal(uid: u32) -> Signal {
        Signal {
            uid: NamedId::debug_id(uid).into(),
            kind: SignalKind::IQ,
            signal_delay: 0,
            start_delay: 0,
            channels: vec![],
            oscillator: None,
            automute: false,
        }
    }

    /// Test fanout for awg for Section inlining
    ///
    /// Test that fanout works correctly for Sections and that the
    /// Sections are inlined correctly.
    ///
    /// The test checks that the offsets are aggregated correctly
    /// and that the inlined node children are sorted correctly.
    #[test]
    fn test_fanout_for_awg_section_inline() {
        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.section("s0", 0, 8, |b| {
                // 2 parallel sections where the first reset happens after second Section reset.
                b.section("s1", 0, 16, |b| {
                    b.reset_precompensation(32, Arc::new(create_signal(0)));
                });
                b.section("s2", 0, 16, |b| {
                    b.reset_precompensation(16, Arc::new(create_signal(0)));
                });
                b.reset_precompensation(0, Arc::new(create_signal(1)));
            });
        });

        let fanout = fanout_for_awg(
            &builder.build(),
            &create_awg_core(vec![create_signal(0)], DeviceKind::SHFSG),
        );

        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.reset_precompensation(8 + 16 + 16, Arc::new(create_signal(0)));
            b.reset_precompensation(8 + 16 + 32, Arc::new(create_signal(0)));
        });
        assert_eq!(builder.build(), fanout);
    }

    /// Test fanout for awg for parent capture.
    ///
    /// Test that fanout works correctly for internal nodes where
    /// a child belongs to the AWG.
    #[test]
    fn test_fanout_for_awg_parent_capture() {
        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.sweep(5, |b| {
                b.section("s0", 0, 0, |b| {
                    b.reset_precompensation(5, Arc::new(create_signal(0)));
                    b.reset_precompensation(0, Arc::new(create_signal(1)));
                });
            });
        });

        let fanout = fanout_for_awg(
            &builder.build(),
            &create_awg_core(vec![create_signal(0)], DeviceKind::SHFSG),
        );

        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.sweep(5, |b| {
                b.reset_precompensation(5, Arc::new(create_signal(0)));
            });
        });
        assert_eq!(builder.build(), fanout);
    }

    /// Test fanout for device with measurement support.
    ///
    /// Test that Sections are not inlined for SHFQA.
    #[test]
    fn test_fanout_for_awg_qa_skip_inline() {
        let mut builder = IrBuilder::new();
        builder.with(|b| {
            b.sweep(5, |b| {
                b.section("s0", 0, 0, |b| {
                    b.reset_precompensation(5, Arc::new(create_signal(0)));
                });
            });
        });
        let ir = builder.build();
        let fanout = fanout_for_awg(
            &ir,
            &create_awg_core(vec![create_signal(0)], DeviceKind::SHFQA),
        );
        assert_eq!(ir, fanout);
    }
}
