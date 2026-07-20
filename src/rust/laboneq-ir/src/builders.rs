// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{
    ComplexOrFloat, HandleUid, MatchTarget, PulseUid, SectionUid, SignalUid, ValueOrParameter,
};
use laboneq_units::tinysample::{TinySamples, tiny_samples};

use crate::ir::{Acquire, Case, IrKind, Match, PlayPulse, PrngSetup, Section, Trigger};
use crate::node::IrNode;

pub struct SectionBuilder {
    inner: Section,
}

impl SectionBuilder {
    pub fn new(uid: SectionUid) -> Self {
        Self {
            inner: Section {
                uid,
                triggers: Vec::new(),
                prng_setup: None,
            },
        }
    }

    pub fn add_trigger(mut self, signal: SignalUid, state: u8) -> Self {
        self.inner.triggers.push(Trigger { signal, state });
        self
    }

    pub fn prng_setup(mut self, range: u32, seed: u32) -> Self {
        assert!(
            self.inner.prng_setup.is_none(),
            "PRNG setup already defined for this section"
        );
        self.inner.prng_setup = Some(PrngSetup { range, seed });
        self
    }

    pub fn build(self) -> Section {
        self.inner
    }
}

/// Builder for constructing [`IrNode`] trees in tests.
///
/// Container nodes nest via closures; leaves are plain calls. Children are
/// placed at offset 0 unless preceded by [`Self::at_offset`].
pub struct IrNodeBuilder {
    node_stack: Vec<(TinySamples, IrNode)>,

    /// Offset of the next node we create.
    open_offset: TinySamples,
}

impl Default for IrNodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrNodeBuilder {
    pub fn new() -> IrNodeBuilder {
        IrNodeBuilder {
            node_stack: vec![],
            open_offset: tiny_samples(0),
        }
    }

    pub fn add_node_with<F>(&mut self, node: IrNode, f: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.node_stack.push((self.open_offset, node));
        self.open_offset = tiny_samples(0);
        f(self);
        if let Some((offset, node)) = self.node_stack.pop() {
            match self.node_stack.last_mut() {
                None => self.node_stack.push((offset, node)),
                Some((_, parent)) => parent.add_child(offset, node),
            }
        }
        self
    }

    pub fn add_node(&mut self, node: IrNode) -> &mut Self {
        self.add_node_with(node, |_| ())
    }

    pub fn at_offset(&mut self, offset: TinySamples) -> &mut Self {
        self.open_offset = offset;
        self
    }

    pub fn root<F>(&mut self, length: TinySamples, f: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_node_with(IrNode::new(IrKind::Root, length), f)
    }

    pub fn section<F>(&mut self, uid: SectionUid, length: TinySamples, f: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_node_with(
            IrNode::new(IrKind::Section(SectionBuilder::new(uid).build()), length),
            f,
        )
    }

    pub fn match_node<F>(
        &mut self,
        uid: SectionUid,
        target: MatchTarget,
        length: TinySamples,
        f: F,
    ) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_node_with(
            IrNode::new(
                IrKind::Match(Match {
                    uid,
                    target,
                    local: false,
                }),
                length,
            ),
            f,
        )
    }

    pub fn case<F>(&mut self, uid: SectionUid, state: usize, length: TinySamples, f: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_node_with(IrNode::new(IrKind::Case(Case { uid, state }), length), f)
    }

    pub fn acquire(
        &mut self,
        signal: SignalUid,
        handle: HandleUid,
        kernels: &[PulseUid],
        length: TinySamples,
    ) -> &mut Self {
        self.add_node(IrNode::new(
            IrKind::Acquire(Acquire {
                signal,
                handle,
                integration_length: length,
                kernels: kernels.to_vec(),
                parameters: vec![HashMap::new(); kernels.len()],
                pulse_parameters: vec![HashMap::new(); kernels.len()],
            }),
            length,
        ))
    }

    /// Constant-amplitude pulse play with no phase or pulse parameters.
    pub fn play(
        &mut self,
        signal: SignalUid,
        pulse: PulseUid,
        amplitude: f64,
        length: TinySamples,
    ) -> &mut Self {
        self.add_node(IrNode::new(
            IrKind::PlayPulse(PlayPulse {
                signal,
                pulse,
                amplitude: ValueOrParameter::Value(ComplexOrFloat::Float(amplitude)),
                phase: None,
                increment_oscillator_phase: None,
                set_oscillator_phase: None,
                parameters: HashMap::new(),
                pulse_parameters: HashMap::new(),
                markers: vec![],
            }),
            length,
        ))
    }

    pub fn build(&mut self) -> IrNode {
        self.node_stack.pop().expect("builder is empty").1
    }
}
