// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Check that coprocessor values are live before use.

use std::collections::HashMap;

use laboneq_common::uid::{PulseUid, SectionUid, SignalUid, VariableUid};
use laboneq_dsl::hqcs::{FieldBinding, HqcsDeclarations, IsLiveTarget, Predicate};
use laboneq_dsl::operation::MarkStaleTarget;
use laboneq_error::{LabOneQError, laboneq_error};

use laboneq_ir::node::IrNode;
use laboneq_ir::{IrKind, MatchTarget};

/// Checks that variables and updated pulses are live on every path that uses them.
///
/// The check follows program order, regardless of scheduled times. Each
/// `mark_stale` starts a new update; `expect(is_live(...))` allows subsequent
/// operations to use its value. A target must be live before another `mark_stale`.
///
/// Variables with initial values and all pulses start live. Other variables have
/// neither a value nor a pending update until their first `mark_stale`.
/// Every loop is assumed to run at least once, and every match to select a case.
pub fn check_liveness(root: &IrNode, hqcs: &HqcsDeclarations) -> Result<(), LabOneQError> {
    let checker = Checker::new(hqcs);
    if checker.initial.is_empty() {
        return Ok(());
    }
    let mut env = checker.initial.clone();
    checker.visit_children(root, &mut env)
}

/// Possible states of a target across execution paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct States(u8);

impl States {
    const LIVE: Self = Self(1);
    /// An update is pending.
    const STALE: Self = Self(2);
    /// Never initialized and not yet marked stale.
    const UNSET: Self = Self(4);

    fn may_be(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// State of each tracked target: variables first, then pulses.
type TargetStates = Vec<States>;

fn join(a: &TargetStates, b: &TargetStates) -> TargetStates {
    a.iter().zip(b).map(|(a, b)| a.union(*b)).collect()
}

#[derive(Clone, Copy)]
enum Target {
    Variable(VariableUid),
    Pulse(PulseUid),
}

impl From<MarkStaleTarget> for Target {
    fn from(target: MarkStaleTarget) -> Self {
        match target {
            MarkStaleTarget::Variable(v) => Self::Variable(v),
            MarkStaleTarget::Pulse(p) => Self::Pulse(p),
        }
    }
}

impl From<IsLiveTarget> for Target {
    fn from(target: IsLiveTarget) -> Self {
        match target {
            IsLiveTarget::Variable(v) => Self::Variable(v),
            IsLiveTarget::Pulse(p) => Self::Pulse(p),
        }
    }
}

struct Checker<'a> {
    hqcs: &'a HqcsDeclarations,
    /// Index in [`TargetStates`] for each pulse updated by a stream.
    pulses: HashMap<PulseUid, usize>,
    initial: TargetStates,
}

impl<'a> Checker<'a> {
    fn new(hqcs: &'a HqcsDeclarations) -> Self {
        let mut pulses = HashMap::new();
        for field in hqcs.streams.iter().flat_map(|stream| &stream.fields) {
            if let FieldBinding::Pulse(pulse) = field.binding {
                let next = hqcs.variables.len() + pulses.len();
                pulses.entry(pulse).or_insert(next);
            }
        }
        let initial = hqcs
            .variables
            .iter()
            .map(|v| match v.initial {
                Some(_) => States::LIVE,
                None => States::UNSET,
            })
            .chain(std::iter::repeat_n(States::LIVE, pulses.len()))
            .collect();
        Self {
            hqcs,
            pulses,
            initial,
        }
    }

    fn visit_children(&self, node: &IrNode, env: &mut TargetStates) -> Result<(), LabOneQError> {
        for child in &node.children {
            self.visit(&child.node, env)?;
        }
        Ok(())
    }

    fn visit(&self, node: &IrNode, env: &mut TargetStates) -> Result<(), LabOneQError> {
        match &node.kind {
            IrKind::Match(m) => {
                if let MatchTarget::Variable(variable) = m.target {
                    self.require_live(env, Target::Variable(variable), Usage::Match(m.uid))?;
                }
                self.visit_cases(node, env)
            }
            IrKind::Loop(obj) if obj.iterations.get() as usize != node.children.len() => {
                self.visit_repeated_loop(node, env)
            }
            IrKind::MarkStale(target) => {
                let target = (*target).into();
                if let Some(i) = self.index(target) {
                    if env[i].may_be(States::STALE) {
                        return Err(laboneq_error!(
                            "{} may already have a pending update at mark_stale(...). Call \
                             expect(is_live(...)) before marking it stale again.",
                            self.describe(target)
                        ));
                    }
                    env[i] = States::STALE;
                }
                Ok(())
            }
            IrKind::Expect(Predicate::IsLive(target)) => {
                let target = (*target).into();
                if let Some(i) = self.index(target) {
                    if env[i].may_be(States::UNSET) {
                        return Err(laboneq_error!(
                            "{} has no initial value and may not have been marked stale before \
                             expect(is_live(...)). Call mark_stale(...) first to start an update.",
                            self.describe(target)
                        ));
                    }
                    env[i] = States::LIVE;
                }
                Ok(())
            }
            IrKind::Expect(Predicate::Comparison { variable, .. }) => {
                self.require_live(env, Target::Variable(*variable), Usage::Comparison)
            }
            IrKind::PlayPulse(play) => {
                self.require_live(env, Target::Pulse(play.pulse), Usage::Play(play.signal))
            }
            _ => self.visit_children(node, env),
        }
    }

    /// Checks a loop whose body has not been unrolled.
    ///
    /// Checks the first iteration, then combines the entry and exit states to
    /// check whether later iterations can use stale or uninitialized values.
    fn visit_repeated_loop(
        &self,
        node: &IrNode,
        env: &mut TargetStates,
    ) -> Result<(), LabOneQError> {
        let mut exit = env.clone();
        self.visit_children(node, &mut exit)?;

        // Include states from before the loop to account for the first iteration
        // as well as later iterations.
        let later = join(env, &exit);
        if later != *env {
            exit = later.clone();
            self.visit_children(node, &mut exit)?;
        }

        *env = exit;
        Ok(())
    }

    fn visit_cases(&self, node: &IrNode, env: &mut TargetStates) -> Result<(), LabOneQError> {
        let mut joined: Option<TargetStates> = None;
        for case in &node.children {
            let mut branch = env.clone();
            self.visit_children(&case.node, &mut branch)?;
            joined = Some(match joined {
                None => branch,
                Some(acc) => join(&acc, &branch),
            });
        }
        if let Some(joined) = joined {
            *env = joined;
        }
        Ok(())
    }

    fn require_live(
        &self,
        env: &TargetStates,
        target: Target,
        usage: Usage,
    ) -> Result<(), LabOneQError> {
        match self.index(target) {
            Some(i) if env[i] != States::LIVE => Err(laboneq_error!(
                "{} is {} while it may be stale or uninitialized. Call \
                 expect(is_live(...)) before this operation.",
                self.describe(target),
                self.describe_usage(usage)
            )),
            _ => Ok(()),
        }
    }

    fn index(&self, target: Target) -> Option<usize> {
        match target {
            Target::Variable(v) => {
                let i = v.0 as usize;
                (i < self.hqcs.variables.len()).then_some(i)
            }
            Target::Pulse(p) => self.pulses.get(&p).copied(),
        }
    }

    fn describe(&self, target: Target) -> String {
        match target {
            Target::Variable(v) => {
                let name = self.hqcs.variables[v.0 as usize]
                    .name
                    .as_deref()
                    .unwrap_or("<unnamed>");
                format!("Variable '{name}'")
            }
            Target::Pulse(p) => format!("Pulse '{}'", p.0),
        }
    }

    fn describe_usage(&self, usage: Usage) -> String {
        match usage {
            Usage::Match(section) => format!("matched in section '{}'", section.0),
            Usage::Comparison => "compared".to_owned(),
            Usage::Play(signal) => format!("played on signal '{}'", signal.0),
        }
    }
}

#[derive(Clone, Copy)]
enum Usage {
    Match(SectionUid),
    Comparison,
    Play(SignalUid),
}
