// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::{
    ExperimentNode,
    node::Node,
    operation::{Operation, Section},
    types::SectionAlignment,
};
use laboneq_ir::MatchTarget;

use crate::ParameterStore;

pub(crate) fn resolve_nt_match_case(
    ir: &mut ExperimentNode,
    near_time_parameters: &ParameterStore,
) {
    assert!(
        matches!(ir.kind, Operation::RealTimeBoundary),
        "wrong RT root node type"
    );
    if near_time_parameters.available_parameters().is_empty() {
        return;
    }
    resolve_nt_match_case_impl(ir, near_time_parameters);
}

fn resolve_nt_match_case_impl(node: &mut ExperimentNode, near_time_parameters: &ParameterStore) {
    let (nt_param_val, play_after) = if let Operation::Match(ref m) = node.kind
        && let MatchTarget::SweepParameter(ref p) = m.target
        && let Some(val) = near_time_parameters.get(p)
    {
        (Some(val), m.play_after.clone())
    } else {
        (None, vec![])
    };
    if let Some(val) = nt_param_val {
        for case in node.take_children().iter_mut() {
            if let Operation::Case(ref c) = case.kind {
                if c.state == *val {
                    let mut section = Node::new(Operation::Section(Section {
                        uid: c.uid,
                        alignment: SectionAlignment::Left,
                        length: None,
                        play_after,
                        triggers: vec![],
                        on_system_grid: false,
                    }));
                    section.children = case.make_mut().take_children();
                    *node = section;
                    break;
                }
            } else {
                panic!("Immediate children of Match section have to be Case sections")
            }
        }
    };
    for child in node.children.iter_mut() {
        resolve_nt_match_case_impl(child.make_mut(), near_time_parameters);
    }
}
