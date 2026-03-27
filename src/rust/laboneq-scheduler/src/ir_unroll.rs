// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{ParameterUid, SweepParameter};
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{IrKind, MatchTarget};

use crate::error::Result;
use crate::parameter_resolver::ParameterResolver;
use crate::scheduled_node::NodeRef;
use crate::{ParameterStore, ScheduledNode};

/// This function modifies the IR to unroll all sweep unconditionally.
///
/// The unrolling will do the following:
///
/// - All the loops will be unrolled according to their iteration count
/// - [IrKind::Match] and [IrKind::Case] acting on sweep parameters will be resolved to [IrKind::Section]s
pub(crate) fn unroll_loops(
    node: ScheduledNode,
    parameters: &HashMap<ParameterUid, SweepParameter>,
    nt_parameters: &ParameterStore,
) -> Result<ScheduledNode> {
    let resolver = ParameterResolver::new(parameters, nt_parameters);
    let node = unroll_loops_impl(node.into(), &resolver)?;
    Ok(node
        .try_unwrap()
        .expect("Expected to have unique ownership of the root node after unrolling loops"))
}

fn unroll_loops_impl(node: NodeRef, resolver: &ParameterResolver) -> Result<NodeRef> {
    // Do not handle leaf nodes, as they do not contain any loops or matches.
    if node.children.is_empty() {
        return Ok(node);
    }

    if let IrKind::Loop(obj) = &node.kind {
        let obj = obj.clone();
        let mut node = node.unwrap_or_clone();

        let parameters = &obj.parameters();
        let mut resolver = resolver.child_scope(parameters)?;

        // Check if loop is already unrolled.
        if parameters.is_empty() || obj.iterations.get() as usize == node.children.len() {
            for (iteration, child) in node.take_children().into_iter().enumerate() {
                for param in parameters.iter() {
                    resolver.set_iteration(*param, iteration)?;
                }
                let c = unroll_loops_impl(child.node, &resolver)?;
                node.add_rc_child(0.into(), c);
            }
            return Ok(node.into());
        }

        // Unroll the loop by cloning the body for the number of iterations and setting the correct iteration for each parameter in the resolver.
        let prototype = node.take_children().pop().unwrap();
        node.children = Vec::with_capacity(obj.iterations.get() as usize);
        for iteration in 0..obj.iterations.get() {
            for param in parameters.iter() {
                resolver.set_iteration(*param, iteration as usize)?;
            }
            let new_c = unroll_loops_impl(prototype.node.clone_ref(), &resolver)?;
            node.add_rc_child(0.into(), new_c);
        }
        return Ok(node.into());
    }

    // Dissolve match statements on sweep parameters by selecting the appropriate case
    // and convert them to `Section`s.
    // TODO: What to do with `Loop` that only uses the parameter in the match statement? Flatten and inline?
    if let IrKind::Match(obj) = &node.kind
        && let MatchTarget::SweepParameter(param) = &obj.target
    {
        let target_iteration = resolver.current_iteration(param)?;
        // Convert Match and Case to Sections
        let mut scheduled_case = node.children[target_iteration].clone();
        let case_section = match &scheduled_case.node.kind {
            IrKind::Case(obj) => IrKind::Section(SectionBuilder::new(obj.uid).build()),
            _ => unreachable!("Match statements can only contain case sections."),
        };
        let case_node = scheduled_case.node.make_mut();
        case_node.kind = case_section;
        for case_child in case_node.take_children() {
            let newz = unroll_loops_impl(case_child.node, resolver)?;
            case_node.add_rc_child(case_child.offset, newz);
        }
        let new_n = ScheduledNode {
            kind: IrKind::Section(SectionBuilder::new(obj.uid).build()),
            schedule: node.schedule.clone(),
            children: vec![scheduled_case],
        };
        return Ok(new_n.into());
    }

    let mut node = node.unwrap_or_clone();
    for child in node.take_children() {
        let new_c = unroll_loops_impl(child.node, resolver)?;
        node.add_rc_child(0.into(), new_c);
    }
    Ok(node.into())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use laboneq_dsl::types::SectionUid;
    use laboneq_ir::{Case, IrKind, Loop, LoopKind, Match};

    use crate::ParameterStoreBuilder;
    use crate::scheduled_node::ir_node_structure;

    fn create_section_kind(uid: SectionUid) -> IrKind {
        IrKind::Section(SectionBuilder::new(uid).build())
    }

    #[test]
    fn test_unroll_loop() {
        let parameter0 = SweepParameter::new(1.into(), Vec::from_iter(0..4)).unwrap();
        let loop_top = Loop {
            uid: 0.into(),
            iterations: NonZeroU32::new(8).unwrap(),
            kind: LoopKind::Sweeping { parameters: vec![] },
        };
        let loop_to_unroll = Loop {
            uid: 1.into(),
            iterations: NonZeroU32::new(parameter0.len() as u32).unwrap(),
            kind: LoopKind::Sweeping {
                parameters: vec![parameter0.uid],
            },
        };
        let root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top.clone()),
                [(
                    0,
                    IrKind::Loop(loop_to_unroll.clone()),
                    [(0, IrKind::LoopIteration, [])]
                ),]
            )]
        );
        let root =
            unroll_loops(root, &HashMap::new(), &ParameterStoreBuilder::new().build()).unwrap();
        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(
                    0,
                    IrKind::Loop(loop_to_unroll),
                    [
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, [])
                    ]
                ),]
            )]
        );
        assert_eq!(root, root_expected);
        // Unroll again, should have no effect
        let root =
            unroll_loops(root, &HashMap::new(), &ParameterStoreBuilder::new().build()).unwrap();
        assert_eq!(root, root_expected);
    }

    /// Test that unrolling a loop containing a match on a sweep parameter works correctly.
    #[test]
    fn test_unroll_loop_parameter_match_handling() {
        let parameter0 = SweepParameter::new(1.into(), Vec::from_iter(0..2)).unwrap();
        let loop_to_unroll = Loop {
            uid: 1.into(),
            iterations: NonZeroU32::new(parameter0.len() as u32).unwrap(),
            kind: LoopKind::Sweeping {
                parameters: vec![parameter0.uid],
            },
        };

        let section_match_uid = 2.into();
        let section_case_0_uid = 3.into();
        let section_case_1_uid = 4.into();

        let match_ = Match {
            uid: section_match_uid,
            target: MatchTarget::SweepParameter(parameter0.uid),
            local: false,
        };
        let case_0 = Case {
            uid: section_case_0_uid,
            state: 0,
        };
        let case_1 = Case {
            uid: section_case_1_uid,
            state: 1,
        };
        let root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_to_unroll.clone()),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        IrKind::Match(match_),
                        [(0, IrKind::Case(case_0), []), (0, IrKind::Case(case_1), [])]
                    )]
                ),]
            )]
        );

        let root =
            unroll_loops(root, &HashMap::new(), &ParameterStoreBuilder::new().build()).unwrap();

        // Expected structure after unrolling:
        // The match should be replaced by sections corresponding to the selected case
        // for each iteration of the loop.
        let section_match_uid = 2.into();
        let section_case_0_uid = 3.into();
        let section_case_1_uid = 4.into();
        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_to_unroll.clone()),
                [
                    (
                        0,
                        IrKind::LoopIteration,
                        [(
                            0,
                            create_section_kind(section_match_uid),
                            [(0, create_section_kind(section_case_0_uid), [])]
                        )]
                    ),
                    (
                        0,
                        IrKind::LoopIteration,
                        [(
                            0,
                            create_section_kind(section_match_uid),
                            [(0, create_section_kind(section_case_1_uid), [])]
                        )]
                    ),
                ]
            )]
        );
        assert_eq!(root, root_expected);
    }
}
