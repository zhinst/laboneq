// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{ParameterUid, SectionUid, SweepParameter};

use crate::error::{Error, Result};
use crate::ir::builders::SectionBuilder;
use crate::ir::{IrKind, MatchTarget};
use crate::parameter_resolver::ParameterResolver;
use crate::{ParameterStore, ScheduledNode};

/// This function modifies the IR to unroll all sweep unconditionally.
///
/// The unrolling will do the following:
///
/// - All the loops will be unrolled according to their iteration count
/// - [IrKind::Match] and [IrKind::Case] acting on sweep parameters will be resolved to [IrKind::Section]s
pub(crate) fn unroll_loops(
    node: &mut ScheduledNode,
    parameters: &HashMap<ParameterUid, SweepParameter>,
    nt_parameters: &ParameterStore,
) -> Result<()> {
    let resolver = ParameterResolver::new(parameters, nt_parameters);
    unroll_loops_impl(node, &resolver)
}

fn unroll_loops_impl(node: &mut ScheduledNode, resolver: &ParameterResolver) -> Result<()> {
    match &node.kind {
        IrKind::Loop(obj) => {
            let parameters = &obj.parameters();
            let mut resolver = resolver.child_scope(parameters)?;
            if parameters.is_empty() || obj.iterations == node.children.len() {
                // Loop is already unrolled
                for (iteration, child) in node.children.iter_mut().enumerate() {
                    for param in parameters.iter() {
                        resolver.set_iteration(*param, iteration)?;
                    }
                    unroll_loops_impl(child.node.make_mut(), &resolver)?;
                }
                return Ok(());
            }
            assert_eq!(
                node.children.len(),
                1,
                "Loop must have exactly one child to unroll."
            );
            let prototype = node.children.pop().unwrap();
            node.children = Vec::with_capacity(obj.iterations);
            for iteration in 0..obj.iterations {
                for param in parameters.iter() {
                    resolver.set_iteration(*param, iteration)?;
                }
                let mut proto = prototype.clone();
                unroll_loops_impl(proto.node.make_mut(), &resolver)?;
                node.children.push(proto);
            }
        }
        IrKind::Match(obj) if matches!(obj.target, MatchTarget::SweepParameter(_)) => {
            // Dissolve match statements on sweep parameters by selecting the appropriate case
            // and convert them to `Section`s.
            // TODO: What to do with `Loop` that only uses the parameter in the match statement? Flatten and inline?
            if let MatchTarget::SweepParameter(param) = &obj.target {
                let target_iteration = resolver.current_iteration(param)?;
                let target_case = node.children.iter().position(|child| {
                    matches!(&child.node.kind, IrKind::Case(obj) if obj.state == target_iteration)
                }).ok_or_else(|| Error::new(format!("Missing a case for iteration {target_iteration}")))?; // Should be handled earlier
                // Convert Match and Case to Sections
                let mut scheduled_case = node.children[target_case].clone();
                let case_section = match &scheduled_case.node.kind {
                    IrKind::Case(obj) => create_section_kind(obj.uid),
                    _ => unreachable!("Match statements can only contain case sections."),
                };
                let case_node = scheduled_case.node.make_mut();
                case_node.kind = case_section;
                for case_child in case_node.children.iter_mut() {
                    unroll_loops_impl(case_child.node.make_mut(), resolver)?;
                }
                // Take the grid of the child
                node.schedule.grid = scheduled_case.node.schedule.grid;
                node.schedule.sequencer_grid = scheduled_case.node.schedule.sequencer_grid;
                node.kind = create_section_kind(obj.uid);
                node.children = vec![scheduled_case];
            }
        }
        _ => {
            for child in node.children.iter_mut() {
                unroll_loops_impl(child.node.make_mut(), resolver)?;
            }
        }
    };
    Ok(())
}

fn create_section_kind(uid: SectionUid) -> IrKind {
    IrKind::Section(SectionBuilder::new(uid).build())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ParameterStoreBuilder;
    use crate::ir::{Case, IrKind, Loop, LoopKind, Match};
    use crate::scheduled_node::ir_node_structure;
    use laboneq_common::named_id::NamedId;

    #[test]
    fn test_unroll_loop() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..4));
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            kind: LoopKind::Sweeping { parameters: vec![] },
        };
        let loop_to_unroll = Loop {
            uid: SectionUid(NamedId::debug_id(1)),
            iterations: parameter0.len(),
            kind: LoopKind::Sweeping {
                parameters: vec![parameter0.uid],
            },
        };
        let mut root = ir_node_structure!(
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
        unroll_loops(
            &mut root,
            &HashMap::new(),
            &ParameterStoreBuilder::new().build(),
        )
        .unwrap();
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
        unroll_loops(
            &mut root,
            &HashMap::new(),
            &ParameterStoreBuilder::new().build(),
        )
        .unwrap();
        assert_eq!(root, root_expected);
    }

    /// Test that unrolling a loop containing a match on a sweep parameter works correctly.
    #[test]
    fn test_unroll_loop_parameter_match_handling() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..2));
        let loop_to_unroll = Loop {
            uid: SectionUid(NamedId::debug_id(1)),
            iterations: parameter0.len(),
            kind: LoopKind::Sweeping {
                parameters: vec![parameter0.uid],
            },
        };

        let section_match_uid = SectionUid(NamedId::debug_id(2));
        let section_case_0_uid = SectionUid(NamedId::debug_id(3));
        let section_case_1_uid = SectionUid(NamedId::debug_id(4));

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
        let mut root = ir_node_structure!(
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

        unroll_loops(
            &mut root,
            &HashMap::new(),
            &ParameterStoreBuilder::new().build(),
        )
        .unwrap();

        // Expected structure after unrolling:
        // The match should be replaced by sections corresponding to the selected case
        // for each iteration of the loop.
        let section_match_uid = SectionUid(NamedId::debug_id(2));
        let section_case_0_uid = SectionUid(NamedId::debug_id(3));
        let section_case_1_uid = SectionUid(NamedId::debug_id(4));
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
