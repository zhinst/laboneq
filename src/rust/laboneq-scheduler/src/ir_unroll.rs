// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{ParameterUid, SectionUid, SweepParameter};

use crate::error::{Error, Result};
use crate::parameter_resolver::ParameterResolver;
use crate::{ParameterStore, ScheduledNode};
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{IrKind, MatchTarget};

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
            if parameters.is_empty() || obj.iterations.get() as usize == node.children.len() {
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
            node.children = Vec::with_capacity(obj.iterations.get() as usize);
            for iteration in 0..obj.iterations.get() {
                for param in parameters.iter() {
                    resolver.set_iteration(*param, iteration as usize)?;
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
    use std::num::NonZeroU32;

    use super::*;

    use crate::ParameterStoreBuilder;
    use crate::scheduled_node::ir_node_structure;
    use laboneq_ir::{Case, IrKind, Loop, LoopKind, Match};

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
