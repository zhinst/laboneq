// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::ParameterUid;
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{IrKind, MatchTarget};

use crate::ScheduledNode;
use crate::schedule_info::ScheduleInfoBuilder;
use crate::scheduled_node::NodeChild;

/// This function modifies the IR to unroll all sweep unconditionally.
///
/// The unrolling will do the following:
///
/// - All the loops will be unrolled according to their iteration count
/// - [IrKind::Match] and [IrKind::Case] acting on sweep parameters will be resolved to [IrKind::Section]s
pub(crate) fn unroll_loops(node: &mut ScheduledNode) {
    unroll(node, &mut Vec::new());
}

/// The iteration of an unrolled loop, per parameter of the loop.
type Iterations = Vec<(ParameterUid, usize)>;

/// Unrolls the loops in `node` and its descendants.
///
/// `enclosing` holds the iterations of the unrolled loops around `node`.
///
/// Unrolling sounds simple, but various ways in which loops and matches can be nested, make the code intricate.
/// We need to make sure that the following requirements are satisfied:
///
/// - **Cases are moved, not copied.** In an RB-like experiment each case holds a long pulse
///   sequence, and the cases are the largest part of the tree. Naively copying cases into every
///   iteration, only to drop their original copies is costly on compiler performance.
/// - **Loops are unrolled outermost first.** Otherwise an inner loop copies every case of a
///   match on an outer parameter once per inner iteration, before the outer loop selects one.
/// - **Matches are resolved in two places.** Copying an iteration resolves the matches on the
///   loop's parameters. A match left inside a moved case, such as a nested match on the same parameter,
///   is resolved later, by the walk that unrolls the inner loops. An inner loop over an outer loop's
///   parameter shadows it.
fn unroll(node: &mut ScheduledNode, enclosing: &mut Iterations) {
    if let IrKind::Match(obj) = &node.kind
        && let MatchTarget::SweepParameter(param) = &obj.target
    {
        // Find the iteration of the matched parameter, taking the innermost loop that sweeps the parameter.
        let iteration = enclosing
            .iter()
            .rfind(|(parameter, _)| parameter == param)
            .map(|&(_, iteration)| iteration)
            .expect("parameter match not inside parameter's sweep scope");

        // Keep the case for `iteration` only
        let mut case = std::mem::replace(&mut node.children[iteration].node, dummy_node());
        node.children = Vec::with_capacity(1);
        transform_into_section(node);
        transform_into_section(&mut case);
        node.add_child(0.into(), case);
    }

    if let IrKind::Loop(obj) = &node.kind
        && !obj.parameters().is_empty()
        && node.children.len() == 1
    {
        let parameters = obj.parameters().to_vec();
        let num_iterations = obj.iterations.get() as usize;
        let mut body = node.take_children().pop().unwrap().node;
        // Unfolds `body` into `num_iterations` iterations of the loop
        node.children = (0..num_iterations)
            .map(|i| NodeChild {
                offset: 0.into(),
                node: copy_iteration(&mut body, &parameters, num_iterations, i),
            })
            .collect();

        let outer_bindings = enclosing.len();
        for (iteration, child) in node.children.iter_mut().enumerate() {
            enclosing.extend(parameters.iter().map(|&parameter| (parameter, iteration)));
            unroll(&mut child.node, enclosing);
            enclosing.truncate(outer_bindings);
        }
        return;
    }

    for child in node.children.iter_mut() {
        unroll(&mut child.node, enclosing);
    }
}

/// Copies `node` and its descendants for `iteration`.
///
/// Each match on one of `parameters` becomes a section holding its case `iteration`. The
/// case is moved out of the match, not copied, and its contents are left as they are: the
/// walk neither copies nor resolves anything inside it. Below an inner loop that sweeps
/// one of `parameters` again, the matches on that parameter are left to the inner loop.
fn copy_iteration(
    node: &mut ScheduledNode,
    parameters: &[ParameterUid],
    num_iterations: usize,
    iteration: usize,
) -> ScheduledNode {
    if is_match_on(node, parameters, num_iterations) {
        let mut case = std::mem::replace(&mut node.children[iteration].node, dummy_node());
        let mut match_ = copy_with_empty_children(node, 1);
        transform_into_section(&mut match_);
        transform_into_section(&mut case);
        match_.add_child(0.into(), case);
        return match_;
    }
    let unshadowed;
    let parameters = match unshadowed_parameters(&node.kind, parameters) {
        Some(remaining) => {
            unshadowed = remaining;
            &unshadowed
        }
        None => parameters,
    };
    let mut copy = copy_with_empty_children(node, node.children.len());
    for child in node.children.iter_mut() {
        copy.children.push(NodeChild {
            offset: child.offset,
            node: copy_iteration(&mut child.node, parameters, num_iterations, iteration),
        });
    }
    copy
}

/// Returns whether `node` is a match on one of `parameters`, which then has a case per
/// iteration of their loop.
fn is_match_on(node: &ScheduledNode, parameters: &[ParameterUid], num_iterations: usize) -> bool {
    let IrKind::Match(obj) = &node.kind else {
        return false;
    };
    let MatchTarget::SweepParameter(param) = &obj.target else {
        return false;
    };
    if !parameters.contains(param) {
        return false;
    }
    assert_eq!(
        node.children.len(),
        num_iterations,
        "number of cases not matching loop iterations"
    );
    true
}

/// Returns `parameters` without those that a loop of `kind` sweeps again, if it does.
///
/// A match inside such a loop selects by the iteration of the inner loop.
fn unshadowed_parameters(kind: &IrKind, parameters: &[ParameterUid]) -> Option<Vec<ParameterUid>> {
    let IrKind::Loop(obj) = kind else {
        return None;
    };
    let swept = obj.parameters();
    if !swept.iter().any(|param| parameters.contains(param)) {
        return None;
    }
    Some(
        parameters
            .iter()
            .filter(|param| !swept.contains(param))
            .copied()
            .collect(),
    )
}

fn copy_with_empty_children(node: &ScheduledNode, children_capacity: usize) -> ScheduledNode {
    ScheduledNode {
        kind: node.kind.clone(),
        schedule: node.schedule.clone(),
        children: Vec::with_capacity(children_capacity),
    }
}

/// Arbitrary node without meaning, to use as replacement when taking other nodes.
fn dummy_node() -> ScheduledNode {
    ScheduledNode {
        kind: IrKind::LoopIteration,
        schedule: ScheduleInfoBuilder::new().build(),
        children: vec![],
    }
}

/// Change the node kind to `Section`
fn transform_into_section(node: &mut ScheduledNode) {
    let section_uid = node.kind.section_info().expect("not a section").uid;
    node.kind = IrKind::Section(SectionBuilder::new(*section_uid).build());
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use laboneq_dsl::types::{SectionUid, SweepParameter};
    use laboneq_ir::{Case, IrKind, Loop, LoopKind, Match};

    use crate::schedule_info::ScheduleInfoBuilder;
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
        unroll_loops(&mut root);
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
        unroll_loops(&mut root);
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

        unroll_loops(&mut root);

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

    /// Test that every iteration of an outer loop receives a fully unrolled copy of an inner loop.
    #[test]
    fn test_unroll_nested_loops() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 20, 3);
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(outer.clone()),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        IrKind::Loop(inner.clone()),
                        [(0, IrKind::LoopIteration, [])]
                    )]
                )]
            )]
        );

        unroll_loops(&mut root);

        let inner_unrolled = || {
            ir_node_structure!(
                IrKind::Loop(inner.clone()),
                [
                    (0, IrKind::LoopIteration, []),
                    (0, IrKind::LoopIteration, []),
                    (0, IrKind::LoopIteration, [])
                ]
            )
        };
        let mut outer_expected = ir_node_structure!(IrKind::Loop(outer.clone()), []);
        for _ in 0..2 {
            let mut iteration = ir_node_structure!(IrKind::LoopIteration, []);
            iteration.add_child(0.into(), inner_unrolled());
            outer_expected.add_child(0.into(), iteration);
        }
        let mut root_expected = ir_node_structure!(IrKind::Root, []);
        root_expected.add_child(0.into(), outer_expected);
        assert_eq!(root, root_expected);
    }

    /// Test that a match on an outer sweep parameter inside an inner loop selects
    /// the same case in every inner iteration.
    #[test]
    fn test_unroll_match_on_outer_parameter_in_inner_loop() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 20, 3);
        let LoopKind::Sweeping { parameters } = &outer.kind else {
            unreachable!()
        };
        let match_ = Match {
            uid: 3.into(),
            target: MatchTarget::SweepParameter(parameters[0]),
            local: false,
        };
        let case_0 = Case {
            uid: 4.into(),
            state: 0,
        };
        let case_1 = Case {
            uid: 5.into(),
            state: 1,
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(outer.clone()),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        IrKind::Loop(inner.clone()),
                        [(
                            0,
                            IrKind::LoopIteration,
                            [(
                                0,
                                IrKind::Match(match_.clone()),
                                [
                                    (0, IrKind::Case(case_0.clone()), []),
                                    (0, IrKind::Case(case_1.clone()), [])
                                ]
                            )]
                        )]
                    )]
                )]
            )]
        );

        unroll_loops(&mut root);

        let mut outer_expected = ir_node_structure!(IrKind::Loop(outer.clone()), []);
        for case_uid in [case_0.uid, case_1.uid] {
            let mut inner_expected = ir_node_structure!(IrKind::Loop(inner.clone()), []);
            for _ in 0..3 {
                inner_expected.add_child(
                    0.into(),
                    ir_node_structure!(
                        IrKind::LoopIteration,
                        [(
                            0,
                            create_section_kind(match_.uid),
                            [(0, create_section_kind(case_uid), [])]
                        )]
                    ),
                );
            }
            let mut iteration = ir_node_structure!(IrKind::LoopIteration, []);
            iteration.add_child(0.into(), inner_expected);
            outer_expected.add_child(0.into(), iteration);
        }
        let mut root_expected = ir_node_structure!(IrKind::Root, []);
        root_expected.add_child(0.into(), outer_expected);
        assert_eq!(root, root_expected);
    }

    /// Test that a match on an inner sweep parameter selects the case of the inner iteration
    /// in every outer iteration.
    #[test]
    fn test_unroll_match_on_inner_parameter_in_nested_loops() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 20, 2);
        let LoopKind::Sweeping { parameters } = &inner.kind else {
            unreachable!()
        };
        let match_ = Match {
            uid: 3.into(),
            target: MatchTarget::SweepParameter(parameters[0]),
            local: false,
        };
        let case_0 = Case {
            uid: 4.into(),
            state: 0,
        };
        let case_1 = Case {
            uid: 5.into(),
            state: 1,
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(outer.clone()),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        IrKind::Loop(inner.clone()),
                        [(
                            0,
                            IrKind::LoopIteration,
                            [(
                                0,
                                IrKind::Match(match_.clone()),
                                [
                                    (0, IrKind::Case(case_0.clone()), []),
                                    (0, IrKind::Case(case_1.clone()), [])
                                ]
                            )]
                        )]
                    )]
                )]
            )]
        );

        unroll_loops(&mut root);

        let mut outer_expected = ir_node_structure!(IrKind::Loop(outer.clone()), []);
        for _ in 0..2 {
            let mut inner_expected = ir_node_structure!(IrKind::Loop(inner.clone()), []);
            for case_uid in [case_0.uid, case_1.uid] {
                inner_expected.add_child(
                    0.into(),
                    ir_node_structure!(
                        IrKind::LoopIteration,
                        [(
                            0,
                            create_section_kind(match_.uid),
                            [(0, create_section_kind(case_uid), [])]
                        )]
                    ),
                );
            }
            let mut iteration = ir_node_structure!(IrKind::LoopIteration, []);
            iteration.add_child(0.into(), inner_expected);
            outer_expected.add_child(0.into(), iteration);
        }
        let mut root_expected = ir_node_structure!(IrKind::Root, []);
        root_expected.add_child(0.into(), outer_expected);
        assert_eq!(root, root_expected);
    }

    /// Test a match is resolved to its case in the edge case of a single-iteration loop.
    #[test]
    fn test_unroll_single_iteration_loop_with_match() {
        let loop_ = sweep_loop(1, 10, 1);
        let match_ = Match {
            uid: 2.into(),
            target: MatchTarget::SweepParameter(10.into()),
            local: false,
        };
        let case = Case {
            uid: 3.into(),
            state: 0,
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_.clone()),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        IrKind::Match(match_.clone()),
                        [(0, IrKind::Case(case.clone()), [])]
                    )]
                )]
            )]
        );

        unroll_loops(&mut root);

        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_),
                [(
                    0,
                    IrKind::LoopIteration,
                    [(
                        0,
                        create_section_kind(match_.uid),
                        [(0, create_section_kind(case.uid), [])]
                    )]
                )]
            )]
        );
        assert_eq!(root, root_expected);
    }

    fn leaf(id: u32) -> ScheduledNode {
        ScheduledNode {
            kind: IrKind::Delay { signal: id.into() },
            schedule: ScheduleInfoBuilder::new().build(),
            children: vec![],
        }
    }

    fn node(kind: IrKind, children: Vec<ScheduledNode>) -> ScheduledNode {
        let mut node = ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
        for child in children {
            node.add_child(0.into(), child);
        }
        node
    }

    fn section(uid: u32, children: Vec<ScheduledNode>) -> ScheduledNode {
        node(create_section_kind(uid.into()), children)
    }

    /// A match `uid` on `parameter`, whose case `i` has uid `uid + 1 + i`.
    fn match_on(uid: u32, parameter: u32, cases: Vec<Vec<ScheduledNode>>) -> ScheduledNode {
        let cases = cases
            .into_iter()
            .enumerate()
            .map(|(state, children)| {
                let case = Case {
                    uid: (uid + 1 + state as u32).into(),
                    state,
                };
                node(IrKind::Case(case), children)
            })
            .collect();
        let match_ = Match {
            uid: uid.into(),
            target: MatchTarget::SweepParameter(parameter.into()),
            local: false,
        };
        node(IrKind::Match(match_), cases)
    }

    /// What unrolling makes of case `state` of the match `uid` built by [`match_on`].
    fn resolved(uid: u32, state: u32, children: Vec<ScheduledNode>) -> ScheduledNode {
        section(uid, vec![section(uid + 1 + state, children)])
    }

    fn loop_over(loop_: &Loop, body: Vec<ScheduledNode>) -> ScheduledNode {
        unrolled(loop_, vec![body])
    }

    fn unrolled(loop_: &Loop, iterations: Vec<Vec<ScheduledNode>>) -> ScheduledNode {
        let iterations = iterations
            .into_iter()
            .map(|body| node(IrKind::LoopIteration, body))
            .collect();
        node(IrKind::Loop(loop_.clone()), iterations)
    }

    /// Test that a match on an outer sweep parameter around an inner loop selects its case
    /// before the inner loop is unrolled.
    #[test]
    fn test_unroll_match_on_outer_parameter_around_inner_loop() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 20, 2);
        let case = |i| vec![loop_over(&inner, vec![leaf(i)])];
        let mut root = loop_over(&outer, vec![match_on(3, 10, vec![case(0), case(1)])]);

        let iteration = |i| {
            vec![resolved(
                3,
                i,
                vec![unrolled(&inner, vec![vec![leaf(i)], vec![leaf(i)]])],
            )]
        };
        unroll_loops(&mut root);
        assert_eq!(root, unrolled(&outer, vec![iteration(0), iteration(1)]));
    }

    /// Test that a match on an inner sweep parameter inside a case of a match on an outer
    /// sweep parameter selects the case of both iterations.
    #[test]
    fn test_unroll_match_on_inner_parameter_in_case_of_outer_match() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 20, 2);
        let inner_match =
            |i: u32| match_on(4, 20, vec![vec![leaf(10 * i)], vec![leaf(10 * i + 1)]]);
        let outer_match = match_on(7, 10, vec![vec![inner_match(0)], vec![inner_match(1)]]);
        let mut root = loop_over(&outer, vec![loop_over(&inner, vec![outer_match])]);

        let inner_iteration =
            |i, j| vec![resolved(7, i, vec![resolved(4, j, vec![leaf(10 * i + j)])])];
        let outer_iteration = |i| {
            vec![unrolled(
                &inner,
                vec![inner_iteration(i, 0), inner_iteration(i, 1)],
            )]
        };
        unroll_loops(&mut root);
        assert_eq!(
            root,
            unrolled(&outer, vec![outer_iteration(0), outer_iteration(1)])
        );
    }

    /// Test that a match nested in a case of a match on the same parameter selects the
    /// same case.
    #[test]
    fn test_unroll_nested_matches_on_same_parameter() {
        let loop_ = sweep_loop(1, 10, 2);
        let nested = |i: u32| match_on(4, 10, vec![vec![leaf(10 * i)], vec![leaf(10 * i + 1)]]);
        let mut root = loop_over(
            &loop_,
            vec![match_on(7, 10, vec![vec![nested(0)], vec![nested(1)]])],
        );

        let iteration = |i| vec![resolved(7, i, vec![resolved(4, i, vec![leaf(11 * i)])])];
        unroll_loops(&mut root);
        assert_eq!(root, unrolled(&loop_, vec![iteration(0), iteration(1)]));
    }

    /// Test that a match on any parameter of a loop over several parameters selects the
    /// case of the loop's iteration.
    #[test]
    fn test_unroll_loop_over_several_parameters() {
        let loop_ = Loop {
            uid: 1.into(),
            iterations: NonZeroU32::new(2).unwrap(),
            kind: LoopKind::Sweeping {
                parameters: vec![10.into(), 11.into()],
            },
        };
        let mut root = loop_over(
            &loop_,
            vec![
                match_on(2, 10, vec![vec![leaf(0)], vec![leaf(1)]]),
                match_on(5, 11, vec![vec![leaf(10)], vec![leaf(11)]]),
            ],
        );

        let iteration = |i| {
            vec![
                resolved(2, i, vec![leaf(i)]),
                resolved(5, i, vec![leaf(10 + i)]),
            ]
        };
        unroll_loops(&mut root);
        assert_eq!(root, unrolled(&loop_, vec![iteration(0), iteration(1)]));
    }

    /// Test that a match inside an inner loop over the parameter of an outer loop selects
    /// by the inner iteration.
    #[test]
    fn test_unroll_inner_loop_over_same_parameter() {
        let outer = sweep_loop(1, 10, 2);
        let inner = sweep_loop(2, 10, 2);
        let mut root = loop_over(
            &outer,
            vec![loop_over(
                &inner,
                vec![match_on(3, 10, vec![vec![leaf(0)], vec![leaf(1)]])],
            )],
        );

        let inner_unrolled = || {
            unrolled(
                &inner,
                vec![
                    vec![resolved(3, 0, vec![leaf(0)])],
                    vec![resolved(3, 1, vec![leaf(1)])],
                ],
            )
        };
        unroll_loops(&mut root);
        assert_eq!(
            root,
            unrolled(&outer, vec![vec![inner_unrolled()], vec![inner_unrolled()]])
        );
    }

    /// Test that matches on one parameter at several depths select the same case.
    #[test]
    fn test_unroll_matches_on_one_parameter_at_several_depths() {
        let loop_ = sweep_loop(1, 10, 2);
        let mut root = loop_over(
            &loop_,
            vec![
                match_on(2, 10, vec![vec![leaf(0)], vec![leaf(1)]]),
                section(
                    5,
                    vec![section(
                        6,
                        vec![match_on(7, 10, vec![vec![leaf(10)], vec![leaf(11)]])],
                    )],
                ),
            ],
        );

        let iteration = |i| {
            vec![
                resolved(2, i, vec![leaf(i)]),
                section(
                    5,
                    vec![section(6, vec![resolved(7, i, vec![leaf(10 + i)])])],
                ),
            ]
        };
        unroll_loops(&mut root);
        assert_eq!(root, unrolled(&loop_, vec![iteration(0), iteration(1)]));
    }

    fn sweep_loop(loop_uid: u32, parameter_uid: u32, iterations: u32) -> Loop {
        Loop {
            uid: loop_uid.into(),
            iterations: NonZeroU32::new(iterations).unwrap(),
            kind: LoopKind::Sweeping {
                parameters: vec![parameter_uid.into()],
            },
        }
    }
}
