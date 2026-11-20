// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use laboneq_common::named_id::NamedId;
use laboneq_dsl::types::VariableUid;
use laboneq_units::duration::{Duration, Second};

use super::*;
use crate::schedule_info::ScheduleInfoBuilder;

const X: CoprocessorValue = CoprocessorValue::Variable(VariableUid(0));
const Y: CoprocessorValue = CoprocessorValue::Variable(VariableUid(1));

#[test]
fn left_aligned_read_waits_for_preceding_write() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((X, Access::Read))),
        ],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn left_aligned_write_waits_for_all_preceding_reads() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [
            operation(100, 0, Some((X, Access::Read))),
            operation(10, 1, Some((X, Access::Read))),
            operation(10, 2, Some((X, Access::Write))),
        ],
    );

    assert_eq!(offsets, [0, 0, 100].map(tiny_samples));
}

#[test]
fn accesses_to_different_values_can_overlap() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((Y, Access::Read))),
        ],
    );

    assert_eq!(offsets, [0, 0].map(tiny_samples));
}

#[test]
fn right_aligned_write_finishes_before_all_following_reads() {
    let offsets = arrange_children(
        SectionAlignment::Right,
        [
            operation(10, 0, Some((X, Access::Write))),
            operation(100, 1, Some((X, Access::Read))),
            operation(10, 2, Some((X, Access::Read))),
        ],
    );

    assert_eq!(offsets, [0, 10, 100].map(tiny_samples));
}

#[test]
fn right_aligned_read_finishes_before_following_write() {
    let offsets = arrange_children(
        SectionAlignment::Right,
        [
            operation(100, 0, Some((X, Access::Read))),
            operation(10, 1, Some((X, Access::Write))),
        ],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn left_aligned_write_waits_for_preceding_write() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((X, Access::Write))),
        ],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn right_aligned_write_finishes_before_following_write() {
    let offsets = arrange_children(
        SectionAlignment::Right,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((X, Access::Write))),
        ],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn left_aligned_operations_on_the_same_signal_run_in_order() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [operation(100, 0, None), operation(10, 0, None)],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn left_aligned_operations_on_different_signals_can_overlap() {
    let offsets = arrange_children(
        SectionAlignment::Left,
        [operation(100, 0, None), operation(10, 1, None)],
    );

    assert_eq!(offsets, [0, 0].map(tiny_samples));
}

#[test]
fn right_aligned_operations_on_the_same_signal_run_in_order() {
    let offsets = arrange_children(
        SectionAlignment::Right,
        [operation(100, 0, None), operation(10, 0, None)],
    );

    assert_eq!(offsets, [0, 100].map(tiny_samples));
}

#[test]
fn right_aligned_operations_on_different_signals_can_overlap() {
    let offsets = arrange_children(
        SectionAlignment::Right,
        [operation(100, 0, None), operation(10, 1, None)],
    );

    assert_eq!(offsets, [0, 90].map(tiny_samples));
}

#[test]
fn left_aligned_nested_sections_respect_shared_signals() {
    let nested = section(
        SectionAlignment::Right,
        [operation(100, 0, None), operation(10, 0, None)],
    );
    let mut outer = section(
        SectionAlignment::Left,
        [operation(20, 0, None), nested, operation(30, 0, None)],
    );

    calculate_timing(&mut outer, None::<&NoFeedback>).unwrap();

    assert_eq!(outer.schedule.length(), tiny_samples(160));
    assert_eq!(outer.children[1].offset, tiny_samples(20));
    assert_eq!(outer.children[2].offset, tiny_samples(130));
    let nested = &outer.children[1].node;
    assert_eq!(nested.schedule.length(), tiny_samples(110));
    assert_eq!(nested.schedule.absolute_start, tiny_samples(20));
    assert_eq!(nested.children[0].offset, tiny_samples(0));
    assert_eq!(nested.children[1].offset, tiny_samples(100));
    assert_eq!(
        nested.children[0].node.schedule.absolute_start,
        tiny_samples(20)
    );
    assert_eq!(
        nested.children[1].node.schedule.absolute_start,
        tiny_samples(120)
    );
}

#[test]
fn right_aligned_nested_sections_respect_shared_signals() {
    let nested = section(
        SectionAlignment::Left,
        [operation(100, 0, None), operation(10, 0, None)],
    );
    let mut outer = section(
        SectionAlignment::Right,
        [operation(20, 0, None), nested, operation(30, 0, None)],
    );

    calculate_timing(&mut outer, None::<&NoFeedback>).unwrap();

    assert_eq!(outer.schedule.length(), tiny_samples(160));
    assert_eq!(outer.children[1].offset, tiny_samples(20));
    assert_eq!(outer.children[2].offset, tiny_samples(130));
    let nested = &outer.children[1].node;
    assert_eq!(nested.schedule.length(), tiny_samples(110));
    assert_eq!(nested.schedule.absolute_start, tiny_samples(20));
    assert_eq!(nested.children[0].offset, tiny_samples(0));
    assert_eq!(nested.children[1].offset, tiny_samples(100));
    assert_eq!(
        nested.children[0].node.schedule.absolute_start,
        tiny_samples(20)
    );
    assert_eq!(
        nested.children[1].node.schedule.absolute_start,
        tiny_samples(120)
    );
}

fn section(
    alignment: SectionAlignment,
    children: impl IntoIterator<Item = ScheduledNode>,
) -> ScheduledNode {
    let schedule = ScheduleInfoBuilder::new().alignment_mode(alignment).build();
    let mut section = ScheduledNode::new(IrKind::LoopIteration, schedule);
    for child in children {
        section.schedule.signals.extend(&child.schedule.signals);
        section.schedule.merge_accesses(&child.schedule.accesses);
        section.add_child(tiny_samples(0), child);
    }
    section
}

#[test]
fn left_aligned_nested_sections_respect_variable_accesses() {
    let nested = section(
        SectionAlignment::Right,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((X, Access::Read))),
        ],
    );
    let mut outer = section(
        SectionAlignment::Left,
        [
            operation(20, 2, Some((X, Access::Read))),
            nested,
            operation(30, 3, Some((X, Access::Read))),
        ],
    );

    calculate_timing(&mut outer, None::<&NoFeedback>).unwrap();

    assert_eq!(outer.schedule.length(), tiny_samples(160));
    assert_eq!(outer.children[1].offset, tiny_samples(20));
    assert_eq!(outer.children[2].offset, tiny_samples(130));
    let nested = &outer.children[1].node;
    assert_eq!(nested.schedule.length(), tiny_samples(110));
    assert_eq!(nested.schedule.absolute_start, tiny_samples(20));
    assert_eq!(nested.children[0].offset, tiny_samples(0));
    assert_eq!(nested.children[1].offset, tiny_samples(100));
    assert_eq!(
        nested.children[0].node.schedule.absolute_start,
        tiny_samples(20)
    );
    assert_eq!(
        nested.children[1].node.schedule.absolute_start,
        tiny_samples(120)
    );
}

#[test]
fn right_aligned_nested_sections_respect_variable_accesses() {
    let nested = section(
        SectionAlignment::Left,
        [
            operation(100, 0, Some((X, Access::Write))),
            operation(10, 1, Some((X, Access::Read))),
        ],
    );
    let mut outer = section(
        SectionAlignment::Right,
        [
            operation(20, 2, Some((X, Access::Read))),
            nested,
            operation(30, 3, Some((X, Access::Read))),
        ],
    );

    calculate_timing(&mut outer, None::<&NoFeedback>).unwrap();

    assert_eq!(outer.schedule.length(), tiny_samples(160));
    assert_eq!(outer.children[1].offset, tiny_samples(20));
    assert_eq!(outer.children[2].offset, tiny_samples(130));
    let nested = &outer.children[1].node;
    assert_eq!(nested.schedule.length(), tiny_samples(110));
    assert_eq!(nested.schedule.absolute_start, tiny_samples(20));
    assert_eq!(nested.children[0].offset, tiny_samples(0));
    assert_eq!(nested.children[1].offset, tiny_samples(100));
    assert_eq!(
        nested.children[0].node.schedule.absolute_start,
        tiny_samples(20)
    );
    assert_eq!(
        nested.children[1].node.schedule.absolute_start,
        tiny_samples(120)
    );
}

/// Return child start offsets for a section with the given alignment.
fn arrange_children<const N: usize>(
    alignment: SectionAlignment,
    children: [ScheduledNode; N],
) -> [TinySamples; N] {
    let schedule = ScheduleInfoBuilder::new().alignment_mode(alignment).build();
    let mut section = ScheduledNode::new(IrKind::LoopIteration, schedule);
    for child in children {
        section.add_child(tiny_samples(0), child);
    }

    calculate_timing(&mut section, None::<&NoFeedback>).unwrap();

    std::array::from_fn(|i| section.children[i].offset)
}

/// Create an operation with a duration in tiny samples and optional variable access.
fn operation(
    length: i64,
    signal: u32,
    variable: Option<(CoprocessorValue, Access)>,
) -> ScheduledNode {
    let signal = SignalUid(NamedId::debug_id(signal));
    let mut schedule = ScheduleInfoBuilder::new()
        .length(length)
        .signals([signal].into());
    if let Some((value, access)) = variable {
        schedule = schedule.access(value, access);
    }
    ScheduledNode::new(IrKind::Delay { signal }, schedule.build())
}

struct NoFeedback;

impl FeedbackCalculator for NoFeedback {
    type Error = Infallible;

    fn compute_feedback_latency(
        &self,
        _: Duration<Second>,
        _: Duration<Second>,
        _: bool,
        _: SignalUid,
        _: &[SignalUid],
    ) -> std::result::Result<Duration<Second>, Infallible> {
        unreachable!("feedback latency is unused because these tests contain no acquisitions")
    }
}
