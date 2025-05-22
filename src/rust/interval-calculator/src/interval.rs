// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::hash::Hash;

#[derive(Clone, Eq, PartialEq, Hash, Default, Debug)]
pub struct OrderedRange<Num: Ord>(pub std::ops::Range<Num>);

impl<Num: Ord> OrderedRange<Num> {
    pub fn overlaps_range(&self, start: Num, stop: Num) -> bool {
        self.0.start < stop && self.0.end > start
    }

    pub fn overlaps_point(&self, point: Num) -> bool {
        self.0.contains(&point)
    }
}

impl<Num: Ord> Ord for OrderedRange<Num> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.start.cmp(&other.0.start) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => self.0.end.cmp(&other.0.end),
        }
    }
}

impl<Num: Ord> PartialOrd for OrderedRange<Num> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl OrderedRange<i64> {
    pub fn length(&self) -> i64 {
        self.0.end - self.0.start
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Default, Debug)]
pub struct Interval<Num: Ord> {
    span: OrderedRange<Num>,
    pub data: Vec<usize>,
}

impl Interval<i64> {
    pub fn length(&self) -> i64 {
        self.span.0.end - self.span.0.start
    }
}

impl<Num: Ord + Clone> Interval<Num> {
    pub fn new(start: Num, end: Num, data: Vec<usize>) -> Self {
        Interval {
            span: OrderedRange(start..end),
            data,
        }
    }

    pub fn from_range(range: std::ops::Range<Num>, data: Vec<usize>) -> Self {
        Interval {
            span: OrderedRange(range),
            data,
        }
    }

    pub fn start(&self) -> Num {
        self.span.0.start.clone()
    }

    pub fn start_mut(&mut self) -> &mut Num {
        &mut self.span.0.start
    }

    pub fn end(&self) -> Num {
        self.span.0.end.clone()
    }

    pub fn end_mut(&mut self) -> &mut Num {
        &mut self.span.0.end
    }

    pub fn span(&self) -> &std::ops::Range<Num> {
        &self.span.0
    }

    pub fn overlaps_range(&self, start: Num, stop: Num) -> bool {
        self.span.overlaps_range(start, stop)
    }

    pub fn overlaps_point(&self, point: Num) -> bool {
        self.span.overlaps_point(point)
    }
}

impl<Num: Ord> Ord for Interval<Num> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.span.cmp(&other.span)
    }
}

impl<Num: Ord> PartialOrd for Interval<Num> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
