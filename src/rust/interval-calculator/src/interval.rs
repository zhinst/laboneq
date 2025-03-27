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
