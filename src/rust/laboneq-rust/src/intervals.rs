// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use interval_tree::ArrayBackedIntervalTree;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::PyTraverseError;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

#[derive(Clone, Eq, PartialEq, Hash)]
struct OrderedRange<Num: Ord>(std::ops::Range<Num>);

impl<Num: Ord> OrderedRange<Num> {
    fn overlaps_range(&self, start: Num, stop: Num) -> bool {
        self.0.start < stop && self.0.end > start
    }

    fn overlaps_point(&self, point: Num) -> bool {
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

#[pyclass]
pub struct Interval {
    range: OrderedRange<i64>,
    data: PyObject,
}

impl Clone for Interval {
    fn clone(&self) -> Interval {
        Python::with_gil(|py| Interval {
            range: self.range.clone(),
            data: self.data.clone_ref(py),
        })
    }
}

impl Interval {
    fn range(&self) -> &OrderedRange<i64> {
        &self.range
    }
}

impl PartialEq for Interval {
    fn eq(&self, _other: &Self) -> bool {
        self.range.eq(&_other.range)
    }
}

impl Eq for Interval {}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.range.cmp(&other.range)
    }
}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[pymethods]
impl Interval {
    #[new]
    #[pyo3(signature = (begin, end, data=None))]
    fn new(begin: i64, end: i64, data: Option<PyObject>, py: Python) -> PyResult<Self> {
        // TODO: bio intervals does not support zero length intervals out-of-the box.
        // It accepts them, but does not include them in the queries.
        // rust-lapper crate has support for them, but a custom implementation
        // should be possible.
        match begin.cmp(&end) {
            Ordering::Equal => Err(PyValueError::new_err(
                "Zero length intervals are not allowed",
            )),
            Ordering::Greater => Err(PyValueError::new_err("Begin cannot be larger than end")),
            Ordering::Less => Ok(Interval {
                range: OrderedRange(begin..end),
                data: data.unwrap_or(py.None()),
            }),
        }
    }

    fn __eq__(&self, other: &Self, py: Python) -> PyResult<bool> {
        if self.range() != other.range() {
            return Ok(false);
        }
        match self.data.bind(py).eq(other.data.bind(py)) {
            Ok(e) => Ok(e),
            Err(e) => Err(e),
        }
    }

    fn __lt__(&self, other: &Self) -> bool {
        self.range() < other.range()
    }

    fn __gt__(&self, other: &Self) -> bool {
        self.range() > other.range()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Interval(begin={}, end={}, data={})",
            &self.range().0.start,
            &self.range().0.end,
            &self.data
        ))
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.range.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn data(&self) -> PyResult<&PyObject> {
        Ok(&self.data)
    }

    #[getter]
    fn begin(&self) -> PyResult<i64> {
        Ok(self.range().0.start)
    }

    #[getter]
    fn end(&self) -> PyResult<i64> {
        Ok(self.range().0.end)
    }

    #[pyo3(signature = (begin, end=None))]
    fn overlap(&self, begin: i64, end: Option<i64>) -> bool {
        match end {
            Some(value) => self.range().overlaps_range(begin, value),
            None => self.range().overlaps_point(begin),
        }
    }

    fn length(&self) -> i64 {
        self.range().0.end - self.range().0.start
    }

    // Python garbage collection integration
    // __traverse__ and __clear__ must be implemented according to PyO3
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.data)
    }

    fn __clear__(&mut self, py: Python) {
        // Clear reference, this decrements ref counter.
        self.data = py.None();
    }
}

#[pyclass]
pub struct IntervalTree {
    start: i64,
    stop: i64,
    tree: ArrayBackedIntervalTree<i64, usize>,
    intervals: Vec<Interval>,
    is_sorted: bool,
}

impl IntervalTree {
    fn len(&self) -> usize {
        self.intervals.len()
    }

    fn create_intervals(&mut self, start: i64, stop: i64) -> Vec<Interval> {
        if !self.is_sorted {
            self.sort();
        }
        let mut start = start;
        let mut stop = stop;
        // TODO: At some point LabOne Q queries in wrong order, I dont know why
        // this is here now.
        if start > stop {
            std::mem::swap(&mut start, &mut stop);
        }
        let overlaps: Vec<_> = self
            .tree
            .find(start..stop)
            .into_iter()
            .map(|x| self.intervals[*x.data()].clone())
            .collect();
        overlaps
    }
}

#[pymethods]
impl IntervalTree {
    #[new]
    #[pyo3(signature = (intervals=None))]
    fn new(intervals: Option<Vec<Interval>>) -> Self {
        let mut intervals = intervals.unwrap_or_default();
        let mut start: i64 = i64::MAX;
        let mut stop: i64 = i64::MIN;
        if !intervals.is_empty() {
            intervals.sort();
            start = intervals.first().unwrap().range().0.start;
            stop = intervals.last().unwrap().range().0.end;
        };
        IntervalTree {
            start,
            stop,
            tree: ArrayBackedIntervalTree::from_iter(
                intervals
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| (x.range().clone().0, idx)),
            ),
            intervals,
            is_sorted: true,
        }
    }

    #[pyo3(signature = (begin, end, data=None))]
    fn addi(
        &mut self,
        begin: i64,
        end: i64,
        data: Option<PyObject>,
        py: Python,
    ) -> PyResult<Option<PyObject>> {
        match begin.cmp(&end) {
            Ordering::Equal => Err(PyValueError::new_err(
                "Zero length intervals are not allowed",
            )),
            Ordering::Greater => Err(PyValueError::new_err("Begin cannot be larger than end")),
            Ordering::Less => {
                if begin < self.start {
                    self.start = begin
                }
                if end > self.stop {
                    self.stop = end
                }
                self.intervals.push(Interval {
                    range: OrderedRange(begin..end),
                    data: data.unwrap_or(py.None()),
                });
                let len = self.len() + 1;
                self.tree.insert(begin..end, len);
                self.is_sorted = false;
                Ok(None)
            }
        }
    }

    /// Merge intervals and data associated with them with `func(obj1, obj2)`
    ///
    /// TODO: func to optional
    fn merge_overlaps(&mut self, func: PyObject, py: Python) -> PyResult<()> {
        self.intervals.sort();
        let intervals = std::mem::take(&mut self.intervals);
        let mut intervals: VecDeque<Interval> = intervals.into();

        let mut stack: VecDeque<Interval> = VecDeque::new();
        if let Some(first) = intervals.pop_front() {
            stack.push_back(first);
            while let Some(interval) = intervals.pop_front() {
                let mut top = stack.pop_back().unwrap();
                if top.range().0.end <= interval.range().0.start {
                    stack.push_back(top);
                    stack.push_back(interval);
                } else if top.range().0.end < interval.range().0.end {
                    top.data = func.call1(py, (&top.data, &interval.data))?;
                    top.range.0.end = interval.range().0.end;
                    stack.push_back(top);
                } else {
                    top.data = func.call1(py, (&top.data, &interval.data))?;
                    stack.push_back(top);
                }
            }
            self.intervals = stack.into();
            let intervals = &self.intervals;
            self.tree = ArrayBackedIntervalTree::from_iter(
                intervals
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| (x.range.0.clone(), idx)),
            );
            self.is_sorted = true;
        }
        Ok(())
    }

    fn at(&mut self, value: i64) -> Vec<Interval> {
        if self.intervals.is_empty() {
            return Vec::new();
        }
        self.overlap(value, value)
    }

    fn overlaps_range(&self, begin: i64, end: i64) -> bool {
        if self.intervals.is_empty() {
            return false;
        }
        begin < self.stop && end > self.start
    }

    fn overlap(&mut self, begin: i64, end: i64) -> Vec<Interval> {
        if self.intervals.is_empty() {
            return Vec::new();
        }
        let mut end = end;
        if begin == end {
            end += 1;
        }
        self.create_intervals(begin, end)
    }

    #[getter]
    fn intervals(&mut self) -> Vec<Interval> {
        if self.intervals.is_empty() {
            return Vec::new();
        }
        self.intervals.clone()
    }

    fn sort(&mut self) {
        if self.is_sorted {
            return;
        }
        self.intervals.sort();
        let intervals = &self.intervals;
        let is = intervals
            .iter()
            .enumerate()
            .map(|(idx, x)| (x.range.0.clone(), idx));
        self.tree = ArrayBackedIntervalTree::from_iter(is);
        self.is_sorted = true;
    }

    fn begin(&self) -> PyResult<i64> {
        match self.is_empty() {
            true => Err(PyRuntimeError::new_err("Interval tree is empty.")),
            false => Ok(self.start),
        }
    }

    fn end(&self) -> PyResult<i64> {
        match self.is_empty() {
            true => Err(PyRuntimeError::new_err("Interval tree is empty.")),
            false => Ok(self.stop),
        }
    }

    fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    fn length(&self) -> usize {
        self.len()
    }

    fn is_sorted(&self) -> bool {
        self.is_sorted
    }
}