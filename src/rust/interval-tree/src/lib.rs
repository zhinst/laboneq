// Copyright 2016 Johannes Köster, the Rust-Bio team, Google Inc.
// SPDX-License-Identifier: MIT

// This package is vendored from rust-bio. The original source code is available at
// https://github.com/rust-bio/rust-bio.

mod interval;
pub use self::interval::Interval;

mod array_backed_interval_tree;
pub use array_backed_interval_tree::ArrayBackedIntervalTree;
