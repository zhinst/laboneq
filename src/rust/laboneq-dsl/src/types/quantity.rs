// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unit {
    Volt,
    Dbm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub unit: Option<Unit>,
}
