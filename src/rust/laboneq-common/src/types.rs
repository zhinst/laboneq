// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::named_id::NamedId;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct DeviceUid(pub NamedId);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AwgKey(pub u64);

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub enum DeviceKind {
    Hdawg,
    Shfqa,
    Shfsg,
    Uhfqa,
    PrettyPrinterDevice,
}
