// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::AwgKey;
use smallvec::SmallVec;

#[derive(Debug, Clone, PartialEq)]
pub struct AwgCore {
    uid: AwgKey,
    number: SmallVec<[u16; 4]>,
}

impl AwgCore {
    pub fn new(uid: AwgKey, number: SmallVec<[u16; 4]>) -> Self {
        Self { uid, number }
    }

    /// Returns the unique identifier of the AWG.
    pub fn uid(&self) -> AwgKey {
        self.uid
    }

    /// AWG number on a device it is part of.
    pub fn number(&self) -> &SmallVec<[u16; 4]> {
        &self.number
    }
}
