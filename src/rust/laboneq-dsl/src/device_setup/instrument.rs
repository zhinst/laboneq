// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::DeviceUid;
use laboneq_common::types::AuxiliaryDeviceKind;

/// Auxiliary devices used in the experiment, which do not have signals but are still relevant for the setup.
#[derive(Debug, Clone, PartialEq)]
pub struct AuxiliaryDevice {
    uid: DeviceUid,
    kind: AuxiliaryDeviceKind,
}

impl AuxiliaryDevice {
    pub fn new(uid: DeviceUid, kind: AuxiliaryDeviceKind) -> Self {
        Self { uid, kind }
    }

    pub fn uid(&self) -> DeviceUid {
        self.uid
    }

    pub fn kind(&self) -> AuxiliaryDeviceKind {
        self.kind
    }
}
