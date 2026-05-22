// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::setup_description_qccs::SetupDescriptionQccs;
use crate::setup_description_zqcs::SetupDescriptionZqcs;

#[derive(Debug, Clone, PartialEq)]
pub enum SetupDescription {
    Qccs(SetupDescriptionQccs),
    Zqcs(SetupDescriptionZqcs),
}
