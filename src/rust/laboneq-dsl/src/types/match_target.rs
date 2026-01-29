// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::{HandleUid, ParameterUid, SectionUid, UserRegister};

#[derive(Debug, Clone, PartialEq)]
pub enum MatchTarget {
    Handle(HandleUid),
    UserRegister(UserRegister),
    /// PRNG Loop UID
    PrngSample(SectionUid),
    SweepParameter(ParameterUid),
}

impl MatchTarget {
    pub fn description(&self) -> &'static str {
        match self {
            MatchTarget::Handle(_) => "acquisition handle",
            MatchTarget::UserRegister(_) => "user register",
            MatchTarget::PrngSample(_) => "PRNG sample",
            MatchTarget::SweepParameter(_) => "sweep parameter",
        }
    }
}
