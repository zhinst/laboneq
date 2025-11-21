// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::{AwgKey, DeviceUid};

pub struct SingleFeedbackRegisterLayoutItem {
    pub width: u8,
    pub signal: Option<String>,
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum FeedbackRegister {
    Local { device: DeviceUid },
    Global { awg_key: AwgKey },
}

type SingleFeedbackRegisterLayout = Vec<SingleFeedbackRegisterLayoutItem>;
pub type FeedbackRegisterLayout =
    std::collections::HashMap<FeedbackRegister, SingleFeedbackRegisterLayout>;
