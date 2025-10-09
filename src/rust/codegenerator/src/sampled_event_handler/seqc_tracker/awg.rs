// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::compilation_job::{AwgKey, AwgKind, ChannelIndex, DeviceKind, TriggerMode};

#[derive(Clone)]
pub struct Awg {
    pub signal_kind: AwgKind,
    pub awg_key: AwgKey,
    pub device_kind: DeviceKind,
    pub play_channels: Vec<ChannelIndex>,
    pub sampling_rate: f64,
    pub shf_output_mute_min_duration: Option<f64>,
    pub trigger_mode: TriggerMode,
    pub is_reference_clock_internal: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct HwOscillator {
    pub uid: String,
    pub index: u16,
}
