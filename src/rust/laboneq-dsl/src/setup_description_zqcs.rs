// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::DeviceUid;

#[derive(Debug, Clone, PartialEq)]
pub struct SetupDescriptionZqcs {
    pub data: Vec<u8>,
    pub device_uid: DeviceUid,
    pub channels: Vec<ZqcsChannel>,
}

impl SetupDescriptionZqcs {
    pub fn new(data: Vec<u8>, device_uid: DeviceUid, channels: Vec<ZqcsChannel>) -> Self {
        SetupDescriptionZqcs {
            data,
            device_uid,
            channels,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZqcsChannel {
    pub geolocation: String,
    pub channel_type: ChannelType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelType {
    Rf,
    Qa,
    Flux,
}
