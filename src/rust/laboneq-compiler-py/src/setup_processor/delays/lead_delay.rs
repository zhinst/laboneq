// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::DeviceKind;
use laboneq_units::duration::{Duration, Second};

pub(super) fn get_lead_delay(
    device_type: &DeviceKind,
    sampling_rate: f64,
    is_desktop_setup: bool,
) -> Duration<Second> {
    use laboneq_common::device_traits::{
        DEFAULT_HDAWG_LEAD_DESKTOP_SETUP, DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ,
        DEFAULT_HDAWG_LEAD_PQSC, DEFAULT_HDAWG_LEAD_PQSC_2GHZ, DEFAULT_SHFQA_LEAD_PQSC,
        DEFAULT_SHFSG_LEAD_PQSC, DEFAULT_TESTDEVICE_LEAD, DEFAULT_UHFQA_LEAD_PQSC,
    };

    match device_type {
        DeviceKind::Hdawg => {
            let hdawg_uses_2ghz = sampling_rate == 2e9;
            if !is_desktop_setup {
                if hdawg_uses_2ghz {
                    DEFAULT_HDAWG_LEAD_PQSC_2GHZ
                } else {
                    DEFAULT_HDAWG_LEAD_PQSC
                }
            } else if hdawg_uses_2ghz {
                DEFAULT_HDAWG_LEAD_DESKTOP_SETUP_2GHZ
            } else {
                DEFAULT_HDAWG_LEAD_DESKTOP_SETUP
            }
        }
        DeviceKind::Zqcs => DEFAULT_TESTDEVICE_LEAD,
        DeviceKind::Uhfqa => DEFAULT_UHFQA_LEAD_PQSC,
        DeviceKind::Shfqa => DEFAULT_SHFQA_LEAD_PQSC,
        DeviceKind::Shfsg => DEFAULT_SHFSG_LEAD_PQSC,
    }
}
