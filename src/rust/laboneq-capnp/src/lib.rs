// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Generated Cap'n Proto bindings for LabOne Q pulse-level schemas.

#![allow(unused_qualifications)]
#![allow(unreachable_pub)]

pub mod pulse {
    pub mod v1 {
        capnp::generated_code!(pub mod calibration_capnp, "pulse/v1/calibration_capnp.rs");
        capnp::generated_code!(pub mod common_capnp, "pulse/v1/common_capnp.rs");
        capnp::generated_code!(pub mod device_setup_capnp, "pulse/v1/device_setup_capnp.rs");
        capnp::generated_code!(pub mod experiment_capnp, "pulse/v1/experiment_capnp.rs");
        capnp::generated_code!(pub mod operation_capnp, "pulse/v1/operation_capnp.rs");

        capnp::generated_code!(pub mod pulse_capnp, "pulse/v1/pulse_capnp.rs");
        capnp::generated_code!(pub mod section_capnp, "pulse/v1/section_capnp.rs");
        capnp::generated_code!(pub mod sweep_capnp, "pulse/v1/sweep_capnp.rs");
    }
}
