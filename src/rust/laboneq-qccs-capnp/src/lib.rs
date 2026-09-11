// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Generated Cap'n Proto bindings for LabOne Q QCCS schemas.

#![allow(unused_qualifications)]
#![allow(unreachable_pub)]

pub mod payload {
    pub mod v1 {
        capnp::generated_code!(
            pub mod payload_capnp,
            "payload/v1/payload_capnp.rs"
        );
    }
}

pub mod setup {
    pub mod v1 {
        capnp::generated_code!(
            pub mod setup_description_capnp,
            "setup/v1/setup_description_capnp.rs"
        );
    }
}
