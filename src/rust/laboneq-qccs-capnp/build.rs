// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_capnp_build::SchemaCompiler;

fn main() {
    let compiler = SchemaCompiler {
        schema_dir: "../../../schemas/qccs",
        ..Default::default()
    };

    compiler.compile("payload/v1", &["payload", "v1"], &["payload"]);
    compiler.compile("setup/v1", &["setup", "v1"], &["setup_description"]);
}
