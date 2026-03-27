// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

fn main() {
    let schema_dir = "../../../schemas";
    let schema_prefix = "pulse/v1";

    let schema_names = [
        "calibration",
        "common",
        "device_setup",
        "experiment",
        "operation",
        "pulse",
        "section",
        "sweep",
    ];

    let mut cmd = capnpc::CompilerCommand::new();
    cmd.default_parent_module(vec!["pulse".into(), "v1".into()])
        .src_prefix(schema_dir)
        .import_path(schema_dir);

    for name in &schema_names {
        let path = format!("{schema_dir}/{schema_prefix}/{name}.capnp");
        println!("cargo:rerun-if-changed={path}");
        cmd.file(&path);
    }

    cmd.run().expect("Cap'n Proto schema compilation failed");
}
