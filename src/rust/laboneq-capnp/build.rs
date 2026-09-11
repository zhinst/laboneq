// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

fn compile_schemas(
    schema_dir: &str,
    schema_prefix: &str,
    parent_module: Vec<String>,
    names: &[&str],
) {
    let mut cmd = capnpc::CompilerCommand::new();
    cmd.default_parent_module(parent_module)
        .src_prefix(schema_dir)
        .import_path(schema_dir);

    for name in names {
        let path = format!("{schema_dir}/{schema_prefix}/{name}.capnp");
        println!("cargo:rerun-if-changed={path}");
        cmd.file(&path);
    }

    cmd.run().expect("Cap'n Proto schema compilation failed");
}

fn main() {
    let schema_dir = "../../../schemas";

    compile_schemas(
        schema_dir,
        "pulse/v1",
        vec!["pulse".into(), "v1".into()],
        &[
            "calibration",
            "common",
            "device_setup",
            "coprocessor",
            "experiment",
            "operation",
            "pulse",
            "section",
            "sweep",
        ],
    );

    compile_schemas(
        schema_dir,
        "compiled_experiment/v1",
        vec!["compiled_experiment".into(), "v1".into()],
        &[
            "common",
            "compiled_experiment",
            "execution",
            "result_shape",
            "rt_loop_properties",
            "values",
            "pulse_sheet",
        ],
    );
}
