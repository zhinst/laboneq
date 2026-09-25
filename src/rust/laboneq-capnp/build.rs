// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_capnp_build::SchemaCompiler;

fn main() {
    let compiler = SchemaCompiler {
        schema_dir: "../../../schemas",
        ..Default::default()
    };

    compiler.compile(
        "pulse/v1",
        &["pulse", "v1"],
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

    compiler.compile(
        "compiled_experiment/v1",
        &["compiled_experiment", "v1"],
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
