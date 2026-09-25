// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Build-script helper for compiling Cap'n Proto schemas into a crate.

#[derive(Default)]
pub struct SchemaCompiler<'a> {
    pub schema_dir: &'a str,
    pub import_paths: &'a [&'a str],
    pub crate_provides: &'a [(&'a str, &'a [u64])],
}

impl SchemaCompiler<'_> {
    pub fn compile(&self, schema_prefix: &str, parent_module: &[&str], names: &[&str]) {
        let mut cmd = capnpc::CompilerCommand::new();
        cmd.default_parent_module(parent_module.iter().map(|m| (*m).to_owned()).collect())
            .src_prefix(self.schema_dir)
            .import_path(self.schema_dir);

        for dir in self.import_paths {
            cmd.import_path(dir);
        }
        for (crate_name, ids) in self.crate_provides {
            cmd.crate_provides(*crate_name, ids.iter().copied());
        }
        for name in names {
            cmd.file(format!("{}/{schema_prefix}/{name}.capnp", self.schema_dir));
        }

        cmd.run().expect("Cap'n Proto schema compilation failed");

        for dir in std::iter::once(&self.schema_dir).chain(self.import_paths) {
            println!("cargo:rerun-if-changed={dir}");
        }
    }
}
