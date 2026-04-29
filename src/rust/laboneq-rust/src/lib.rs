// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod compiler;

#[pymodule]
#[pyo3(name = "_rust")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    // To enable import from submodules, we must make submodules a package,
    // not only a module. To avoid having to create several Rust extensions,
    // we register the module through sys.modules at runtime, which has the
    // same effect.
    // See:
    // - https://github.com/PyO3/pyo3/issues/759
    // - https://docs.python.org/3/tutorial/modules.html#packages
    let modules = py.import("sys")?.getattr("modules")?;

    let codegenerator = codegenerator_py::create_py_module(m.py(), "codegenerator")?;
    modules.set_item("laboneq._rust.codegenerator", &codegenerator)?;
    m.add_submodule(&codegenerator)?;

    let compiler = laboneq_compiler_py::create_py_module(m.py(), "compiler")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    compiler.add_function(wrap_pyfunction!(
        compiler::build_experiment_capnp_py,
        &compiler
    )?)?;
    modules.set_item("laboneq._rust.compiler", &compiler)?;
    m.add_submodule(&compiler)?;
    Ok(())
}
