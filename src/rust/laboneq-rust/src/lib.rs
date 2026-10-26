// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod compiled_experiment;
mod compiler;
mod utils;

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

    // Register backend name constant
    m.add("BACKEND_NAME", laboneq_qccs_backend::BACKEND_NAME)?;

    let codegenerator = codegenerator_py::create_py_module(m.py(), "codegenerator")?;
    modules.set_item("laboneq._rust.codegenerator", &codegenerator)?;
    m.add_submodule(&codegenerator)?;

    let device_setup = utils::create_py_module(m.py(), "utils")?;
    modules.set_item("laboneq._rust.utils", &device_setup)?;
    m.add_submodule(&device_setup)?;

    let compiler = laboneq_compiler_py::create_py_module(m.py(), "compiler")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    compiler.add_function(wrap_pyfunction!(
        compiler::compile_experiment_py,
        &compiler
    )?)?;
    compiler.add_function(wrap_pyfunction!(
        compiler::serialize_experiment_py,
        &compiler
    )?)?;
    modules.set_item("laboneq._rust.compiler", &compiler)?;
    m.add_submodule(&compiler)?;

    let compiled_experiment = compiled_experiment::create_py_module(py, "compiled_experiment")?;
    modules.set_item("laboneq._rust.compiled_experiment", &compiled_experiment)?;
    m.add_submodule(&compiled_experiment)?;

    // The feature is internal-only; the public tree parses OpenQASM in Python.
    #[cfg(feature = "openqasm3")]
    {
        let openqasm3 = PyModule::new(py, "openqasm3")?;
        openqasm3_py::register(&openqasm3, "laboneq._rust.openqasm3")?;
        modules.set_item("laboneq._rust.openqasm3", &openqasm3)?;
        m.add_submodule(&openqasm3)?;
    }

    Ok(())
}
