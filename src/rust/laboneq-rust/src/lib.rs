// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

mod intervals;

#[pymodule]
mod _rust {

    use super::*;
    #[pymodule(submodule)]
    mod intervals {
        #[pymodule_export]
        use crate::intervals::Interval;

        #[pymodule_export]
        use crate::intervals::IntervalTree;
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        // To enable import from submodules, we must make submodules a package,
        // not only a module. To avoid having to create several Rust extensions,
        // we register the module through sys.modules at runtime, which has the
        // same effect.
        // See:
        // - https://github.com/PyO3/pyo3/issues/759
        // - https://docs.python.org/3/tutorial/modules.html#packages
        let intervals_module = m.getattr("intervals")?;
        let modules = py.import_bound("sys")?.getattr("modules")?;
        modules.set_item("laboneq._rust.intervals", intervals_module)?;

        Ok(())
    }
}
