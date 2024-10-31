// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

use ir::IrNode;
use pyo3::prelude::*;

use crate::{loop_ir::LoopPy, section_ir::SectionPy};

pub fn unlock_mutex<T>(node: &Arc<Mutex<T>>) -> Result<MutexGuard<'_, T>, PyErr> {
    node.lock().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PySystemError, _>(
            "Unable to lock mutex of shared python reference.",
        )
    })
}

pub trait ExtractableIRNode {
    fn extract_ir_node(&self) -> Arc<Mutex<IrNode>>;
}

#[macro_export]
macro_rules! impl_extractable_ir_node {
    ($type:ty) => {
        impl ExtractableIRNode for $type {
            fn extract_ir_node(&self) -> Arc<Mutex<IrNode>> {
                Arc::clone(&self.0)
            }
        }
    };
}

#[derive(FromPyObject)]
enum IRNodePy {
    Section(SectionPy),
    Loop(LoopPy),
}

pub fn extract_child(item: &Bound<PyAny>) -> Result<Arc<Mutex<IrNode>>, PyErr> {
    let ir_node_py = item.extract::<IRNodePy>()?;

    match ir_node_py {
        IRNodePy::Section(node) => Ok(node.extract_ir_node()),
        IRNodePy::Loop(node) => Ok(node.extract_ir_node()),
    }
}

#[macro_export]
macro_rules! impl_interval_methods {
    ($type:ty, $ir_variant:ident) => {
        #[pymethods]
        impl $type {
            #[setter]
            pub fn set_interval(&mut self, interval: IntervalPy) -> PyResult<()> {
                let mut guard = unlock_mutex(&self.0)?;
                let IrNode::$ir_variant(ir_entity) = &mut *guard else {
                    panic!("Encountered wrapper with inconsistent internal type");
                };
                ir_entity.interval = interval.0.clone();
                Ok(())
            }

            #[getter]
            pub fn get_interval(&self) -> PyResult<IntervalPy> {
                let guard = unlock_mutex(&self.0)?;
                let IrNode::$ir_variant(ir_entity) = &*guard else {
                    panic!("Encountered wrapper with inconsistent internal type");
                };
                Ok(IntervalPy(ir_entity.interval.clone()))
            }
        }
    };
}

#[macro_export]
macro_rules! impl_python_dunders {
    ($type:ty, $ir_variant:ident) => {
        #[pymethods]
        impl $type {
            fn __repr__(&self) -> PyResult<String> {
                let guard = self.0.lock().map_err(|_| {
                    PyErr::new::<PySystemError, _>(
                        "Unable to lock mutex of shared python reference.",
                    )
                })?;
                Ok(format!("{:?}", guard))
            }

            fn __deepcopy__(&self, _memo: &Bound<PyDict>) -> PyResult<Self> {
                let guard = self.0.lock().map_err(|_| {
                    PyErr::new::<PySystemError, _>(
                        "Unable to lock mutex of shared python reference.",
                    )
                })?;
                let node_clone = deep_copy_ir_node(&guard).map_err(|_| {
                    PyErr::new::<PySystemError, _>(
                        "Unable to lock mutex of shared python reference.",
                    )
                })?;
                Ok(Self(Arc::new(Mutex::new(node_clone))))
            }

            fn __copy__(&self) -> Self {
                Self(self.0.clone())
            }
        }
    };
}
