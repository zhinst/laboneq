// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use pyo3::{IntoPyObjectExt, intern, prelude::*, types::PyDict};

use laboneq_common::{named_id::NamedIdStore, types::Literal};
use laboneq_dsl::operation::{ExternalOrValue, ValueEntry};
use laboneq_dsl::types::{ExternalParameterUid, ValueOrParameter};
use laboneq_py_utils::py_export::{numeric_literal_to_py, value_to_py};
use laboneq_py_utils::py_object_interner::PyObjectInterner;
use numeric_array::NumericArray;

use crate::execution::Statement;

/// Creates a Python execution from a list of statements.
///
/// The type of the returned Python object is `laboneq.executor.executor.Statement`.
pub(crate) fn create_py_execution<'py>(
    py: Python<'py>,
    statements: Vec<Statement>,
    id_store: &NamedIdStore,
    py_objects: &PyObjectInterner<ExternalParameterUid>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut builder = PyStatementBuilder {
        py,
        id_store,
        py_objects,
    };

    let mut out = Vec::with_capacity(statements.len());
    for statement in statements {
        let py_stmt = match statement {
            Statement::SetParameter { parameter, values } => builder.handle_set_parameter(
                id_store
                    .resolve(parameter)
                    .expect("Internal Error: Parameter ID not found"),
                &values,
            )?,
            Statement::Loop { count, body } => {
                let body = create_py_execution(py, body, id_store, py_objects)?;
                builder.handle_loop(count, body)?
            }
            Statement::ExecRealTime => builder.handle_exec_realtime()?,
            Statement::ExecCallback { callback_id, args } => builder.handle_exec_neartime(
                id_store
                    .resolve(callback_id)
                    .expect("Internal Error: Callback ID not found"),
                &args,
            )?,
            Statement::SetNode { path, value } => builder.handle_set_node(
                id_store
                    .resolve(path)
                    .expect("Internal Error: Path ID not found"),
                &value,
            )?,
        };
        out.push(py_stmt);
    }
    builder.create_sequence(out)
}

struct PyStatementBuilder<'py, 'a> {
    py: Python<'py>,
    id_store: &'a NamedIdStore,
    py_objects: &'a PyObjectInterner<ExternalParameterUid>,
}

impl<'py> PyStatementBuilder<'py, '_> {
    fn executor_module(&self) -> PyResult<Bound<'py, PyModule>> {
        self.py
            .import(intern!(self.py, "laboneq.executor.executor"))
    }

    fn handle_exec_neartime(
        &mut self,
        callback_id: &str,
        args: &[ValueEntry],
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;

        let dict = PyDict::new(self.py);
        for arg in args {
            let key = self
                .id_store
                .resolve(arg.key)
                .expect("Internal Error: Arg key not found");
            match &arg.value {
                ExternalOrValue::ExternalParameter(uid) => {
                    let py_obj = self
                        .py_objects
                        .resolve(uid)
                        .expect("Internal Error: External parameter not found");
                    dict.set_item(key, py_obj)?;
                }
                ExternalOrValue::ValueOrParameter(vop) => {
                    let py_val: Py<PyAny> = match vop {
                        ValueOrParameter::Value(v) => numeric_literal_to_py(self.py, v)?.unbind(),
                        ValueOrParameter::Parameter(uid) => {
                            let param_uid = self
                                .id_store
                                .resolve(*uid)
                                .expect("Internal Error: Parameter ID not found");

                            let uid_reference = module.getattr(intern!(self.py, "UIDReference"))?;
                            uid_reference.call1((param_uid,))?.unbind()
                        }
                        _ => panic!(
                            "Internal Error: Expected ValueOrParameter to be either Value, or Parameter"
                        ),
                    };
                    dict.set_item(key, py_val)?;
                }
            }
        }

        let exec_callback_class = module.getattr(intern!(self.py, "ExecNeartimeCall"))?;

        let kwargs = PyDict::new(self.py);
        kwargs.set_item("func_name", callback_id)?;
        kwargs.set_item("args", dict)?;
        let py_obj = exec_callback_class.call((), Some(&kwargs))?;
        Ok(py_obj)
    }

    fn handle_exec_realtime(&mut self) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;
        let exec_realtime_class = module.getattr(intern!(self.py, "ExecRT"))?;
        let py_obj = exec_realtime_class.call1((
            self.py.None(),
            self.create_sequence(vec![])?,
            self.py.None(),
            self.py.None(),
            self.py.None(),
        ))?;
        Ok(py_obj)
    }

    fn handle_set_node(
        &mut self,
        path: &str,
        value: &ValueOrParameter<Literal>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;
        let set_node_class = module.getattr(intern!(self.py, "ExecSet"))?;

        let value = match value {
            ValueOrParameter::Value(v) => value_to_py(self.py, v),
            ValueOrParameter::Parameter(uid) => {
                let param_uid = self
                    .id_store
                    .resolve(*uid)
                    .expect("Internal Error: Parameter ID not found");
                param_uid.into_bound_py_any(self.py)
            }
            _ => {
                panic!("Internal Error: Expected ValueOrParameter to be either Value, or Parameter")
            }
        }?;

        let py_obj = set_node_class.call1((path, value))?;
        Ok(py_obj)
    }

    fn handle_loop(
        &mut self,
        count: NonZeroU32,
        body: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;

        let for_loop_class = module.getattr(intern!(self.py, "ForLoop"))?;
        let py_obj = for_loop_class.call1((count.get(), body))?;
        Ok(py_obj)
    }

    fn handle_set_parameter(
        &mut self,
        name: &str,
        values: &NumericArray,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;
        let set_parameter_class = module.getattr(intern!(self.py, "SetSoftwareParam"))?;
        let values = values.to_py(self.py)?;

        let kwargs = PyDict::new(self.py);
        kwargs.set_item("name", name)?;
        kwargs.set_item("values", values)?;
        kwargs.set_item("axis_name", self.py.None())?;

        let py_obj = set_parameter_class.call((), Some(&kwargs))?;
        Ok(py_obj)
    }

    fn create_sequence(&mut self, body: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let module = self.executor_module()?;
        let sequence_class = module.getattr(intern!(self.py, "Sequence"))?;
        let py_obj = sequence_class.call1((body,))?;
        Ok(py_obj)
    }
}
