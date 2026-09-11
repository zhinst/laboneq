// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Deserializes `CompiledExperiment.execution` (a flat list of instructions, with `Loop`s
//! represented as `loopStart`/`loopEnd` marker pairs) directly into
//! `laboneq.executor.executor` Python objects.

use capnp::struct_list;
use pyo3::IntoPyObjectExt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use laboneq_capnp::compiled_experiment::v1::execution_capnp;

use laboneq_error::LabOneQError;

use crate::Result;
use crate::context::{DecodedValue, DeserializerContext, capnp_err, decode_value, text_to_string};

fn executor_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    py.import(intern!(py, "laboneq.executor.executor"))
}

/// Builds the `laboneq.executor.executor.Sequence` for `CompiledExperiment.execution`.
pub(crate) fn build_execution<'py>(
    py: Python<'py>,
    instructions: struct_list::Reader<'_, execution_capnp::instruction::Owned>,
    ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>> {
    let mut cursor = 0u32;
    let statements = build_block(py, &instructions, &mut cursor, ctx)?;
    Ok(create_sequence(py, statements)?)
}

/// Consumes instructions from `*cursor` until either the list ends (top-level call) or a
/// matching `loopEnd` is found (nested call from a `loopStart`), building each statement's
/// Python object along the way.
fn build_block<'py>(
    py: Python<'py>,
    instructions: &struct_list::Reader<'_, execution_capnp::instruction::Owned>,
    cursor: &mut u32,
    ctx: &DeserializerContext,
) -> Result<Vec<Bound<'py, PyAny>>> {
    use execution_capnp::instruction::Which;

    let mut out = Vec::new();
    while *cursor < instructions.len() {
        let index = *cursor;
        let instr = instructions.get(index);
        let statement = match instr.which().map_err(capnp_err)? {
            Which::LoopEnd(()) => {
                *cursor += 1;
                return Ok(out);
            }
            Which::LoopStart(ls) => {
                let count = ls.map_err(capnp_err)?.get_count();
                *cursor += 1;
                let body = build_block(py, instructions, cursor, ctx)?;
                let body_seq = create_sequence(py, body)?;
                build_for_loop(py, count, body_seq).map_err(LabOneQError::from)
            }
            Which::SetParameter(sp) => {
                let idx = sp.map_err(capnp_err)?.get_parameter_ref();
                *cursor += 1;
                build_set_parameter(py, idx, ctx)
            }
            Which::ExecRealTime(_) => {
                *cursor += 1;
                build_exec_realtime(py).map_err(LabOneQError::from)
            }
            Which::ExecCallback(cb) => {
                let cb = cb.map_err(capnp_err)?;
                *cursor += 1;
                build_exec_callback(py, cb, ctx)
            }
            Which::SetNode(sn) => {
                let sn = sn.map_err(capnp_err)?;
                *cursor += 1;
                build_set_node(py, sn, ctx)
            }
        };
        out.push(statement?);
    }
    Ok(out)
}

fn create_sequence<'py>(
    py: Python<'py>,
    body: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let sequence_class = executor_module(py)?.getattr(intern!(py, "Sequence"))?;
    sequence_class.call1((body,))
}

fn build_for_loop<'py>(
    py: Python<'py>,
    count: u32,
    body: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let for_loop_class = executor_module(py)?.getattr(intern!(py, "ForLoop"))?;
    for_loop_class.call1((count, body))
}

fn build_exec_realtime<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let module = executor_module(py)?;
    let exec_realtime_class = module.getattr(intern!(py, "ExecRT"))?;
    let empty_body = create_sequence(py, vec![])?;
    exec_realtime_class.call1((py.None(), empty_body, py.None(), py.None(), py.None()))
}

fn build_set_parameter<'py>(
    py: Python<'py>,
    idx: u32,
    ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>> {
    let module = executor_module(py)?;
    let set_parameter_class = module.getattr(intern!(py, "SetSoftwareParam"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", ctx.parameter_name(idx)?)?;
    kwargs.set_item("values", ctx.parameter_array(py, idx)?)?;
    kwargs.set_item("axis_name", py.None())?;
    Ok(set_parameter_class.call((), Some(&kwargs))?)
}

fn build_exec_callback<'py>(
    py: Python<'py>,
    reader: execution_capnp::exec_callback::Reader<'_>,
    ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>> {
    let module = executor_module(py)?;
    let callback_id = text_to_string(reader.get_callback_id().map_err(capnp_err)?)?;

    let dict = PyDict::new(py);
    for entry in reader.get_args().map_err(capnp_err)?.iter() {
        let key = text_to_string(entry.get_key().map_err(capnp_err)?)?;
        let value_reader = entry.get_value().map_err(capnp_err)?;
        let py_val: Py<PyAny> = match decode_value(py, value_reader)? {
            DecodedValue::None => py.None(),
            DecodedValue::Constant(obj) => obj,
            DecodedValue::ParameterRef(idx) => {
                let uid_reference_class = module.getattr(intern!(py, "UIDReference"))?;
                uid_reference_class
                    .call1((ctx.parameter_name(idx)?,))?
                    .unbind()
            }
        };
        dict.set_item(key, py_val)?;
    }

    let exec_callback_class = module.getattr(intern!(py, "ExecNeartimeCall"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("func_name", callback_id)?;
    kwargs.set_item("args", dict)?;
    Ok(exec_callback_class.call((), Some(&kwargs))?)
}

fn build_set_node<'py>(
    py: Python<'py>,
    reader: execution_capnp::set_node::Reader<'_>,
    ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>> {
    let module = executor_module(py)?;
    let set_node_class = module.getattr(intern!(py, "ExecSet"))?;
    let path = text_to_string(reader.get_path().map_err(capnp_err)?)?;

    let value_reader = reader.get_value().map_err(capnp_err)?;
    let value: Py<PyAny> = match decode_value(py, value_reader)? {
        DecodedValue::None => py.None(),
        DecodedValue::Constant(obj) => obj,
        DecodedValue::ParameterRef(idx) => ctx.parameter_name(idx)?.into_py_any(py)?,
    };

    Ok(set_node_class.call1((path, value))?)
}
