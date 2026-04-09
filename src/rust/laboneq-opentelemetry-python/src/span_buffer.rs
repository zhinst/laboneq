// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use std::time::UNIX_EPOCH;

use opentelemetry_sdk::trace::SpanData;
use serde_json::json;

/// Python wrapper for the in-memory span buffer, allowing retrieval of spans as JSON strings.
#[pyclass(name = "SpanBuffer")]
pub struct SpanBufferPy {}

#[pymethods]
impl SpanBufferPy {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn flush_spans(&self, py: Python) -> PyResult<Vec<String>> {
        Python::detach(py, || {
            if let Some(store) =
                laboneq_tracing::opentelemetry_in_memory_exporter::SpanStore::get_global_store()
            {
                if let Ok(mut guard) = store.lock() {
                    let spans: Vec<String> = guard
                        .drain_spans()
                        .map(|span_data| {
                            let span_json = span_to_json(&span_data).unwrap();
                            serde_json::to_string(&span_json).unwrap()
                        })
                        .collect();
                    Ok(spans)
                } else {
                    Ok(Vec::new())
                }
            } else {
                Ok(Vec::new())
            }
        })
    }
}

/// Convert SpanData to a JSON value, including converting timestamps to nanoseconds since UNIX epoch.
///
/// The output format is designed to be easily consumed by Python code, with timestamps as integers and attributes as a JSON object.
/// It is not officially defined as a standard format, but is structured to be convenient for our use case of exporting spans to Python and
/// is close to the Python OpenTelemetry SDK's span representation.
fn span_to_json(span: &SpanData) -> Result<serde_json::Value, String> {
    let (start_time, end_time) = (
        span.start_time
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos(),
        span.end_time
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos(),
    );
    let attributes: serde_json::Value = span
        .attributes
        .iter()
        .map(|kv| {
            json!({
                kv.key.as_str(): kv.value.as_str(),
            })
        })
        .collect();
    Ok(json!({
        "context": {
            "trace_id": span.span_context.trace_id().to_string(),
            "span_id": span.span_context.span_id().to_string(),
            "trace_flags": format!("{:?}", span.span_context.trace_flags()),
            "is_remote": span.span_context.is_remote(),
        },
        "parent_id": span.parent_span_id.to_string(),
        "name": span.name,
        "kind": format!("{:?}", span.span_kind),
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attributes,
        "status_code": format!("{:?}", span.status),
        "status_message": match &span.status {
            opentelemetry::trace::Status::Ok => "Ok",
            opentelemetry::trace::Status::Error { description } => description,
            opentelemetry::trace::Status::Unset => "Unset",
        },
    }))
}
