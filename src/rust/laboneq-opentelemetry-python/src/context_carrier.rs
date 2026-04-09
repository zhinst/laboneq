// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides a function to attach an active OpenTelemetry context from Python to the Rust context.
//! This allows Rust code called from Python to continue a trace started in Python.
use std::collections::HashMap;

use opentelemetry::{Context, propagation::Extractor};
use pyo3::{prelude::*, types::IntoPyDict};

/// Get the current OpenTelemetry context from Python and attach it to the current Rust context.
/// This should be called at the beginning of a function or method that is called from Python and
/// that should continue a trace started in Python.
///
/// Returns `None` if the context cannot be evaluated, e.g., because OpenTelemetry is not installed in Python.
pub fn attach_otel_context(py: Python<'_>) -> PyResult<Option<opentelemetry::ContextGuard>> {
    let opentelemetry_context = py.import("opentelemetry.context");
    if opentelemetry_context.is_err() {
        // OpenTelemetry is not installed in Python
        return Ok(None);
    }
    let get_current_context = opentelemetry_context?.getattr("get_current")?;
    let inject = py.import("opentelemetry.propagate")?.getattr("inject")?;

    let current_context = get_current_context.call0()?;
    let data = pyo3::types::PyDict::new(py);
    let kwargs = [
        ("context", current_context),
        ("carrier", data.as_any().clone()),
    ]
    .into_py_dict(py)?;
    inject.call((), Some(&kwargs))?;
    let carrier = kwargs.get_item("carrier")?.unwrap();
    let data: HashMap<String, String> = carrier.extract()?;
    let carrier: SpanContext = data.into();
    Ok(Some(carrier.attach()))
}

struct SpanContext {
    traceparent: Option<String>,
    tracestate: Option<String>,
}

impl Extractor for SpanContext {
    fn get(&self, key: &str) -> Option<&str> {
        match key.to_lowercase().as_str() {
            "traceparent" => self.traceparent.as_deref(),
            "tracestate" => self.tracestate.as_deref(),
            _ => None,
        }
    }

    fn keys(&self) -> Vec<&str> {
        vec!["traceparent", "tracestate"]
    }
}

impl From<HashMap<String, String>> for SpanContext {
    fn from(value: HashMap<String, String>) -> Self {
        Self {
            tracestate: value.get("tracestate").cloned(),
            traceparent: value.get("traceparent").cloned(),
        }
    }
}

impl SpanContext {
    fn attach(&self) -> opentelemetry::ContextGuard {
        use opentelemetry::propagation::TextMapPropagator;

        let propagator = opentelemetry_sdk::propagation::TraceContextPropagator::new();
        Context::attach(propagator.extract(self))
    }
}
