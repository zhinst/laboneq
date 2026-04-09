// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides tracing utilities for LabOneQ Rust code, including integration with OpenTelemetry.
//!
//! The main entry point is the `with_tracing` function, which configures tracing for the duration of a provided closure.
//! Tracing can be enabled and configured via environment variables, allowing for flexible integration with different tracing backends and exporters.
//! The crate currently supports exporting spans to an OTLP collector over HTTP, as well as buffering spans in memory and exporting them as JSON.
//!
//! The tracing configuration is controlled via environment variables:
//!
//! - `LABONEQ_TRACING_ENABLE`: If set to "1", enables tracing. Otherwise, tracing is disabled by default.
//! - `LABONEQ_TRACING_IN_MEMORY_EXPORTER`: If set to "1", enables the in-memory exporter. This is disabled by default.
//! - OTLP HTTP exporter can be configured via standard OpenTelemetry environment variables, such as `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

pub mod opentelemetry_in_memory_exporter;
mod opentelemetry_provider;

use std::env;
use tracing::{Dispatch, dispatcher};

use opentelemetry::trace::TracerProvider;
use opentelemetry_provider::ExporterType;
use tracing_subscriber::Registry;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;

/// Configuration
/// Environment variables:
///
/// - `LABONEQ_TRACING_ENABLE`: If set to "1" or "true" (case-insensitive), enables tracing. Otherwise, tracing is disabled by default.
/// - `LABONEQ_TRACING_IN_MEMORY_EXPORTER`: If set to "1" or "true" (case-insensitive), enables the in-memory exporter. This is disabled by default.
const ENV_VAR_ENABLE: &str = "LABONEQ_TRACING_ENABLE";
const ENV_VAR_IN_MEMORY_EXPORTER: &str = "LABONEQ_TRACING_IN_MEMORY_EXPORTER";

/// Checks if tracing is enabled by reading the `LABONEQ_TRACING_ENABLE` environment variable.
pub fn tracing_is_enabled() -> bool {
    env_var_is_true(ENV_VAR_ENABLE)
}

fn env_var_is_true(var_name: &str) -> bool {
    env::var(var_name)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Configures tracing for the duration of the provided closure.
///
/// This function is meant to be called once at the beginning of the program execution,
/// wrapping the main logic of the program.
pub fn with_tracing<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let Some(settings) = build_settings() else {
        return f();
    };
    // We create a new dispatcher for each call to `with_tracing` to ensure that the tracing configuration is applied correctly.
    let state = init_dispatcher(settings.clone());
    let r = dispatcher::with_default(&state.dispatch, f);
    // Currently we must force flush the dispatcher manually to ensure the traces are written.
    // TODO: This means that between every call to `with_tracing` a new tracer provider is created.
    // Find a way to reuse the same tracer provider across multiple calls, or avoid multiple calls to `with_tracing`.
    if let Some(t) = &state._otel
        && settings.force_flush
    {
        let _ = t.force_flush();
    }
    r
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TracingConfig {
    opentelemetry_exporters: Vec<ExporterType>,
    /// Whether to force flush the tracer provider at the end of the scope.
    force_flush: bool,
}

fn build_settings() -> Option<TracingConfig> {
    if !tracing_is_enabled() {
        return None;
    }
    // Currently we always enable the OTLP exporter if tracing is enabled,
    // unless an in-memory exporter is explicitly configured.
    // For now we do not support configuring both exporters at the same time.
    let config = if env_var_is_true(ENV_VAR_IN_MEMORY_EXPORTER) {
        TracingConfig {
            opentelemetry_exporters: vec![ExporterType::InMemory],
            force_flush: true,
        }
    } else {
        TracingConfig {
            opentelemetry_exporters: vec![ExporterType::OtlpHttp],
            force_flush: false,
        }
    };
    Some(config)
}

struct TracingState {
    dispatch: Dispatch,
    _otel: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl Drop for TracingState {
    fn drop(&mut self) {
        if let Some(otel) = &self._otel {
            let _ = otel.force_flush();
        }
    }
}

fn init_dispatcher(config: TracingConfig) -> TracingState {
    let otel_tracer: opentelemetry_sdk::trace::SdkTracerProvider =
        opentelemetry_provider::create_tracer(&config.opentelemetry_exporters).unwrap();
    // Each layer must be boxed to be type erased.
    let otel_layer = tracing_opentelemetry::layer().with_tracer(otel_tracer.tracer("laboneq-rust"));
    // To add more layers, extend the `with` chain. Make sure each layer is boxed and optional.
    // TODO: Add log interoperability layer
    let registry = Registry::default().with(otel_layer);
    let dispatch = Dispatch::new(registry);
    TracingState {
        dispatch,
        _otel: Some(otel_tracer),
    }
}
