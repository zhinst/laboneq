// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides tracing utilities for LabOneQ Rust code, including integration with OpenTelemetry.
//!
//! The main entry point is the `with_tracing` function, which configures tracing for the duration of a provided closure.
//! Tracing can be enabled and configured via environment variables, allowing for flexible integration with different tracing backends and exporters.
//! The crate currently supports exporting spans to an OTLP collector over HTTP, as well as buffering spans in memory and exporting them as JSON.
//!
//! - OTLP HTTP exporter can be configured via standard OpenTelemetry environment variables, such as `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

pub mod opentelemetry_in_memory_exporter;
mod opentelemetry_provider;

use std::env;
use tracing::{Dispatch, dispatcher};

use opentelemetry::trace::TracerProvider;
use opentelemetry_provider::ExporterType;
use tracing_subscriber::Registry;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;

/// Configuration
/// Environment variables:
///
/// - `LABONEQ_TRACING_ENABLE`: If set to "1" or "true" (case-insensitive), enables tracing. Otherwise, tracing is disabled by default.
/// - `LABONEQ_TRACING_IN_MEMORY_EXPORTER`: If set to "1" or "true" (case-insensitive), enables the in-memory exporter. This is disabled by default.
/// - `_LABONEQ_DEV_TRACING_LEVEL`: The verbosity of the captured spans, analogous to a log level. Possible values: "off", "error", "warn", "info",
///   "debug", "trace", or not set which is equivalent to "info".
///   This variable is deliberately not set by the user facing tracing helpers - it is supposed to be used as a debug tool by LabOne Q developers.
const ENV_VAR_ENABLE: &str = "LABONEQ_TRACING_ENABLE";
const ENV_VAR_IN_MEMORY_EXPORTER: &str = "LABONEQ_TRACING_IN_MEMORY_EXPORTER";
const ENV_VAR_LEVEL: &str = "_LABONEQ_DEV_TRACING_LEVEL";

const DEFAULT_LEVEL: LevelFilter = LevelFilter::INFO;

/// Checks if tracing is enabled by reading the `LABONEQ_TRACING_ENABLE` environment variable.
pub fn tracing_is_enabled() -> bool {
    env_var_is_true(ENV_VAR_ENABLE)
}

fn env_var_is_true(var_name: &str) -> bool {
    env::var(var_name)
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

fn tracing_level() -> LevelFilter {
    match env::var(ENV_VAR_LEVEL) {
        Ok(value) => {
            let value = value.trim();
            // An empty string parses to `LevelFilter::ERROR`, which is a surprising result for an
            // unset-but-present variable (e.g. `LABONEQ_TRACING_LEVEL=`), so treat it as the default.
            if value.is_empty() {
                return DEFAULT_LEVEL;
            }
            value.parse().unwrap_or(DEFAULT_LEVEL)
        }
        Err(_) => DEFAULT_LEVEL,
    }
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
    /// Maximum verbosity of the spans that are captured.
    level: LevelFilter,
}

fn build_settings() -> Option<TracingConfig> {
    if !tracing_is_enabled() {
        return None;
    }
    let level = tracing_level();
    // Currently we always enable the OTLP exporter if tracing is enabled,
    // unless an in-memory exporter is explicitly configured.
    // For now we do not support configuring both exporters at the same time.
    let config = if env_var_is_true(ENV_VAR_IN_MEMORY_EXPORTER) {
        TracingConfig {
            opentelemetry_exporters: vec![ExporterType::InMemory],
            force_flush: true,
            level,
        }
    } else {
        TracingConfig {
            opentelemetry_exporters: vec![ExporterType::OtlpHttp],
            force_flush: false,
            level,
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
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(otel_tracer.tracer("laboneq-rust"))
        .with_target(false)
        .with_threads(false)
        .with_tracked_inactivity(false)
        .with_location(false);
    // To add more layers, extend the `with` chain. Make sure each layer is boxed and optional.
    // TODO: Add log interoperability layer
    let registry = Registry::default().with(config.level).with(otel_layer);
    let dispatch = Dispatch::new(registry);
    TracingState {
        dispatch,
        _otel: Some(otel_tracer),
    }
}
