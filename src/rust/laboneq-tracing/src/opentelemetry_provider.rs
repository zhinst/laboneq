// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use crate::opentelemetry_in_memory_exporter::InMemorySpanExporter;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExporterType {
    OtlpHttp,
    InMemory,
}

/// Create an OpenTelemetry tracer.
///
/// The tracer is configured with the provided OTLP endpoint and/or in-memory exporter.
/// If both are `None`, the function will panic.
///
/// Opentelemetry is configured to use batch processing of spans, which can be configured
/// by using the dedicated set of environment variables provided by the OpenTelemetry SDK.
pub(crate) fn create_tracer(exporters: &[ExporterType]) -> Option<SdkTracerProvider> {
    if exporters.is_empty() {
        return None;
    }
    let mut builder = SdkTracerProvider::builder();
    for exporter in exporters {
        match exporter {
            ExporterType::OtlpHttp => {
                let exporter = SpanExporter::builder()
                    .with_http()
                    .build()
                    .expect("Failed to create trace exporter");
                builder = builder.with_batch_exporter(exporter);
            }
            ExporterType::InMemory => {
                let exporter = InMemorySpanExporter::new();
                builder = builder.with_batch_exporter(exporter);
            }
        }
    }

    let tracer_provider = builder.with_resource(get_resource());
    Some(tracer_provider.build())
}

fn get_resource() -> Resource {
    static RESOURCE: OnceLock<Resource> = OnceLock::new();
    RESOURCE
        .get_or_init(|| {
            Resource::builder()
                .with_service_name("laboneq-rust")
                .build()
        })
        .clone()
}
