# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager

import orjson

# Check if OpenTelemetry SDK is available
_OTEL_SDK_AVAILABLE = False
try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _OTEL_SDK_AVAILABLE = True
except ImportError:

    class ReadableSpan:
        pass

    class InMemorySpanExporter:
        pass


@contextmanager
def laboneq_tracing(span_exporter: InMemorySpanExporter):
    """Tracing context manager for LabOne Q.

    The context manager sets up the tracing required for LabOne Q Rust components.
    and exports the spans to the provided span exporter.

    This function supports currently only `InMemorySpanExporter`.

    For other OTLP exporter types, the LabOne Q Rust component tracing can be enable with environmental variable
    `LABONEQ_TRACING_ENABLE=1`. Otherwise standard OpenTelemetry environmental variable configuration
    applies. For example `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
    """

    if not _OTEL_SDK_AVAILABLE:
        raise ImportError(
            "OpenTelemetry SDK is required for `laboneq_tracing()`. Please install opentelemetry-sdk."
        )
    from laboneq._rust.compiler import SpanBuffer

    streamer = SpanBuffer()
    try:
        with _temporary_env_vars(
            LABONEQ_TRACING_ENABLE="1", LABONEQ_TRACING_IN_MEMORY_EXPORTER="1"
        ):
            yield
    finally:
        for span in streamer.flush_spans():
            span_exporter.export([_json_trace_to_span(orjson.loads(span))])


def _json_trace_to_span(span: dict) -> ReadableSpan:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext

    attrs = {k: v for d in span["attributes"] for k, v in d.items()}
    return ReadableSpan(
        name=span["name"],
        context=SpanContext(
            trace_id=int(span["context"]["trace_id"], 16),
            span_id=int(span["context"]["span_id"], 16),
            is_remote=False,
            trace_flags=0,
            trace_state=None,
        ),
        parent=SpanContext(
            trace_id=int(span["context"]["trace_id"], 16),
            span_id=int(span["parent_id"], 16),
            is_remote=False,
            trace_flags=0,
            trace_state=None,
        ),
        attributes=attrs,
        start_time=span["start_time"],
        end_time=span["end_time"],
    )


@contextmanager
def _temporary_env_vars(**kwargs):
    """Context manager to temporarily set environment variables."""
    prev_env = dict(os.environ)
    try:
        os.environ.update(kwargs)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prev_env)
