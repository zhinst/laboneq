// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides an OpenTelemetry exporter that stores spans in memory for later retrieval.
//! The spans are stored as raw `SpanData` in a bounded buffer, and can be retrieved via the global `SpanStore`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::SpanData;
use opentelemetry_sdk::trace::SpanExporter;
use thiserror;

#[derive(thiserror::Error, Debug)]
pub enum SpanCollectionError {
    #[error("Internal error: OpenTelemetry span exporter buffer overflow - {0} spans dropped.")]
    BufferOverflow(usize),
}

/// The [`SpanStore`] is used to store spans in memory for later retrieval.
#[derive(Debug)]
pub struct SpanStore {
    spans: VecDeque<SpanData>, // Raw span data, serialize on demand
    max_size: usize,
    dropped: usize,
}

impl SpanStore {
    fn new(max_size: usize) -> Self {
        Self {
            spans: VecDeque::with_capacity(max_size),
            max_size,
            dropped: 0,
        }
    }

    /// Drain all spans from the store and return an iterator over the raw [`SpanData`].
    ///
    /// Returns an error if the store ran out of room while spans were being collected,
    /// rather than handing back a trace with an arbitrary subset of its spans missing.
    pub fn drain_spans(
        &mut self,
    ) -> Result<impl Iterator<Item = SpanData> + '_, SpanCollectionError> {
        let spans = self.spans.drain(..);
        let dropped = std::mem::take(&mut self.dropped);
        if dropped != 0 {
            Err(SpanCollectionError::BufferOverflow(dropped))
        } else {
            Ok(spans)
        }
    }

    fn push_span(&mut self, span_data: SpanData) {
        if self.spans.len() >= self.max_size {
            self.dropped += 1;
            return;
        }
        self.spans.push_back(span_data);
    }
}

impl SpanStore {
    /// Get the global span store, if it has been initialized.
    pub fn get_global_store() -> Option<&'static Arc<Mutex<SpanStore>>> {
        GLOBAL_SPAN_STORE.get()
    }
}

/// The store the [`InMemorySpanExporter`] writes to, shared with the code that drains it.
static GLOBAL_SPAN_STORE: OnceLock<Arc<Mutex<SpanStore>>> = OnceLock::new();

/// Initialize the global span store with the given maximum buffer size.
///
/// The first call fixes the buffer size for the lifetime of the process.
fn init_span_store(max_buffer_size: usize) {
    GLOBAL_SPAN_STORE.get_or_init(|| Arc::new(Mutex::new(SpanStore::new(max_buffer_size))));
}

/// Exporter that stores spans in memory for later retrieval.
///
/// Stores raw [`SpanData`] into the global [`SpanStore`].
#[derive(Debug)]
pub(crate) struct InMemorySpanExporter {}

impl InMemorySpanExporter {
    pub(crate) fn new() -> Self {
        init_span_store(crate::tracing_buffer_size());
        Self {}
    }
}

impl SpanExporter for InMemorySpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if let Some(store) = GLOBAL_SPAN_STORE.get() {
            let mut store = store.lock().unwrap();
            for span in batch {
                store.push_span(span);
            }
        }
        Ok(())
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown(&self) -> OTelSdkResult {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::borrow::Cow;
    use std::time::SystemTime;

    use opentelemetry::InstrumentationScope;
    use opentelemetry::trace::{SpanContext, SpanId, SpanKind, Status};
    use opentelemetry_sdk::trace::{SpanEvents, SpanLinks};

    #[test]
    fn drains_every_span_it_held() {
        let mut store = SpanStore::new(2);
        store.push_span(span("first"));
        store.push_span(span("second"));

        let names: Vec<_> = store.drain_spans().unwrap().map(|s| s.name).collect();

        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn draining_an_overflowed_store_returns_error() {
        let mut store = SpanStore::new(1);
        store.push_span(span("kept"));
        store.push_span(span("dropped"));
        store.push_span(span("dropped"));

        let res = store.drain_spans();
        assert!(matches!(res, Err(SpanCollectionError::BufferOverflow(2))));
    }

    fn span(name: &'static str) -> SpanData {
        SpanData {
            span_context: SpanContext::NONE,
            parent_span_id: SpanId::INVALID,
            parent_span_is_remote: false,
            span_kind: SpanKind::Internal,
            name: Cow::Borrowed(name),
            start_time: SystemTime::UNIX_EPOCH,
            end_time: SystemTime::UNIX_EPOCH,
            attributes: Vec::new(),
            dropped_attributes_count: 0,
            events: SpanEvents::default(),
            links: SpanLinks::default(),
            status: Status::Unset,
            instrumentation_scope: InstrumentationScope::default(),
        }
    }
}
