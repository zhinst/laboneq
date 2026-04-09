// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides an OpenTelemetry exporter that stores spans in memory for later retrieval.
//! The spans are stored as raw `SpanData` in a circular buffer, and can be retrieved via the global `SpanStore`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::SpanData;
use opentelemetry_sdk::trace::SpanExporter;

/// The [`SpanStore`] is used to store spans in memory for later retrieval.
#[derive(Debug)]
pub struct SpanStore {
    spans: VecDeque<SpanData>, // Raw span data, serialize on demand
    max_size: usize,
    dropped_count: u64,
}

impl SpanStore {
    fn new(max_size: usize) -> Self {
        Self {
            spans: VecDeque::with_capacity(max_size),
            max_size,
            dropped_count: 0,
        }
    }

    /// Drain all spans from the store and return an iterator over the raw [`SpanData`].
    pub fn drain_spans(&mut self) -> impl Iterator<Item = SpanData> + '_ {
        self.spans.drain(..)
    }

    fn push_span(&mut self, span_data: SpanData) {
        if self.spans.len() >= self.max_size {
            self.spans.pop_front();
            self.dropped_count += 1;
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

/// Global span store configuration and initialization
/// This is used by the InMemorySpanExporter to store spans in memory and retrieve them later.
/// Default maximum number of spans to store in memory. When the limit is reached, the oldest spans will be dropped.
const DEFAULT_MAX_BUFFER_SIZE: usize = 10000;
/// Initialize the global span store with the given maximum buffer size.
static GLOBAL_SPAN_STORE: OnceLock<Arc<Mutex<SpanStore>>> = OnceLock::new();

/// Initialize the global span store with the given maximum buffer size.
fn init_span_store(max_buffer_size: Option<usize>) {
    GLOBAL_SPAN_STORE.get_or_init(|| {
        Arc::new(Mutex::new(SpanStore::new(
            max_buffer_size.unwrap_or(DEFAULT_MAX_BUFFER_SIZE),
        )))
    });
}

/// Exporter that stores spans in memory for later retrieval.
///
/// Stores raw [`SpanData`] into the global [`SpanStore`].
#[derive(Debug)]
pub(crate) struct InMemorySpanExporter {}

impl InMemorySpanExporter {
    pub(crate) fn new() -> Self {
        init_span_store(None);
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

    fn force_flush(&mut self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown(&mut self) -> OTelSdkResult {
        Ok(())
    }
}
