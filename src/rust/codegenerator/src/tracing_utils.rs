// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Utility for managing tracing contexts in parallel code generation.
//!
//! This module provides a [`ParallelTraceContext`] struct that captures the current tracing span and dispatcher,
//! allowing it to be passed into parallel tasks. The [`par_trace!`] macro can be used to execute
//! code within the context of the captured tracing span, ensuring that all generated spans are correctly linked
//! to the parent span, even when executed in parallel threads.

pub(crate) struct ParallelTraceContext {
    pub(super) parent: tracing::Span,
    pub(super) dispatch: tracing::Dispatch,
}

impl ParallelTraceContext {
    pub(crate) fn new() -> Self {
        ParallelTraceContext {
            parent: tracing::Span::current(),
            dispatch: tracing::dispatcher::get_default(|d| d.clone()),
        }
    }
}

// Macro to generate traced calls with custom span names
#[macro_export]
macro_rules! par_trace {
    ($trace_ctx:expr, $span_name:literal, $f:expr) => {
        tracing::dispatcher::with_default(&$trace_ctx.dispatch, || {
            tracing::info_span!(parent: &$trace_ctx.parent, $span_name).in_scope(|| $f)
        })
    };
}
