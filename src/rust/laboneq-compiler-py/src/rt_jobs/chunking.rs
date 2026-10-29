// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Finds the chunking mode of the chunked sweep, and retries compilation with more
//! chunks under auto-chunking.

use std::num::NonZeroU32;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, Sweep};
use laboneq_error::{LabOneQError, WithContext};

use crate::error::Error;
use crate::experiment::Experiment;

const RESOURCE_LIMIT_MSG: &str = "\
Compilation error: resource limitation exceeded.
To circumvent this, try one or more of the following:
- Double-check the integrity of your experiment (look for unexpectedly long pulses, large numbers of sweep steps, etc.)
- Reduce the number of sweep steps
- Reduce the number of variations in the pulses that are being played
- Enable chunking for a sweep
- If chunking is already enabled, increase the chunk count or switch to automatic chunking
- For QCCS instruments, use pulse lengths that are multiples of 8 ns to improve memory efficiency";

const AUTO_CHUNK_EXHAUSTED_MSG: &str = "\
Automatic chunking could not find a chunk count that stays within the resource limits.
One iteration of the chunked sweep is too large to execute.
To circumvent this, try one or more of the following:
- Chunk another sweep instead (e.g. for nested sweeps, enable chunking for the inner one)
- Reduce the size of the program in a single sweep iteration
- For QCCS instruments, use pulse lengths that are multiples of 8 ns to improve memory efficiency";

pub(super) fn collect_chunking_mode(
    experiment: &Experiment,
) -> Result<Option<ChunkingMode>, Error> {
    let mut visitor = ExperimentVisitor::new();
    visitor.visit_node(&experiment.root)?;
    Ok(visitor.chunking)
}

struct ExperimentVisitor {
    chunking: Option<ChunkingMode>,
}
impl ExperimentVisitor {
    fn new() -> Self {
        Self { chunking: None }
    }

    fn generic_visit(&mut self, node: &ExperimentNode) -> Result<(), Error> {
        for child in &node.children {
            self.visit_node(child)?;
        }
        Ok(())
    }

    fn visit_node(&mut self, node: &ExperimentNode) -> Result<(), Error> {
        match &node.kind {
            Operation::Sweep(op) => self.visit_sweep(node, op),
            _ => self.generic_visit(node),
        }
    }

    fn visit_sweep(&mut self, node: &ExperimentNode, op: &Sweep) -> Result<(), Error> {
        if !op.is_chunked() {
            self.generic_visit(node)?;
            return Ok(());
        }
        if op.is_chunked() && self.chunking.is_some() {
            // Should be validated earlier, but check just in case
            return Err(Error::new(
                "Multiple chunked sweeps found in experiment, which is not supported",
            ));
        }
        let sweep_iterations = op.count;
        let mut chunk_count = op.chunk_count;

        let chunking_mode = if op.auto_chunking {
            ChunkingMode::Auto(AutoChunking::new(sweep_iterations, chunk_count))
        } else {
            if chunk_count > sweep_iterations {
                laboneq_log::warn!(
                    "Provided chunk count ({}) is larger than the sweep length ({}). Using {} instead.",
                    chunk_count,
                    sweep_iterations,
                    sweep_iterations
                );
                chunk_count = sweep_iterations;
            }
            if !sweep_iterations.get().is_multiple_of(chunk_count.get()) {
                return Err(Error::new(format!(
                    "Chunk count ({chunk_count}) does not evenly divide sweep length ({sweep_iterations})"
                )));
            }
            ChunkingMode::Manual { chunk_count }
        };
        self.chunking = Some(chunking_mode);
        Ok(())
    }
}

pub(super) enum ChunkingMode {
    Auto(AutoChunking),
    Manual { chunk_count: NonZeroU32 },
}

pub(super) struct AutoChunking {
    iterations: NonZeroU32,
    initial_chunk_count: NonZeroU32,

    divisors: Vec<NonZeroU32>,
}

impl AutoChunking {
    fn new(iterations: NonZeroU32, chunk_count: NonZeroU32) -> Self {
        let divisors = Self::divisors(iterations);
        let initial_chunk_count = Self::chunk_count_trial(chunk_count, &divisors);

        Self {
            iterations,
            initial_chunk_count,
            divisors,
        }
    }

    fn resize(self, chunk_count: NonZeroU32) -> AutoChunking {
        Self::new_with_divisors(self.iterations, chunk_count, self.divisors)
    }

    fn new_with_divisors(
        iterations: NonZeroU32,
        chunk_count: NonZeroU32,
        divisors: Vec<NonZeroU32>,
    ) -> Self {
        let initial_chunk_count = Self::chunk_count_trial(chunk_count, &divisors);
        Self {
            iterations,
            initial_chunk_count,
            divisors,
        }
    }

    fn divisors(n: NonZeroU32) -> Vec<NonZeroU32> {
        (1..=n.get() / 2)
            .filter(|&i| n.get().is_multiple_of(i))
            .map(|i| NonZeroU32::new(i).unwrap())
            .chain(std::iter::once(n))
            .collect()
    }

    fn chunk_count_trial(requested: NonZeroU32, candidates: &[NonZeroU32]) -> NonZeroU32 {
        let idx = candidates.partition_point(|&x| x < requested);
        candidates[idx.min(candidates.len() - 1)]
    }
}

/// Runs `compile` for the chunk count of `chunking_mode`. With auto-chunking, retries
/// with more chunks while the attempt exceeds a resource limit and the sweep allows
/// more. Returns the output of the attempt that succeeded.
pub(super) fn run_chunked<T>(
    chunking_mode: Option<ChunkingMode>,
    compile: impl Fn(Option<NonZeroU32>) -> Result<T, LabOneQError>,
) -> Result<T, LabOneQError> {
    let mut auto_chunking = match chunking_mode {
        None => {
            return compile(None).map_err(|e| with_hint(e, RESOURCE_LIMIT_MSG));
        }
        Some(ChunkingMode::Manual { chunk_count }) => {
            return compile(sanitize_chunk_count(chunk_count))
                .map_err(|e| with_hint(e, RESOURCE_LIMIT_MSG));
        }
        Some(ChunkingMode::Auto(auto_chunking)) => auto_chunking,
    };

    loop {
        let requested = auto_chunking.initial_chunk_count;
        laboneq_log::debug!("Attempting to compile with {} chunks", requested);
        match compile(sanitize_chunk_count(requested)) {
            Ok(output) => {
                laboneq_log::info!("Auto-chunked sweep divided into {} chunks", requested);
                return Ok(output);
            }
            Err(LabOneQError::ResourceExhaustion(e)) => {
                laboneq_log::debug!(
                    "The attempt with {} chunks failed with resource exhaustion: {}",
                    requested,
                    e
                );
                if requested == auto_chunking.iterations {
                    return Err(with_hint(
                        LabOneQError::ResourceExhaustion(e),
                        AUTO_CHUNK_EXHAUSTED_MSG,
                    ));
                }
                let multiplier = e.usage.unwrap_or(2.0).ceil().max(2.0) as u32;
                auto_chunking = auto_chunking.resize(
                    requested
                        .get()
                        .saturating_mul(multiplier)
                        .try_into()
                        .expect("Internal error: Expected chunk count to be non-zero"),
                );
            }
            Err(e) => return Err(e),
        }
    }
}

/// Returns `None` for a chunk count of 1, which is equivalent to not chunking.
fn sanitize_chunk_count(chunk_count: NonZeroU32) -> Option<NonZeroU32> {
    Some(chunk_count).filter(|c| c.get() != 1)
}

fn with_hint(error: LabOneQError, hint: &'static str) -> LabOneQError {
    if let LabOneQError::ResourceExhaustion(e) = error {
        laboneq_error::laboneq_error!("{}", hint).with_context(|| e.to_string())
    } else {
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    #[test]
    fn divisors_composite() {
        assert_eq!(
            AutoChunking::divisors(nz(12)),
            vec![nz(1), nz(2), nz(3), nz(4), nz(6), nz(12)]
        );
    }

    #[test]
    fn divisors_prime() {
        assert_eq!(AutoChunking::divisors(nz(7)), vec![nz(1), nz(7)]);
    }

    #[test]
    fn divisors_one() {
        assert_eq!(AutoChunking::divisors(nz(1)), vec![nz(1)]);
    }

    #[test]
    fn chunk_count_trial_exact_divisor() {
        let divs = vec![nz(1), nz(2), nz(3), nz(4), nz(6), nz(12)];
        assert_eq!(AutoChunking::chunk_count_trial(nz(6), &divs), nz(6));
    }

    #[test]
    fn chunk_count_trial_rounds_up_to_next_divisor() {
        let divs = vec![nz(1), nz(2), nz(3), nz(4), nz(6), nz(12)];
        assert_eq!(AutoChunking::chunk_count_trial(nz(5), &divs), nz(6));
    }

    #[test]
    fn chunk_count_trial_clamped_at_max() {
        let divs = vec![nz(1), nz(2), nz(3), nz(4), nz(6), nz(12)];
        assert_eq!(AutoChunking::chunk_count_trial(nz(20), &divs), nz(12));
    }

    #[test]
    fn chunk_count_trial_below_all_returns_minimum() {
        let divs = vec![nz(1), nz(2), nz(3), nz(4), nz(6), nz(12)];
        assert_eq!(AutoChunking::chunk_count_trial(nz(1), &divs), nz(1));
    }

    #[test]
    fn new_normalizes_to_nearest_divisor() {
        assert_eq!(
            AutoChunking::new(nz(12), nz(5)).initial_chunk_count.get(),
            6
        );
    }

    #[test]
    fn new_keeps_exact_divisor_unchanged() {
        assert_eq!(
            AutoChunking::new(nz(12), nz(4)).initial_chunk_count.get(),
            4
        );
    }

    #[test]
    fn new_clamps_above_iterations() {
        assert_eq!(
            AutoChunking::new(nz(12), nz(20)).initial_chunk_count.get(),
            12
        );
    }

    #[test]
    fn new_prime_iterations_rounds_to_n() {
        // 7 is prime: divisors are [1, 7]; requested 4 rounds up to 7
        assert_eq!(AutoChunking::new(nz(7), nz(4)).initial_chunk_count.get(), 7);
    }

    #[test]
    fn new_preserves_iterations() {
        assert_eq!(AutoChunking::new(nz(12), nz(3)).iterations.get(), 12);
    }

    #[test]
    fn resize_normalizes_new_count() {
        let ac = AutoChunking::new(nz(12), nz(2)).resize(nz(5));
        assert_eq!(ac.initial_chunk_count.get(), 6);
    }

    #[test]
    fn resize_preserves_iterations() {
        let ac = AutoChunking::new(nz(12), nz(2)).resize(nz(5));
        assert_eq!(ac.iterations.get(), 12);
    }

    #[test]
    fn resize_progression_doubles_each_step() {
        // Simulates the compiler retry loop multiplying chunk_count by 2 each time
        let ac = AutoChunking::new(nz(12), nz(1));
        assert_eq!(ac.initial_chunk_count.get(), 1);

        let ac = ac.resize(nz(2));
        assert_eq!(ac.initial_chunk_count.get(), 2);

        let ac = ac.resize(nz(4));
        assert_eq!(ac.initial_chunk_count.get(), 4);

        // 8 is not a divisor of 12; next valid divisor is 12
        let ac = ac.resize(nz(8));
        assert_eq!(ac.initial_chunk_count.get(), 12);
    }
}
