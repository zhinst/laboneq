// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU32;

use laboneq_common::near_time::NtStepKey;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, Sweep};
use laboneq_dsl::types::{ParameterUid, SweepParameter};
use laboneq_error::LabOneQError;

use super::environment::Environment;

/// One run of the real-time program: a near-time step, narrowed to one
/// chunk when the experiment chunks a sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RtJob<'a> {
    /// The iteration of each enclosing near-time sweep, then the chunk index.
    pub(crate) key: &'a NtStepKey,
    /// The values of the near-time parameters in scope.
    pub(crate) env: &'a Environment,
    /// The index of the chunk, when the experiment chunks a sweep.
    pub(crate) chunk: Option<usize>,
}

/// Calls `f` with every real-time job of the experiment at `root`, in the
/// order the experiment runs them. Stops at the first error.
///
/// Every sweep outside the real-time boundary is a near-time loop. With a
/// `chunk_count`, every run of the real-time program makes one job per chunk.
/// A job's `env` holds integer parameter values as floats.
pub(crate) fn for_each_rt_job(
    root: &ExperimentNode,
    parameters: &[SweepParameter],
    chunk_count: Option<NonZeroU32>,
    f: impl FnMut(RtJob<'_>) -> Result<(), LabOneQError>,
) -> Result<(), LabOneQError> {
    let mut walk = Walk {
        parameters: parameters.iter().map(|p| (p.uid, p)).collect(),
        chunk_count,
        key: NtStepKey(vec![]),
        env: Environment::default(),
        f,
    };
    walk.visit(root)
}

/// Returns the number of real-time jobs of the experiment at `node`.
pub(crate) fn count_rt_jobs(node: &ExperimentNode, chunk_count: Option<NonZeroU32>) -> usize {
    let children = || -> usize {
        node.children
            .iter()
            .map(|child| count_rt_jobs(child, chunk_count))
            .sum()
    };
    match &node.kind {
        Operation::Sweep(sweep) => sweep.count.get() as usize * children(),
        Operation::RealTimeBoundary => chunk_count.map_or(1, |count| count.get() as usize),
        _ => children(),
    }
}

struct Walk<'a, F> {
    parameters: HashMap<ParameterUid, &'a SweepParameter>,
    chunk_count: Option<NonZeroU32>,
    /// The iteration of each enclosing sweep, outermost first, then the chunk
    /// index while a chunked job runs.
    key: NtStepKey,
    /// The parameter values of the enclosing sweeps.
    env: Environment,
    f: F,
}

impl<F: FnMut(RtJob<'_>) -> Result<(), LabOneQError>> Walk<'_, F> {
    fn visit(&mut self, node: &ExperimentNode) -> Result<(), LabOneQError> {
        match &node.kind {
            Operation::Sweep(sweep) => self.visit_sweep(node, sweep),
            Operation::RealTimeBoundary => self.run_jobs(),
            _ => self.visit_children(node),
        }
    }

    fn visit_children(&mut self, node: &ExperimentNode) -> Result<(), LabOneQError> {
        node.children.iter().try_for_each(|child| self.visit(child))
    }

    fn visit_sweep(&mut self, node: &ExperimentNode, sweep: &Sweep) -> Result<(), LabOneQError> {
        for iteration in 0..sweep.count.get() as usize {
            let scope = self.env.len();
            self.key.0.push(iteration);
            for uid in &sweep.parameters {
                let value = self.parameters[uid]
                    .value_numeric_at_index(iteration)
                    .expect("Internal error: Sweep parameter shorter than its sweep");
                self.env.push(*uid, value.to_float());
            }
            self.visit_children(node)?;
            self.key.0.pop();
            self.env.truncate(scope);
        }
        Ok(())
    }

    fn run_jobs(&mut self) -> Result<(), LabOneQError> {
        let Some(count) = self.chunk_count else {
            return (self.f)(RtJob {
                key: &self.key,
                env: &self.env,
                chunk: None,
            });
        };
        for index in 0..count.get() as usize {
            self.key.0.push(index);
            (self.f)(RtJob {
                key: &self.key,
                env: &self.env,
                chunk: Some(index),
            })?;
            self.key.0.pop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use laboneq_common::named_id::NamedId;
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::NearTimeCallback;
    use laboneq_dsl::operation::builders::SweepBuilder;
    use laboneq_dsl::types::NumericLiteral;

    use crate::execution::{Statement, create_execution};

    #[test]
    fn nested_sweeps_make_one_job_per_step_in_execution_order() {
        let outer = parameter(1, vec![0.1, 0.2]);
        let inner = parameter(2, vec![1.0, 2.0, 3.0]);
        let root = node_structure!(
            Operation::Root,
            [(
                sweep(1, &outer),
                [
                    (callback(), []),
                    (sweep(2, &inner), [(Operation::RealTimeBoundary, [])])
                ]
            )]
        );

        let jobs = collect_jobs(&root, &[outer.clone(), inner.clone()], None);

        let steps: Vec<_> = jobs
            .iter()
            .map(|(key, env)| (key.0.clone(), values(env, &[&outer, &inner])))
            .collect();
        assert_eq!(
            steps,
            vec![
                (vec![0, 0], vec![Some(0.1), Some(1.0)]),
                (vec![0, 1], vec![Some(0.1), Some(2.0)]),
                (vec![0, 2], vec![Some(0.1), Some(3.0)]),
                (vec![1, 0], vec![Some(0.2), Some(1.0)]),
                (vec![1, 1], vec![Some(0.2), Some(2.0)]),
                (vec![1, 2], vec![Some(0.2), Some(3.0)]),
            ]
        );
    }

    #[test]
    fn a_sweep_beside_the_boundary_takes_no_part_in_its_jobs() {
        let outer = parameter(1, vec![0.1]);
        let sibling = parameter(2, vec![1.0, 2.0]);
        let root = node_structure!(
            Operation::Root,
            [(
                sweep(1, &outer),
                [(sweep(2, &sibling), []), (Operation::RealTimeBoundary, [])]
            )]
        );

        let jobs = collect_jobs(&root, &[outer.clone(), sibling.clone()], None);

        assert_eq!(jobs.len(), 1);
        assert_eq!(
            values(&jobs[0].1, &[&outer, &sibling]),
            vec![Some(0.1), None]
        );
    }

    #[test]
    fn integer_values_read_as_floats() {
        let selector = SweepParameter::new(1.into(), vec![0_i64, 1]).unwrap();
        let root = node_structure!(
            Operation::Root,
            [(sweep(1, &selector), [(Operation::RealTimeBoundary, [])])]
        );

        let jobs = collect_jobs(&root, std::slice::from_ref(&selector), None);

        assert_eq!(
            jobs[1].1.get(selector.uid),
            Some(NumericLiteral::Float(1.0))
        );
    }

    #[test]
    fn job_keys_and_count_follow_the_real_time_runs_of_the_execution() {
        let outer = parameter(1, vec![0.1, 0.2]);
        let inner = parameter(2, vec![1.0, 2.0, 3.0]);
        let sibling = parameter(3, vec![5.0, 6.0]);
        let root = node_structure!(
            Operation::Root,
            [(
                sweep(1, &outer),
                [
                    (callback(), []),
                    (sweep(3, &sibling), []),
                    (sweep(2, &inner), [(Operation::RealTimeBoundary, [])])
                ]
            )]
        );
        let parameters = [outer, inner, sibling];
        let execution = create_execution(&root).unwrap();

        for chunk_count in [None, NonZeroU32::new(3)] {
            let keys: Vec<_> = collect_jobs(&root, &parameters, chunk_count)
                .into_iter()
                .map(|(key, _)| key)
                .collect();
            let runs = real_time_runs(&execution.statements, &mut vec![], chunk_count);
            assert_eq!(keys, runs);
            assert_eq!(count_rt_jobs(&root, chunk_count), runs.len());
        }
    }

    fn collect_jobs(
        root: &ExperimentNode,
        parameters: &[SweepParameter],
        chunk_count: Option<NonZeroU32>,
    ) -> Vec<(NtStepKey, Environment)> {
        let mut jobs = vec![];
        for_each_rt_job(root, parameters, chunk_count, |job| {
            jobs.push((job.key.clone(), job.env.clone()));
            Ok(())
        })
        .unwrap();
        jobs
    }

    /// Returns the step key of every real-time run of `statements`: the indices
    /// of the enclosing loops, then the chunk index.
    fn real_time_runs(
        statements: &[Statement],
        iterations: &mut Vec<usize>,
        chunk_count: Option<NonZeroU32>,
    ) -> Vec<NtStepKey> {
        let mut keys = vec![];
        for statement in statements {
            match statement {
                Statement::Loop { count, body } => {
                    for iteration in 0..count.get() as usize {
                        iterations.push(iteration);
                        keys.extend(real_time_runs(body, iterations, chunk_count));
                        iterations.pop();
                    }
                }
                Statement::ExecRealTime => match chunk_count {
                    None => keys.push(NtStepKey(iterations.clone())),
                    Some(count) => keys.extend(
                        (0..count.get() as usize)
                            .map(|chunk| NtStepKey([iterations.as_slice(), &[chunk]].concat())),
                    ),
                },
                _ => {}
            }
        }
        keys
    }

    fn sweep(id: u32, parameter: &SweepParameter) -> Operation {
        let count = NonZeroU32::new(parameter.len() as u32).unwrap();
        Operation::Sweep(SweepBuilder::new(id.into(), vec![parameter.uid], count).build())
    }

    fn callback() -> Operation {
        Operation::NearTimeCallback(NearTimeCallback {
            callback_id: NamedId::debug_id(10),
            args: vec![],
        })
    }

    fn parameter(id: u32, values: Vec<f64>) -> SweepParameter {
        SweepParameter::new(id.into(), values).unwrap()
    }

    fn values(env: &Environment, parameters: &[&SweepParameter]) -> Vec<Option<f64>> {
        parameters
            .iter()
            .map(|p| env.get(p.uid).map(|v| v.try_into().unwrap()))
            .collect()
    }
}
