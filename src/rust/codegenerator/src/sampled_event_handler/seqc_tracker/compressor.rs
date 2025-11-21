// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_log::warn;

use super::seqc_generator::SeqCGenerator;
use super::seqc_statements::SeqCStatement;
use crate::ir::compilation_job::DeviceKind;
use std::{
    hash::{Hash, Hasher},
    rc::Rc,
};

#[derive(PartialEq, Clone, Debug)]
enum HashOrRun {
    Hash(u64),
    Run {
        count: usize,
        statements_hashes: Vec<HashOrRun>,
    },
}

impl Hash for HashOrRun {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            HashOrRun::Hash(h) => h.hash(state),
            HashOrRun::Run {
                count,
                statements_hashes,
            } => {
                count.hash(state);
                for h in statements_hashes {
                    h.hash(state);
                }
            }
        }
    }
}

trait CostFunction {
    fn cost_function(&self, run: &Run) -> i64;
}

struct StatementCostFunction<'a> {
    statement_by_hash: &'a std::collections::HashMap<u64, Rc<SeqCStatement>>,
}
impl CostFunction for StatementCostFunction<'_> {
    fn cost_function(&self, run: &Run) -> i64 {
        let mut complexity: i64 = 0;
        for h in run.word.iter() {
            let HashOrRun::Hash(hash) = h else {
                unreachable!();
            };
            let statement = self.statement_by_hash.get(hash).unwrap();
            complexity += statement.complexity() as i64;
        }
        let cost = -((run.count as i64) - 1) * complexity + 2;
        let HashOrRun::Hash(hash) = run.word[0] else {
            unreachable!();
        };
        let first_statement = self.statement_by_hash.get(&hash).unwrap();
        if let SeqCStatement::FunctionCall { name, .. } = first_statement.as_ref()
            && name == "startQA"
        {
            // Never separate startQA from the preceding playWave or playZero (HBAR-2075)
            return cost + 1_000_000;
        }
        cost
    }
}

struct GeneratorCostFunction<'a> {
    generator_by_hash: &'a std::collections::HashMap<u64, &'a SeqCGenerator>,
}
impl CostFunction for GeneratorCostFunction<'_> {
    fn cost_function(&self, run: &Run) -> i64 {
        let mut complexity: i64 = 0;
        for h in run.word.iter() {
            let HashOrRun::Hash(hash) = h else {
                unreachable!();
            };
            let generator = self.generator_by_hash.get(hash).unwrap();
            complexity += generator.estimate_complexity() as i64;
        }
        -((run.count as i64) - 1) * complexity + 2
    }
}

pub fn to_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone)]
struct Run {
    word: Vec<HashOrRun>,
    count: usize,
}
impl Run {
    fn span(&self) -> usize {
        self.word.len() * self.count
    }
}

fn compress_hashes<T: CostFunction>(
    hashes: &[u64],
    cost_function: &T,
    recurse: bool, // defaults to false
) -> Result<Vec<HashOrRun>, String> {
    compress_hashes_recursive(
        hashes.iter().map(|h| HashOrRun::Hash(*h)).collect(),
        cost_function,
        recurse,
    )
}

// Idea of the compression algorithm:
//
// Find possible compression 'words' by looking for equal tokens (the first token has to agree)
// foreach input_token:
//   if compression is possbly a repeating word
//     compute max repeat of this word as current best
//     check until the end of the current best if there is an overlapping even better possibility
//     when reachin the end of the current best, compress away
fn compress_hashes_recursive<T: CostFunction>(
    hashes: Vec<HashOrRun>,
    cost_function: &T,
    recurse: bool,
) -> Result<Vec<HashOrRun>, String> {
    // todo: This can be optimized by avoiding the clone during `to_vec()` as
    // noted in the comment below. Consider working with ranges/start and end of
    // original list.

    let mut hashes = hashes;
    let mut retval = Vec::new();
    loop {
        if hashes.len() <= 1 {
            retval.extend(hashes);
            return Ok(retval);
        }

        // first, we compute `offsets`, which, for each hash in the list, gives
        // the relative position to the next place the same hash appears again
        let mut offsets: Vec<Option<usize>> = vec![None; hashes.len()];
        let mut next_seen_map = std::collections::HashMap::<u64, usize>::new();
        for (i, hash) in hashes.iter().enumerate().rev() {
            let HashOrRun::Hash(hash) = hash else {
                unreachable!();
            };
            if let Some(offs) = next_seen_map.insert(*hash, i) {
                offsets[i] = Some(offs - i);
            }
        }
        let mut runs = std::collections::HashMap::<u64, Vec<(usize, Run)>>::new(); // hash -> vec[(start, Run)]
        let mut best_run: Option<Run> = None;
        let mut best_run_start: Option<usize> = None;
        let mut best_run_end: Option<usize> = None;
        let mut best_cost = 0;
        for (index, offset) in offsets.iter().enumerate() {
            match best_run_end {
                None => {}
                Some(end) if index >= end => {
                    // We have found no overlapping better run. Therefore let's compress away the current best_run now and remove it from hashes.
                    // The remainder of the list will be handled in the next iteration
                    // of the outer loop
                    break;
                }
                _ => (),
            }
            if offset.is_none() {
                continue;
            }
            let offset = offset.unwrap();
            let word = &hashes[index..index + offset];
            let first_hash = to_hash(&word);
            let run = runs.get(&first_hash);
            if let Some(runs_for_hash) = run
                && runs_for_hash.iter().any(|(_, r)| r.word != word)
            {
                panic!("hash collision detected");
            }
            if let Some(runs_for_hash) = runs.get(&first_hash)
                && runs_for_hash
                    .iter()
                    .any(|(start, run)| *start < index && index <= start + run.span())
            {
                continue;
            }
            let mut run_length = 1;
            while index + (run_length + 1) * offset <= hashes.len()
                && word
                    == &hashes[(index + run_length * offset)..(index + (run_length + 1) * offset)]
            {
                run_length += 1;
            }
            let this_run = Run {
                // TODO: can we avoid cloning the word?
                word: word.to_vec(),
                count: run_length,
            };
            let this_cost = cost_function.cost_function(&this_run);
            if this_cost < best_cost {
                best_cost = this_cost;
                best_run_start = Some(index);
                best_run_end = Some(index + this_run.span());
                best_run = Some(this_run.clone());
            }
            runs.entry(first_hash).or_default().push((index, this_run));
        }
        if best_run.is_none() {
            retval.extend(hashes);
            return Ok(retval);
        }
        if recurse {
            best_run = Some(Run {
                count: best_run.as_ref().unwrap().count,
                word: compress_hashes_recursive(best_run.unwrap().word, cost_function, true)?,
            });
        }
        let best_run_start = best_run_start.unwrap();
        let best_run_end = best_run_end.unwrap();
        let best_run = best_run.unwrap();
        let remaining_hashes = hashes.split_off(best_run_end);
        retval.extend(hashes.drain(..best_run_start));
        retval.push(HashOrRun::Run {
            count: best_run.count,
            statements_hashes: best_run.word,
        });
        hashes = remaining_hashes;
    }
}

pub fn compress_generator(generator: SeqCGenerator) -> SeqCGenerator {
    let mut statement_hashes: Vec<u64> = Vec::new();
    let mut statement_by_hash = std::collections::HashMap::<u64, Rc<SeqCStatement>>::new();

    for statement in generator.statements() {
        let hash = statement.to_hash();
        if let Some(old_value) = statement_by_hash.insert(hash, Rc::clone(statement))
            && &old_value != statement
        {
            warn!("hash collision detected, skipping code compression");
            return generator;
        }
        statement_hashes.push(hash);
    }
    let mut compressed_gen = generator.create();
    let compressed_statements = compress_hashes(
        &statement_hashes,
        &StatementCostFunction {
            statement_by_hash: &statement_by_hash,
        },
        false,
    );
    match compressed_statements {
        Err(e) => {
            warn!("{e}");
            return generator;
        }
        Ok(compressed_statesments) => {
            for cs in compressed_statesments {
                match cs {
                    HashOrRun::Hash(hash) => {
                        let statement = statement_by_hash.get(&hash).unwrap();
                        compressed_gen.add_statement(Rc::clone(statement));
                    }
                    HashOrRun::Run {
                        count,
                        statements_hashes,
                    } => {
                        let mut body = generator.create();
                        for hash in &statements_hashes {
                            let HashOrRun::Hash(hash) = hash else {
                                unreachable!();
                            };
                            let statement = statement_by_hash.get(hash).unwrap();
                            body.add_statement(Rc::clone(statement));
                        }
                        compressed_gen.add_repeat(count as u64, body);
                    }
                }
            }
        }
    }
    compressed_gen
}

pub fn merge_generators(
    generators: Vec<SeqCGenerator>,
    compress: bool, // defaults to True
) -> SeqCGenerator {
    let mut compress = compress;
    if generators.is_empty() {
        // Create an empty generator of arbitrary type
        return SeqCGenerator::new(DeviceKind::SHFQA.traits(), true);
    }

    let mut retval = generators[0].create();

    if compress {
        let mut statement_hashes: Vec<u64> = Vec::new();
        let mut generator_by_hash = std::collections::HashMap::<u64, &SeqCGenerator>::new();

        for generator in &generators {
            let hash = to_hash(generator);
            if let Some(old_value) = generator_by_hash.insert(hash, generator)
                && old_value != generator
            {
                compress = false;
            }
            statement_hashes.push(hash);
        }
        if !compress {
            warn!("hash collision detected, skipping code compression");
        } else {
            let compressed_generators = compress_hashes(
                &statement_hashes,
                &GeneratorCostFunction {
                    generator_by_hash: &generator_by_hash,
                },
                false,
            );
            match compressed_generators {
                Err(error) => {
                    warn!("{}", error);
                }
                Ok(cgs) => {
                    let mut did_compress = false;
                    for cg in cgs {
                        match cg {
                            HashOrRun::Run {
                                count,
                                statements_hashes,
                            } => {
                                let body = if statements_hashes.len() == 1 {
                                    let HashOrRun::Hash(first_hash) = &statements_hashes[0] else {
                                        unreachable!();
                                    };
                                    // TODO: Consider using shared pointers here instead of cloning
                                    (*generator_by_hash.get(first_hash).unwrap()).clone()
                                } else {
                                    did_compress = true;
                                    let mut body = retval.create();
                                    for gen_hash in statements_hashes {
                                        let HashOrRun::Hash(hash) = gen_hash else {
                                            unreachable!();
                                        };
                                        let generator = generator_by_hash.get(&hash).unwrap();
                                        body.append_statements_from(generator);
                                    }
                                    body
                                };
                                retval.add_repeat(count as u64, compress_generator(body));
                            }
                            HashOrRun::Hash(hash) => {
                                retval
                                    .append_statements_from(generator_by_hash.get(&hash).unwrap());
                            }
                        }
                    }
                    if did_compress {
                        // 2nd pass on the merged generator, finding patterns that partially span across
                        // multiple of the original parts.
                        retval = compress_generator(retval);
                    }
                    return retval;
                }
            }
        }
    }
    for g in generators {
        retval.append_statements_from(&g);
    }
    retval
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCostFunctionDefault;
    impl CostFunction for TestCostFunctionDefault {
        fn cost_function(&self, run: &Run) -> i64 {
            -((run.count as i64 - 1) * run.word.len() as i64 - 2)
        }
    }
    struct TestCostFunction2;
    impl CostFunction for TestCostFunction2 {
        fn cost_function(&self, run: &Run) -> i64 {
            let mut score = (run.count as i64 - 1) * run.word.len() as i64 - 2;
            if run.word[0] == HashOrRun::Hash(1) {
                score -= 1;
            }
            -score
        }
    }

    #[test]
    fn test_compress_hashes_cost_functions() {
        // Verifies that different cost functions lead to different compression results
        let hashes = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2];
        let compressed = compress_hashes(&hashes, &TestCostFunctionDefault {}, false);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 3);
        assert_eq!(
            compressed[0],
            HashOrRun::Run {
                count: 3,
                statements_hashes: vec![HashOrRun::Hash(1), HashOrRun::Hash(2), HashOrRun::Hash(3)]
            }
        );
        assert_eq!(compressed[1], HashOrRun::Hash(1));
        assert_eq!(compressed[2], HashOrRun::Hash(2));

        let compressed = compress_hashes(&hashes, &TestCostFunction2 {}, false);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 3);
        assert_eq!(compressed[0], HashOrRun::Hash(1));
        assert_eq!(
            compressed[1],
            HashOrRun::Run {
                count: 3,
                statements_hashes: vec![HashOrRun::Hash(2), HashOrRun::Hash(3), HashOrRun::Hash(1)]
            }
        );
        assert_eq!(compressed[2], HashOrRun::Hash(2));
    }

    #[test]
    fn test_compress_hashes_long_list() {
        // Checks that repeated long list are compressed away
        let mut hashes = Vec::new();
        for _ in 0..10 {
            hashes.extend_from_slice(&[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        }
        let compressed = compress_hashes(&hashes, &TestCostFunctionDefault {}, false);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 10 * 3); // each slice has three runs, and there are 10 slices
        for (i, h) in compressed.iter().enumerate() {
            match h {
                HashOrRun::Run {
                    count,
                    statements_hashes,
                } => {
                    assert_eq!(*count, 4);
                    assert_eq!(statements_hashes.len(), 1);
                    assert_eq!(statements_hashes[0], HashOrRun::Hash(hashes[i * 4]));
                }
                HashOrRun::Hash(_) => {
                    panic!("unexpected Hash in compressed result");
                }
            }
        }
    }

    #[test]
    fn test_compress_recursion() {
        // Verifies that the 'recurse' flag to compress_hashes() works as expected
        let mut hashes = Vec::new();
        for _ in 0..3 {
            hashes.extend_from_slice(&[1, 2, 2, 2, 2, 2]);
        }
        let compressed = compress_hashes(&hashes, &TestCostFunctionDefault {}, false);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 1);
        assert_eq!(
            compressed[0],
            HashOrRun::Run {
                count: 3,
                statements_hashes: vec![
                    HashOrRun::Hash(1),
                    HashOrRun::Hash(2),
                    HashOrRun::Hash(2),
                    HashOrRun::Hash(2),
                    HashOrRun::Hash(2),
                    HashOrRun::Hash(2)
                ]
            }
        );
        let compressed = compress_hashes(&hashes, &TestCostFunction2 {}, true);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 1);
        assert_eq!(
            compressed[0],
            HashOrRun::Run {
                count: 3,
                statements_hashes: vec![
                    HashOrRun::Hash(1),
                    HashOrRun::Run {
                        count: 5,
                        statements_hashes: vec![HashOrRun::Hash(2)]
                    }
                ]
            }
        );
    }

    #[test]
    fn test_compress_hashes_hbar_2384() {
        // Demonstrates that https://zhinst.atlassian.net/browse/HBAR-2384 is fixed.
        let hashes = vec![42, 42, 42, 42, 13, 13, 13, 13, 13];
        let compressed = compress_hashes(&hashes, &TestCostFunctionDefault {}, false);
        assert!(compressed.is_ok());
        let compressed = compressed.unwrap();
        assert_eq!(compressed.len(), 2);
        assert_eq!(
            compressed,
            vec![
                HashOrRun::Run {
                    count: 4,
                    statements_hashes: vec![HashOrRun::Hash(42)]
                },
                HashOrRun::Run {
                    count: 5,
                    statements_hashes: vec![HashOrRun::Hash(13)]
                },
            ]
        );
    }
}
