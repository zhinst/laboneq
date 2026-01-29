// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_log::diagnostic;

use super::seqc_generator::SeqCGenerator;
use super::seqc_statements::SeqCStatement;
use crate::ir::compilation_job::DeviceKind;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    num::NonZero,
    rc::Rc,
};

// This a number chosen by hunch to allow compression improvements that can be done
// for common experiments with up to 2 nested sweeps, while also preventing the
// compressor to do too many rounds for pathological cases.
// If you have found a worthwile example/motivation, feel free to change this number.
const HASH_COMPRESSION_MAX_ROUNDS: usize = 3;

#[derive(PartialEq, Clone, Debug)]
enum HashOrRun {
    Hash(u64),
    Run(Run),
}

impl Hash for HashOrRun {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            HashOrRun::Hash(h) => h.hash(state),
            HashOrRun::Run(r) => {
                r.count.hash(state);
                for h in &r.word {
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
    statement_by_hash: &'a HashMap<u64, Rc<SeqCStatement>>,
}
impl StatementCostFunction<'_> {
    fn calculate_complexity(&self, item: &HashOrRun) -> u64 {
        match item {
            HashOrRun::Hash(hash) => self.statement_by_hash.get(hash).unwrap().complexity(),
            HashOrRun::Run(run) => {
                run.word
                    .iter()
                    .map(|x| self.calculate_complexity(x))
                    .sum::<u64>()
                    + 2 // add 2 for the Run loop itself
            }
        }
    }
}
impl CostFunction for StatementCostFunction<'_> {
    fn cost_function(&self, run: &Run) -> i64 {
        let mut complexity: i64 = 0;
        for item in run.word.iter() {
            complexity += self.calculate_complexity(item) as i64;
        }
        let cost = -((run.count as i64) - 1) * complexity + 2;

        fn get_first_statement(
            run: &Run,
            statement_by_hash: &HashMap<u64, Rc<SeqCStatement>>,
        ) -> Rc<SeqCStatement> {
            assert!(!run.word.is_empty());
            let first_item = &run.word[0];
            match first_item {
                HashOrRun::Hash(hash) => Rc::clone(statement_by_hash.get(hash).unwrap()),
                HashOrRun::Run(run) => get_first_statement(run, statement_by_hash),
            }
        }

        let first_statement = get_first_statement(run, self.statement_by_hash);
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
    generator_by_hash: &'a HashMap<u64, &'a SeqCGenerator>,
}
impl GeneratorCostFunction<'_> {
    fn calculate_complexity(&self, item: &HashOrRun) -> u64 {
        match item {
            HashOrRun::Hash(hash) => self
                .generator_by_hash
                .get(hash)
                .unwrap()
                .estimate_complexity(),
            HashOrRun::Run(run) => {
                run.word
                    .iter()
                    .map(|x| self.calculate_complexity(x))
                    .sum::<u64>()
                    + 2 // add 2 for the Run loop itself
            }
        }
    }
}
impl CostFunction for GeneratorCostFunction<'_> {
    fn cost_function(&self, run: &Run) -> i64 {
        let mut complexity: i64 = 0;
        for item in run.word.iter() {
            complexity += self.calculate_complexity(item) as i64;
        }
        -((run.count as i64) - 1) * complexity + 2
    }
}

pub(crate) fn to_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Debug, PartialEq)]
struct Run {
    word: Vec<HashOrRun>,
    count: usize,
}
impl Run {
    fn span(&self) -> usize {
        self.word.len() * self.count
    }
}

impl Hash for Run {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
        self.count.hash(state);
    }
}

fn compress_hashes<T: CostFunction>(
    input_hashes: Vec<HashOrRun>,
    cost_function: &T,
    max_rounds: usize,
) -> Vec<HashOrRun> {
    let mut compression_result = input_hashes;
    for _ in 0..max_rounds {
        let compression_result_next_round =
            _compress_hashes_one_round(&compression_result, cost_function);
        match compression_result_next_round {
            None => break,
            Some(compressed) => compression_result = compressed,
        }
    }
    compression_result
}

// Note that we only aim for a reasonable compression, not for "the optimal" one.
// We try to find some good ones quickly, balancing compression efficiency for runtime and memory complexity.
//
// Idea of the compression algorithm:
//
// Find possible compression 'words' by looking for equal tokens (the first token has to agree)
// foreach input_token:
//   if compression is possibly a repeating word
//     compute max repeat of this word as current best
//     check until the end of the current best if there is an overlapping even better possibility
//     when reaching the end of the current best, compress away
//
// Returns the compression result, or None if any compression was not found/done.
fn _compress_hashes_one_round<T: CostFunction>(
    hashes: &[HashOrRun],
    cost_function: &T,
) -> Option<Vec<HashOrRun>> {
    if hashes.len() <= 1 {
        return None;
    }

    let mut compressed = Vec::new();

    // first, we compute `offsets`, which, for each hash in the list, gives
    // the relative position to the next place the same hash appears again
    let offsets: Vec<Option<NonZero<usize>>> = {
        let mut offsets = vec![None; hashes.len()];
        let mut next_seen_map = HashMap::<u64, usize>::new();
        for (i, hash_or_run) in hashes.iter().enumerate().rev() {
            let hash = match hash_or_run {
                HashOrRun::Hash(h) => *h,
                HashOrRun::Run(r) => to_hash(r),
            };
            if let Some(offs) = next_seen_map.insert(hash, i) {
                offsets[i] = NonZero::new(offs - i);
            }
        }
        offsets
    };

    // There are some pathological cases of runtime. For example very long sequences repeated a few times.
    // This number was picked so that even in the known bad cases runtime doesn't exceed a few seconds.
    // We sacrifice finding anything better than this.
    const GOOD_ENOUGH_COST: i64 = -10000;

    let mut runs = HashMap::<u64, Vec<(usize, Run)>>::new(); // hash -> vec[(start, Run)]
    let mut best_run: Option<Run> = None;
    let mut best_run_start: Option<usize> = None;
    let mut best_run_end: Option<usize> = None;
    let mut best_cost = 0;
    let mut uncompressed_start: usize = 0;

    for (index, offset) in offsets.iter().enumerate() {
        if let Some(best_run_start_unwrapped) = best_run_start
            && let Some(best_run_end_unwrapped) = best_run_end
            && index >= best_run_end_unwrapped
        {
            // We've reached the end of the current best_run, without finding anything better.
            // Emit uncompressed items before this run, then emit this run, and reset our state tracking.
            compressed.extend_from_slice(&hashes[uncompressed_start..best_run_start_unwrapped]);
            compressed.push(HashOrRun::Run(best_run.unwrap()));
            uncompressed_start = best_run_end.unwrap();
            runs.clear();
            best_run = None;
            best_run_start = None;
            best_run_end = None;
            best_cost = 0;
        }
        if best_cost < GOOD_ENOUGH_COST {
            // We've found something, and searching deeper might be too expensive
            continue;
        };
        if offset.is_none() {
            // Nothing to compress here.
            continue;
        }
        let offset = offset.unwrap().get();
        let word = &hashes[index..index + offset];
        let first_hash = to_hash(&word);
        let run = runs.get(&first_hash);
        if let Some(runs_for_hash) = run
            && runs_for_hash.iter().any(|(_, r)| r.word != word)
        {
            diagnostic!("hash collision detected, skipping compression round");
            return None;
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
            && word == &hashes[index + run_length * offset..index + (run_length + 1) * offset]
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
    if let Some(best_run) = best_run {
        compressed.extend_from_slice(&hashes[uncompressed_start..best_run_start.unwrap()]);
        compressed.push(HashOrRun::Run(best_run));
        uncompressed_start = best_run_end.unwrap();
    }
    if uncompressed_start == 0 {
        // Have not compressed anything
        None
    } else {
        compressed.extend_from_slice(&hashes[uncompressed_start..]);
        Some(compressed)
    }
}

pub(crate) fn compress_generator(generator: SeqCGenerator) -> SeqCGenerator {
    let mut statement_hashes: Vec<HashOrRun> = Vec::new();
    let mut statement_by_hash = HashMap::<u64, Rc<SeqCStatement>>::new();

    for statement in generator.statements() {
        let hash = statement.to_hash();
        if let Some(old_value) = statement_by_hash.insert(hash, Rc::clone(statement))
            && &old_value != statement
        {
            diagnostic!("hash collision detected, skipping code compression");
            return generator;
        }
        statement_hashes.push(HashOrRun::Hash(hash));
    }
    let mut compressed_gen = generator.create();
    let compressed_statements = compress_hashes(
        statement_hashes,
        &StatementCostFunction {
            statement_by_hash: &statement_by_hash,
        },
        HASH_COMPRESSION_MAX_ROUNDS,
    );

    fn handle_compressed_item(
        item: &HashOrRun,
        statement_by_hash: &HashMap<u64, Rc<SeqCStatement>>,
        generator: &SeqCGenerator,
    ) -> Rc<SeqCStatement> {
        match item {
            HashOrRun::Hash(hash) => Rc::clone(statement_by_hash.get(hash).unwrap()),
            HashOrRun::Run(run) => {
                let mut body = generator.create();
                for item in &run.word {
                    let statement = handle_compressed_item(item, statement_by_hash, generator);
                    body.add_statement(statement);
                }
                let mut loop_gen = generator.create();
                loop_gen.add_repeat(run.count as u64, body);
                Rc::clone(&loop_gen.statements()[0])
            }
        }
    }

    for cs in compressed_statements.iter() {
        let statement = handle_compressed_item(cs, &statement_by_hash, &generator);
        compressed_gen.add_statement(statement);
    }

    compressed_gen
}

pub(crate) fn merge_generators(generators: Vec<SeqCGenerator>) -> SeqCGenerator {
    if generators.is_empty() {
        // Create an empty generator of arbitrary type
        return SeqCGenerator::new(DeviceKind::SHFQA.traits(), true);
    }

    let mut generator_hashes: Vec<HashOrRun> = Vec::new();
    let mut generator_by_hash = HashMap::<u64, &SeqCGenerator>::new();
    let mut collision_detected = false;

    for generator in &generators {
        let hash = to_hash(generator);
        if let Some(old_value) = generator_by_hash.insert(hash, generator)
            && old_value != generator
        {
            collision_detected = true;
        }
        generator_hashes.push(HashOrRun::Hash(hash));
    }
    if collision_detected {
        diagnostic!("hash collision detected, merging generators without compression");
        let mut retval = generators[0].create();
        for g in generators.iter() {
            retval.append_statements_from(g);
        }
        return retval;
    }

    let compressed_generators = compress_hashes(
        generator_hashes,
        &GeneratorCostFunction {
            generator_by_hash: &generator_by_hash,
        },
        HASH_COMPRESSION_MAX_ROUNDS,
    );

    fn handle_compressed_item(
        item: &HashOrRun,
        generator_by_hash: &HashMap<u64, &SeqCGenerator>,
        generator: &SeqCGenerator,
    ) -> SeqCGenerator {
        match item {
            HashOrRun::Hash(hash) => {
                // TODO: Consider using shared pointers here instead of cloning
                (*generator_by_hash.get(hash).unwrap()).clone()
            }
            HashOrRun::Run(run) => {
                let mut body = generator.create();
                for item in run.word.iter() {
                    body.append_statements_from(&handle_compressed_item(
                        item,
                        generator_by_hash,
                        generator,
                    ));
                }
                let mut retval: SeqCGenerator = generator.create();
                retval.add_repeat(run.count as u64, compress_generator(body));
                retval
            }
        }
    }

    let mut retval = generators[0].create();
    let mut did_compress = false;
    for item in compressed_generators.iter() {
        retval.append_statements_from(&handle_compressed_item(item, &generator_by_hash, &retval));
        if let HashOrRun::Run(run) = item
            && run.count > 1
        {
            did_compress = true;
        }
    }
    if did_compress {
        // 2nd pass on the merged generator, finding patterns that partially span across
        // multiple of the original parts.
        retval = compress_generator(retval);
    }
    retval
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

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
        let hashes = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
            .into_iter()
            .map(HashOrRun::Hash)
            .collect::<Vec<_>>();
        let compressed = compress_hashes(hashes.clone(), &TestCostFunctionDefault {}, 1);
        assert_eq!(compressed.len(), 3);

        assert_eq!(
            compressed[0],
            HashOrRun::Run(Run {
                count: 3,
                word: vec![HashOrRun::Hash(1), HashOrRun::Hash(2), HashOrRun::Hash(3)]
            })
        );
        assert_eq!(compressed[1], HashOrRun::Hash(1));
        assert_eq!(compressed[2], HashOrRun::Hash(2));

        let compressed = compress_hashes(hashes, &TestCostFunction2 {}, 1);
        assert_eq!(compressed.len(), 3);
        assert_eq!(compressed[0], HashOrRun::Hash(1));
        assert_eq!(
            compressed[1],
            HashOrRun::Run(Run {
                count: 3,
                word: vec![HashOrRun::Hash(2), HashOrRun::Hash(3), HashOrRun::Hash(1)]
            })
        );
        assert_eq!(compressed[2], HashOrRun::Hash(2));
    }

    #[test]
    fn test_compress_hashes_long_list() {
        // Checks that repeated long list are compressed away
        let mut hashes = Vec::new();
        for _ in 0..10 {
            hashes.extend(
                vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
                    .into_iter()
                    .map(HashOrRun::Hash),
            );
        }
        let compressed = compress_hashes(hashes.clone(), &TestCostFunctionDefault {}, 1);
        assert_eq!(compressed.len(), 10 * 3); // each slice has three runs, and there are 10 slices
        for (i, h) in compressed.iter().enumerate() {
            match h {
                HashOrRun::Run(run) => {
                    assert_eq!(run.count, 4);
                    assert_eq!(run.word.len(), 1);
                    assert_eq!(run.word[0], hashes[i * 4]);
                }
                HashOrRun::Hash(_) => {
                    panic!("unexpected Hash in compressed result");
                }
            }
        }
    }

    #[test]
    fn test_compress_hashes_hbar_2384() {
        // Demonstrates that https://zhinst.atlassian.net/browse/HBAR-2384 is fixed.
        let hashes = vec![42, 42, 42, 42, 13, 13, 13, 13, 13]
            .into_iter()
            .map(HashOrRun::Hash)
            .collect::<Vec<_>>();
        let compressed = compress_hashes(hashes, &TestCostFunctionDefault {}, 1);

        assert_eq!(compressed.len(), 2);
        assert_eq!(
            compressed,
            vec![
                HashOrRun::Run(Run {
                    count: 4,
                    word: vec![HashOrRun::Hash(42)]
                }),
                HashOrRun::Run(Run {
                    count: 5,
                    word: vec![HashOrRun::Hash(13)]
                }),
            ]
        );
    }

    #[test]
    fn test_compress_hashes_2_rounds() {
        let hashes = [1, 1, 1, 1, 2, 2, 2, 2]
            .repeat(3)
            .into_iter()
            .map(HashOrRun::Hash)
            .collect::<Vec<_>>();
        let compressed_1_round = compress_hashes(hashes.clone(), &TestCostFunctionDefault {}, 1);
        let compressed_2_rounds = compress_hashes(hashes, &TestCostFunctionDefault {}, 2);

        assert_eq!(
            compressed_1_round,
            vec![
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(1)],
                    count: 4
                }),
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(2)],
                    count: 4
                }),
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(1)],
                    count: 4
                }),
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(2)],
                    count: 4
                }),
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(1)],
                    count: 4
                }),
                HashOrRun::Run(Run {
                    word: vec![HashOrRun::Hash(2)],
                    count: 4
                }),
            ]
        );

        assert_eq!(
            compressed_2_rounds,
            vec![HashOrRun::Run(Run {
                word: vec![
                    HashOrRun::Run(Run {
                        word: vec![HashOrRun::Hash(1)],
                        count: 4
                    }),
                    HashOrRun::Run(Run {
                        word: vec![HashOrRun::Hash(2)],
                        count: 4
                    }),
                ],
                count: 3
            })]
        );
    }

    struct CountingCostFunction {
        // Wraps another cost function
        cost_fn: Box<dyn CostFunction>,
        // Tracks the costs of computing the costs, by assuming that it is linear to the lenght of the word
        cumulative_costs: RefCell<usize>,
    }
    impl CostFunction for CountingCostFunction {
        fn cost_function(&self, run: &Run) -> i64 {
            *self.cumulative_costs.borrow_mut() += run.word.len();
            self.cost_fn.cost_function(run)
        }
    }
    impl CountingCostFunction {
        fn new() -> Self {
            Self {
                cumulative_costs: RefCell::new(0),
                cost_fn: Box::new(TestCostFunctionDefault {}),
            }
        }
    }

    #[test]
    fn test_compress_hashes_very_many_short_sequence() {
        // A previous implementation was O((number of repeats) * (total length)), so 1e5 repeats would take very long.
        // This test doesn't check much functional behaviour, but would explode in runtime if "quadratic" runtime would be reintroduced.
        const MANY_LOOPS: u64 = 100 * 1000;
        let mut hashes = Vec::new();
        for i in 0..MANY_LOOPS {
            for _ in 0..4 {
                hashes.push(HashOrRun::Hash(i));
            }
        }
        let cost_counter = CountingCostFunction::new();
        let compressed = compress_hashes(hashes, &cost_counter, 1);
        assert_eq!(compressed.len(), MANY_LOOPS as usize);
        assert_eq!(*cost_counter.cumulative_costs.borrow(), MANY_LOOPS as usize);
    }

    #[test]
    fn test_compress_hashes_very_long_sequences() {
        // A previous implementation was O((length of repeat)^2) in memory needs, so 1e5 repeat length would be expensive.
        // This test doesn't check much functional behaviour, but would explode in memory if quadratic runtime would be reintroduced.
        const MANY_HASHES: u64 = 100 * 1000;
        let mut hashes = Vec::new();
        for _ in 0..4 {
            for i in 0..MANY_HASHES {
                hashes.push(HashOrRun::Hash(i));
            }
        }
        let cost_counter = CountingCostFunction::new();
        let compressed = compress_hashes(hashes, &cost_counter, 1);
        assert_eq!(compressed.len(), 1);
        assert_eq!(
            *cost_counter.cumulative_costs.borrow(),
            MANY_HASHES as usize
        );
    }
}
