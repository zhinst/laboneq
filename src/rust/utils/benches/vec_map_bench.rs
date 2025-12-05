// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::hint::black_box;
use utils::vec_map::VecMap;

fn create_map(n: usize) -> impl Iterator<Item = (i32, i32)> {
    (0..n as i32).map(|i| (i, i * 10))
}

macro_rules! make_benchmark_for_container {
    ($container:ident, $group:expr, $size:expr) => {
        let map: $container<_, _> = create_map($size).collect();
        $group.bench_with_input(
            BenchmarkId::new(stringify!($container), $size),
            &$size,
            |b, &size| {
                b.iter(|| {
                    // Search for various keys including ones that exist and don't exist
                    for i in 0..size as i32 {
                        black_box(map.get(&i));
                    }
                    // Also test some non-existent keys
                    for i in size as i32..(size + 10) as i32 {
                        black_box(map.get(&i));
                    }
                });
            },
        );
    };
}

fn bench_map_lookup(c: &mut Criterion) {
    let sizes = [16, 64, 256];

    let mut group = c.benchmark_group("lookup");

    for &size in &sizes {
        make_benchmark_for_container!(VecMap, group, size);
        make_benchmark_for_container!(HashMap, group, size);
        make_benchmark_for_container!(IndexMap, group, size);
    }

    group.finish();
}

fn bench_single_lookup(c: &mut Criterion) {
    let sizes = [16, 64, 256];

    let mut group = c.benchmark_group("single_lookup");

    for &size in &sizes {
        let vecmap: VecMap<_, _> = create_map(size).collect();

        // Benchmark VecMap single lookup (middle element)
        let key = (size / 2) as i32;
        group.bench_with_input(BenchmarkId::new("vecmap", size), &size, |b, &_size| {
            b.iter(|| {
                black_box(vecmap.get(&key));
            });
        });
    }

    group.finish();
}
fn bench_worst_case_lookup(c: &mut Criterion) {
    let sizes = [16, 64, 256];

    let mut group = c.benchmark_group("worst_case_lookup");

    for &size in &sizes {
        let vecmap: VecMap<_, _> = create_map(size).collect();

        // Benchmark VecMap worst case (last element)
        let key = (size - 1) as i32;
        group.bench_with_input(BenchmarkId::new("vecmap", size), &size, |b, &_size| {
            b.iter(|| {
                black_box(vecmap.get(&key));
            });
        });
    }

    group.finish();
}

fn bench_not_found_lookup(c: &mut Criterion) {
    let sizes = [16, 64, 256];

    let mut group = c.benchmark_group("not_found_lookup");

    for &size in &sizes {
        let vecmap: VecMap<_, _> = create_map(size).collect();

        // Benchmark VecMap not found
        let key = size as i32 + 100;
        group.bench_with_input(BenchmarkId::new("vecmap", size), &size, |b, &_size| {
            b.iter(|| {
                black_box(vecmap.get(&key));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_map_lookup,
    bench_single_lookup,
    bench_worst_case_lookup,
    bench_not_found_lookup
);
criterion_main!(benches);
