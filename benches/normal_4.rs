use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use utils::rng::{DistributionMany, DistributionSimd4, Distributions, Xoshiro256};

fn normal_four_scalar(rng: &mut impl Distributions) -> [f64; 4] {
    std::array::from_fn(|_| rng.normal())
}

fn benchmark_normal_four(c: &mut Criterion) {
    c.bench_function("normal called four times", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(normal_four_scalar(&mut rng)));
    });

    c.bench_function("normal_multi(4)", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(rng.normal_many(4)));
    });

    c.bench_function("normal_4", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(rng.normal_4()));
    });
}

criterion_group!(benches, benchmark_normal_four);
criterion_main!(benches);
