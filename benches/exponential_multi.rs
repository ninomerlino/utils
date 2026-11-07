use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use utils::rng::{DistributionSimd2, DistributionSimd4, Distributions, Xoshiro256};
use wide::{f64x2, f64x4};

fn scalar_exponential(rng: &mut impl Distributions, n: usize, rate: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);

    while out.len() < n {
        out.push(rng.exponential(rate));
    }

    out
}

fn simd_exponential(rng: &mut impl DistributionSimd4, n: usize, rate: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let rate = f64x4::splat(rate);

    while out.len() < n {
        let samples = (-rng.uniform_4().ln() / rate).to_array();
        let remaining = n - out.len();
        out.extend_from_slice(&samples[..remaining.min(samples.len())]);
    }

    out
}

fn small_simd_exponential(rng: &mut impl DistributionSimd2, n: usize, rate: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let rate = f64x2::splat(rate);

    while out.len() < n {
        let samples = (-rng.uniform_2().ln() / rate).to_array();
        let remaining = n - out.len();
        out.extend_from_slice(&samples[..remaining.min(samples.len())]);
    }

    out
}

fn benchmark_exponential_distributions(c: &mut Criterion) {
    c.bench_function("scalar exponential multi", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(scalar_exponential(&mut rng, 1000, 1.0)));
    });

    c.bench_function("simd exponential multi", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(simd_exponential(&mut rng, 1000, 1.0)));
    });

    c.bench_function("small simd exponential multi", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(small_simd_exponential(&mut rng, 1000, 1.0)));
    });

    c.bench_function("scalar exponential one", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(scalar_exponential(&mut rng, 1, 1.0)));
    });

    c.bench_function("simd exponential one", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(simd_exponential(&mut rng, 1, 1.0)));
    });

    c.bench_function("small simd exponential one", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(small_simd_exponential(&mut rng, 1, 1.0)));
    });
}

criterion_group!(benches, benchmark_exponential_distributions);
criterion_main!(benches);
