use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use utils::rng::{DistributionSimd2, DistributionSimd4, Distributions, Xoshiro256};
use wide::{f64x2, f64x4};
/// scalar normal           time:   [21.945 ns 21.996 ns 22.050 ns]
/// Found 5 outliers among 100 measurements (5.00%)
///   3 (3.00%) low mild
///   2 (2.00%) high mild
///
/// simd normal             time:   [28.402 ns 28.454 ns 28.509 ns]
/// Found 1 outliers among 100 measurements (1.00%)
///   1 (1.00%) high mild
///
/// small simd normal       time:   [24.129 ns 24.174 ns 24.222 ns]
/// Found 4 outliers among 100 measurements (4.00%)
///   4 (4.00%) high mild
fn scalar_normal(rng: &mut impl Distributions, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let factor = (2.0 / std::f64::consts::E).sqrt();
    while out.len() < n {
        let u = rng.uniform();
        let v = (2.0 * rng.uniform() - 1.0) * factor;

        if v * v <= -4.0 * u * u * u.ln() {
            out.push(v / u);
        }
    }
    out
}

fn simd_normal(rng: &mut impl DistributionSimd4, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let factor = f64x4::splat((2.0 / std::f64::consts::E).sqrt());

    while out.len() < n {
        let u = rng.uniform_4();
        let v = (2.0 * rng.uniform_4() - 1.0) * factor;

        let mask = (v * v).simd_le(-4.0 * u * u * u.ln()).to_bitmask();

        let samples = (v / u).to_array();
        let mut remaining = mask;

        while remaining != 0 && out.len() < n {
            let lane = remaining.trailing_zeros() as usize;
            out.push(samples[lane]);

            // Clear the lowest accepted lane.
            remaining &= remaining - 1;
        }
    }

    out
}

fn small_simd_normal(rng: &mut impl DistributionSimd2, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    let factor = f64x2::splat((2.0 / std::f64::consts::E).sqrt());

    while out.len() < n {
        let u = rng.uniform_2();
        let v = (2.0 * rng.uniform_2() - 1.0) * factor;

        let mask = (v * v).simd_le(-4.0 * u * u * u.ln()).to_bitmask();

        let samples = (v / u).to_array();
        let mut remaining = mask;

        while remaining != 0 && out.len() < n {
            let lane = remaining.trailing_zeros() as usize;
            out.push(samples[lane]);

            // Clear the lowest accepted lane.
            remaining &= remaining - 1;
        }
    }

    out
}

fn benchmark_normal_distributions(c: &mut Criterion) {
    c.bench_function("scalar normal multi", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(scalar_normal(&mut rng, 1000)));
    });

    c.bench_function("simd normal multi", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(simd_normal(&mut rng, 1000)));
    });

    c.bench_function("small simd normal", |b| {
        let mut rng = Xoshiro256::new(&[1; 32]).unwrap();

        b.iter(|| black_box(small_simd_normal(&mut rng, 1000)));
    });
}

criterion_group!(benches, benchmark_normal_distributions);
criterion_main!(benches);
