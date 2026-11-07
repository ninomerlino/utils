//! Pseudo-random number generation utilities.
//!
//! This module provides the [`PRNG`] interface and a [`Xoshiro256`] generator.
//! `Xoshiro256` is a fast, non-cryptographic generator intended for simulations,
//! randomized algorithms, and test data. It must not be used for secrets,
//! passwords, or security tokens.

use std::f64;

use wide::{f64x2, f64x4, u64x2, u64x4};

use crate::Error::{EmptySlicePick, ZeroRngState};
use crate::Result;

static E2 : f64x2 = f64x2::splat(std::f64::consts::E);
static E4 : f64x4 = f64x4::splat(std::f64::consts::E);

/// Common operations provided by a pseudo-random number generator.
pub trait PRNG {
    /// Returns a random `u64`.
    fn random(&mut self) -> u64;

    /// Returns a random floating-point value derived from [`Self::random`].
    ///
    /// The result is approximately in the interval `[0.0, 1.0]`. This method
    /// is intended for basic use; callers requiring a particular distribution
    /// should apply an appropriate transformation.
    #[inline]
    fn random_percentage(&mut self) -> f64 {
        return (self.random() as f64) / (u64::MAX as f64);
    }

    /// Returns a random index into `data`.
    ///
    /// The returned index is relative to the supplied slice. An empty slice
    /// cannot provide an index and returns [`crate::Error::EmptySlicePick`].
    #[inline]
    fn random_pick<T>(&mut self, data: &[T]) -> Result<usize> {
        if data.is_empty() {
            return Err(EmptySlicePick);
        }
        return Ok((self.random() as usize) % data.len());
    }

    /// Shuffles `data` in place using this generator.
    fn random_shuffle<T>(&mut self, data: &mut [T]) {
        if data.is_empty() {
            return;
        }
        for i in 0..data.len() - 1 {
            let r = self.random_pick(&mut data[i..]).unwrap() + i;
            data.swap(i, r);
        }
    }

    /// Returns a uniformly distributed `f64` in the open interval `(0, 1)`.
    ///
    /// The value is built from 52 random bits and scaled into the open interval
    /// `(0, 1)`. The open interval is useful for transformations that apply
    /// `ln`, since the result can never be exactly zero or one.
    fn uniform(&mut self) -> f64 {
        let bits = self.random() >> 12;
        (bits as f64 + 1.0) / ((1u64 << 52) + 1) as f64
    }

    /// Returns four uniformly distributed `f64` values in the open interval
    /// `(0, 1)`, using four successive outputs from the generator.
    fn uniform_2(&mut self) -> f64x2 {
        let bits: u64x2 = u64x2::new([self.random(), self.random()]) >> 12;
        let values = f64x2::new(bits.to_array().map(|value| value as f64));
        (values + 1.0) / f64x2::splat(((1u64 << 52) + 1) as f64)
    }

    /// Returns four uniformly distributed `f64` values in the open interval
    /// `(0, 1)`, using four successive outputs from the generator.
    fn uniform_4(&mut self) -> f64x4 {
        let bits: u64x4 =
            u64x4::new([self.random(), self.random(), self.random(), self.random()]) >> 12;
        let values = f64x4::new(bits.to_array().map(|value| value as f64));
        (values + 1.0) / f64x4::splat(((1u64 << 52) + 1) as f64)
    }

    /// Returns a standard-normal value with mean `0` and standard deviation
    /// `1`, using ratio-of-uniforms rejection sampling.
    fn normal(&mut self) -> f64 {
        let e =
        loop {
            let u = self.uniform();
            let v = (2.0 * self.uniform() - 1.0) * (2.0 / f64::consts::E).sqrt();

            if v * v <= -4.0 * u * u * u.ln() {
                return v / u;
            }
        }
    }

    /// Returns `true` with probability `probability` and `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `probability` is outside the inclusive range `0.0..=1.0`.
    fn bernoulli(&mut self, probability: f64) -> bool {
        assert!((0.0..=1.0).contains(&probability));

        self.uniform() < probability
    }

    /// Returns an exponentially distributed waiting time with the given rate.
    ///
    /// The expected value is `1.0 / rate`.
    ///
    /// # Panics
    ///
    /// Panics if `rate` is not positive.
    fn exponential(&mut self, rate: f64) -> f64 {
        assert!(rate > 0.0);
        let u = self.uniform();
        -u.ln() / rate
    }

    /// Returns the number of failures before the first success in independent
    /// Bernoulli trials with the given success probability.
    ///
    /// # Panics
    ///
    /// Panics if `probability` is not in `(0.0, 1.0]`.
    fn geometric(&mut self, probability: f64) -> u64 {
        assert!((0.0..=1.0).contains(&probability));
        assert!(probability > 0.0);

        let u = self.uniform();
        let failure_probability = 1.0 - probability;

        (u.ln() / failure_probability.ln()).floor() as u64
    }

    /// Returns a Poisson-distributed event count with the given expected rate.
    ///
    /// This implementation uses Knuth's product method and is best suited to
    /// small or moderate rates. Its running time grows approximately with
    /// `rate`.
    ///
    /// # Panics
    ///
    /// Panics if `rate` is negative.
    fn poisson(&mut self, rate: f64) -> u64 {
        assert!(rate >= 0.0);

        if rate == 0.0 {
            return 0;
        }

        let limit = (-rate).exp();
        let mut probability = 1.0;
        let mut count = 0;

        loop {
            probability *= self.uniform();

            if probability <= limit {
                return count;
            }

            count += 1;
        }
    }

    /// Returns the number of successes in `trials` independent Bernoulli
    /// trials with the given success probability.
    ///
    /// # Panics
    ///
    /// Panics if `probability` is outside the inclusive range `0.0..=1.0`.
    fn binomial(&mut self, trials: u64, probability: f64) -> u64 {
        assert!((0.0..=1.0).contains(&probability));

        (0..trials).filter(|_| self.uniform() < probability).count() as u64
    }

    fn categorical(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty());
        assert!(weights.iter().all(|weight| *weight >= 0.0));

        let total: f64 = weights.iter().sum();
        assert!(total.is_finite() && total > 0.0);

        let target = self.uniform() * total;
        let mut cumulative = 0.0;

        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;

            if target < cumulative {
                return index;
            }
        }

        // Protects against floating-point rounding.
        weights.len() - 1
    }

    fn log_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        assert!(std_dev >= 0.0);

        (mean + std_dev * self.normal()).exp()
    }

    fn gamma(&mut self, shape: f64, rate: f64) -> f64 {
        assert!(shape > 0.0);
        assert!(rate > 0.0);

        if shape < 1.0 {
            let sample = self.gamma(shape + 1.0, rate);
            return sample * self.uniform().powf(1.0 / shape);
        }

        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.normal();
            let v = 1.0 + c * x;

            if v <= 0.0 {
                continue;
            }

            let v = v * v * v;
            let u = self.uniform();

            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v / rate;
            }

            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v / rate;
            }
        }
    }

    fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        assert!(alpha > 0.0);
        assert!(beta > 0.0);

        let x = self.gamma(alpha, 1.0);
        let y = self.gamma(beta, 1.0);

        x / (x + y)
    }
}

/// A fast xoshiro256++ pseudo-random number generator.
///
/// The generator has a 256-bit internal state and is not cryptographically
/// secure. The seed is interpreted as up to 32 big-endian bytes; shorter seeds
/// are zero-padded and longer seeds are truncated.
pub struct Xoshiro256 {
    state: [u64; 4],
}

impl Default for Xoshiro256 {
    /// Creates a generator seeded from the operating system's random source.
    fn default() -> Self {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("Could not get random seed from OS");
        Self::new(&seed).expect("Random Seed generated by OS is all zero")
    }
}

impl Xoshiro256 {
    /// Creates a generator from a byte seed.
    ///
    /// Returns [`crate::Error::ZeroRngState`] if the resulting 256-bit state
    /// is all zeroes.
    pub fn new(seed: &[u8]) -> Result<Self> {
        let mut state = [[0_u8; 8]; 4];
        for i in 0..seed.len().min(32) {
            state[i / 8][i % 8] = seed[i];
        }

        let state = [
            u64::from_be_bytes(state[0]),
            u64::from_be_bytes(state[1]),
            u64::from_be_bytes(state[2]),
            u64::from_be_bytes(state[3]),
        ];

        if state.iter().all(|&word| word == 0) {
            return Err(ZeroRngState);
        }

        Ok(Self { state })
    }
}

impl PRNG for Xoshiro256 {
    fn random(&mut self) -> u64 {
        let mut state = self.state;

        let result = state[0]
            .wrapping_add(state[3])
            .rotate_left(23)
            .wrapping_add(state[0]);

        let temp = state[1] << 17;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= temp;
        state[3] = state[3].rotate_left(45);

        self.state = state;
        result
    }
}
