//! Utility crate.
//!
//! Current public surface:
//! - [`sha`]: SHA-family hashers, digest storage, and stream hashing support.

pub mod linear_systems;
pub mod rng;
pub mod sha;
pub mod sorting;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("The RNG state cannot be all zero")]
    ZeroRngState,
    #[error("Cannot pick a random position from an empty slice")]
    EmptySlicePick,
}

pub type Result<T> = std::result::Result<T, Error>;
