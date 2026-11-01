//! Digest output support.
//!
//! A digest is stored as raw bytes and formatted as lowercase hexadecimal when
//! printed.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{fmt::Display, ops::Deref};

/// Fixed-size hash output.
///
/// `T` is the digest size in bytes. The inner byte array is public so callers
/// can compare, copy, or serialize the raw digest directly.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest<const T: usize>(pub [u8; T]);

/// Converts raw bytes into lowercase hexadecimal text.
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

impl<const T: usize> Display for Digest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&to_hex(self))
    }
}

impl<const T: usize> Deref for Digest<T> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
