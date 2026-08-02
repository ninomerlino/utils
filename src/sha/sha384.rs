//! SHA-384 hasher.
//!
//! Shape:
//! - block size: 128 bytes
//! - schedule: 80 `u64` words
//! - digest: 48 bytes
//!
//! SHA-384 reuses the SHA-512 compression function with SHA-384 initial
//! constants and returns the first six state words.

use super::SHA512;

use super::Digest;
use super::HashStream;

/// Stateful SHA-384 compressor backed by the SHA-512 state layout.
pub struct SHA384(SHA512);

impl Default for SHA384 {
    fn default() -> Self {
        Self(SHA512 {
            h0: 0xcbbb9d5dc1059ed8,
            h1: 0x629a292a367cd507,
            h2: 0x9159015a3070dd17,
            h3: 0x152fecd8f70e5939,
            h4: 0x67332667ffc00b31,
            h5: 0x8eb44a8768581511,
            h6: 0xdb0c2e0d64f98fa7,
            h7: 0x47b5481dbefa4fa4,
        })
    }
}

impl HashStream<128, 80, 48, 16, u64> for SHA384 {
    fn build_words(buffer: &[u8]) -> [u64; 80] {
        SHA512::build_words(buffer)
    }

    fn hash_block(&mut self, words: [u64; 80]) {
        self.0.hash_block(words)
    }
}

impl From<SHA384> for Digest<48> {
    fn from(value: SHA384) -> Self {
        let value = value.0;
        let mut digest = [0u8; 48];
        digest[0..8].copy_from_slice(&value.h0.to_be_bytes());
        digest[8..16].copy_from_slice(&value.h1.to_be_bytes());
        digest[16..24].copy_from_slice(&value.h2.to_be_bytes());
        digest[24..32].copy_from_slice(&value.h3.to_be_bytes());
        digest[32..40].copy_from_slice(&value.h4.to_be_bytes());
        digest[40..48].copy_from_slice(&value.h5.to_be_bytes());
        super::Digest(digest)
    }
}
