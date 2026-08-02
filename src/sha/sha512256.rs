use super::SHA512;

use super::Digest;
use super::HashStream;

pub struct SHA512256(SHA512);

impl Default for SHA512256 {
    fn default() -> Self {
        Self(SHA512 {
            h0: 0x22312194FC2BF72C,
            h1: 0x9F555FA3C84C64C2,
            h2: 0x2393B86B6F53B151,
            h3: 0x963877195940EABD,
            h4: 0x96283EE2A88EFFE3,
            h5: 0xBE5E1E2553863992,
            h6: 0x2B0199FC2C85B8AA,
            h7: 0x0EB72DDC81C52CA2,
        })
    }
}

impl HashStream<128, 80, 32, 16, u64> for SHA512256 {
    fn build_words(buffer: &[u8]) -> [u64; 80] {
        SHA512::build_words(buffer)
    }

    fn hash_block(&mut self, words: [u64; 80]) {
        self.0.hash_block(words)
    }
}

impl From<SHA512256> for Digest<32> {
    fn from(value: SHA512256) -> Self {
        let value = value.0;
        let mut digest = [0u8; 32];
        digest[0..8].copy_from_slice(&value.h0.to_be_bytes());
        digest[8..16].copy_from_slice(&value.h1.to_be_bytes());
        digest[16..24].copy_from_slice(&value.h2.to_be_bytes());
        digest[24..32].copy_from_slice(&value.h3.to_be_bytes());
        super::Digest(digest)
    }
}
