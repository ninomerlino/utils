use super::SHA512;

use super::Digest;
use super::HashStream;

pub struct SHA512224(SHA512);

impl Default for SHA512224 {
    fn default() -> Self {
        Self(SHA512 {
            h0: 0x8C3D37C819544DA2,
            h1: 0x73E1996689DCD4D6,
            h2: 0x1DFAB7AE32FF9C82,
            h3: 0x679DD514582F9FCF,
            h4: 0x0F6D2B697BD44DA8,
            h5: 0x77E36F7304C48942,
            h6: 0x3F9D85A86A1D36C8,
            h7: 0x1112E6AD91D692A1,
        })
    }
}

impl HashStream<128, 80, 28, 16, u64> for SHA512224 {
    fn build_words(buffer: &[u8]) -> [u64; 80] {
        SHA512::build_words(buffer)
    }

    fn hash_block(&mut self, words: [u64; 80]) {
        self.0.hash_block(words)
    }
}

impl From<SHA512224> for Digest<28> {
    fn from(value: SHA512224) -> Self {
        let value = value.0;
        let mut digest = [0u8; 28];
        digest[0..8].copy_from_slice(&value.h0.to_be_bytes());
        digest[8..16].copy_from_slice(&value.h1.to_be_bytes());
        digest[16..24].copy_from_slice(&value.h2.to_be_bytes());
        digest[24..28].copy_from_slice(&(value.h3.to_be_bytes()[0..4]));
        super::Digest(digest)
    }
}
