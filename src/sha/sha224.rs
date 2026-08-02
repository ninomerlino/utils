use super::{Digest, HashStream, SHA256};

pub struct SHA224(SHA256);

impl HashStream<64, 64, 28, 8, u32> for SHA224 {
    fn build_words(buffer: &[u8]) -> [u32; 64] {
        SHA256::build_words(buffer)
    }

    fn hash_block(&mut self, words: [u32; 64]) {
        self.0.hash_block(words)
    }
}

impl Default for SHA224 {
    fn default() -> Self {
        SHA224(SHA256 {
            h0: 0xc1059ed8,
            h1: 0x367cd507,
            h2: 0x3070dd17,
            h3: 0xf70e5939,
            h4: 0xffc00b31,
            h5: 0x68581511,
            h6: 0x64f98fa7,
            h7: 0xbefa4fa4,
        })
    }
}

impl From<SHA224> for Digest<28> {
    fn from(value: SHA224) -> Self {
        let mut digest = [0u8; 28];
        let value = value.0;
        digest[0..4].copy_from_slice(&value.h0.to_be_bytes());
        digest[4..8].copy_from_slice(&value.h1.to_be_bytes());
        digest[8..12].copy_from_slice(&value.h2.to_be_bytes());
        digest[12..16].copy_from_slice(&value.h3.to_be_bytes());
        digest[16..20].copy_from_slice(&value.h4.to_be_bytes());
        digest[20..24].copy_from_slice(&value.h5.to_be_bytes());
        digest[24..28].copy_from_slice(&value.h6.to_be_bytes());
        Digest(digest)
    }
}
