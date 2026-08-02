use crate::sha::{Digest, HashStream, sha0::SHA0};

#[derive(Debug, Clone, Copy)]
pub struct SHA1(pub SHA0);

impl Default for SHA1 {
    fn default() -> Self {
        let s = SHA0 {
            h0: 0x67452301,
            h1: 0xEFCDAB89,
            h2: 0x98BADCFE,
            h3: 0x10325476,
            h4: 0xC3D2E1F0,
        };
        Self(s)
    }
}

impl HashStream<64, 80, 20> for SHA1 {
    fn hash_block(&mut self, words: [u32; 80]) {
        self.0.hash_block(words);
    }

    fn build_words(buffer: &[u8]) -> [u32; 80] {
        let mut words = [0u32; 80];
        for i in 0..16 {
            let ii = i * 4;
            words[i] =
                u32::from_be_bytes([buffer[ii], buffer[ii + 1], buffer[ii + 2], buffer[ii + 3]]);
        }
        for i in 16..80 {
            words[i] = (words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16]).rotate_left(1);
        }
        words
    }
}

impl From<SHA1> for Digest<20> {
    fn from(value: SHA1) -> Self {
        Self::from(value.0)
    }
}
