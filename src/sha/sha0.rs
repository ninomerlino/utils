use crate::sha::{Digest, HashStream};

#[derive(Debug, Clone, Copy)]
pub struct SHA0 {
    pub h0: u32,
    pub h1: u32,
    pub h2: u32,
    pub h3: u32,
    pub h4: u32,
}

impl Default for SHA0 {
    fn default() -> Self {
        Self {
            h0: 0x67452301,
            h1: 0xEFCDAB89,
            h2: 0x98BADCFE,
            h3: 0x10325476,
            h4: 0xC3D2E1F0,
        }
    }
}

impl HashStream<64, 80, 20, 8, u32> for SHA0 {
    fn build_words(buffer: &[u8]) -> [u32; 80] {
        let mut words = [0u32; 80];
        for (i, word) in words.iter_mut().enumerate().take(16) {
            let ii = i * 4;
            *word =
                u32::from_be_bytes([buffer[ii], buffer[ii + 1], buffer[ii + 2], buffer[ii + 3]]);
        }
        for i in 16..80 {
            words[i] = words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16];
        }
        words
    }

    fn hash_block(&mut self, words: [u32; 80]) {
        let mut a = self.h0;
        let mut b = self.h1;
        let mut c = self.h2;
        let mut d = self.h3;
        let mut e = self.h4;

        //YAY! first 80 words are ready!
        for (t, word) in words.iter().enumerate() {
            let f: u32;
            let k: u32;

            match t {
                0..=19 => {
                    k = 0x5A827999;
                    f = (b & c) | (!b & d);
                }
                20..=39 => {
                    k = 0x6ED9EBA1;
                    f = b ^ c ^ d;
                }
                40..=59 => {
                    k = 0x8F1BBCDC;
                    f = (b & c) | (b & d) | (c & d);
                }
                60..=79 => {
                    k = 0xCA62C1D6;
                    f = b ^ c ^ d;
                }
                _ => unreachable!(),
            }

            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);

            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        self.h0 = self.h0.wrapping_add(a);
        self.h1 = self.h1.wrapping_add(b);
        self.h2 = self.h2.wrapping_add(c);
        self.h3 = self.h3.wrapping_add(d);
        self.h4 = self.h4.wrapping_add(e);
    }
}

impl From<SHA0> for Digest<20> {
    fn from(value: SHA0) -> Self {
        let mut digest = [0u8; 20];
        digest[0..4].copy_from_slice(&value.h0.to_be_bytes());
        digest[4..8].copy_from_slice(&value.h1.to_be_bytes());
        digest[8..12].copy_from_slice(&value.h2.to_be_bytes());
        digest[12..16].copy_from_slice(&value.h3.to_be_bytes());
        digest[16..20].copy_from_slice(&value.h4.to_be_bytes());
        Digest(digest)
    }
}
