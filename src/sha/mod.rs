mod digest;
mod sha0;
mod sha1;
mod sha2;

use std::io::Read;

pub use digest::Digest;
pub use sha0::SHA0;
pub use sha1::SHA1;
pub use sha2::{SHA224, SHA256};

pub trait HashStream<const BLOCK_SIZE: usize, const WORD_COUNT: usize, const DIGEST_SIZE: usize>
where
    Self: Into<Digest<DIGEST_SIZE>>,
{
    /// Hash a stream of bytes using this hash function.
    fn hash_stream(mut self, mut stream: impl Read) -> Digest<DIGEST_SIZE> {
        let mut total_size = 0;

        loop {
            let mut buffer = [0u8; BLOCK_SIZE];
            let size_read = stream.read(&mut buffer).expect("Failed to read file");
            total_size += size_read;

            if size_read < BLOCK_SIZE {
                let size_padding = (total_size * 8).to_be_bytes();
                if total_size as usize % BLOCK_SIZE == 0 {
                    let mut padding_block = [0u8; BLOCK_SIZE];
                    padding_block[0] = 0x80;
                    padding_block[BLOCK_SIZE - 8..].copy_from_slice(&size_padding);
                    let words = Self::build_words(&padding_block);
                    self.hash_block(words);
                } else if BLOCK_SIZE - size_read < 9 {
                    buffer[size_read] = 0x80;
                    let mut padding_block = [0u8; BLOCK_SIZE];
                    padding_block[BLOCK_SIZE - 8..].copy_from_slice(&size_padding);
                    let words = Self::build_words(&buffer);
                    self.hash_block(words);
                    let words = Self::build_words(&padding_block);
                    self.hash_block(words);
                } else {
                    buffer[size_read] = 0x80;
                    buffer[BLOCK_SIZE - 8..].copy_from_slice(&size_padding);
                    let words = Self::build_words(&buffer);
                    self.hash_block(words);
                }
                break;
            }
            let words = Self::build_words(&buffer);
            self.hash_block(words);
        }
        return self.into();
    }
    fn build_words(buffer: &[u8]) -> [u32; WORD_COUNT];
    fn hash_block(&mut self, words: [u32; WORD_COUNT]);
}
