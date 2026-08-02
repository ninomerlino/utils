mod digest;
mod sha0;
mod sha1;
mod sha224;
mod sha256;
mod sha512;

pub use digest::Digest;
pub use sha0::SHA0;
pub use sha1::SHA1;
pub use sha224::SHA224;
pub use sha256::SHA256;
pub use sha512::SHA512;

use std::io::{self, ErrorKind, Read};

pub trait WordType {}

impl WordType for u32 {}
impl WordType for u64 {}

pub trait HashStream<
    const BLOCK_SIZE: usize,
    const WORD_COUNT: usize,
    const DIGEST_SIZE: usize,
    const LENGTH_SIZE: usize,
    WordSize,
> where
    Self: Into<Digest<DIGEST_SIZE>>,
    WordSize: WordType,
{
    /// Hash a stream of bytes using this hash function.
    fn hash_stream(mut self, mut stream: impl Read) -> io::Result<Digest<DIGEST_SIZE>> {
        if LENGTH_SIZE != 8 && LENGTH_SIZE != 16 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "SHA padding uses an 8-byte or 16-byte length field",
            ));
        }

        if LENGTH_SIZE >= BLOCK_SIZE {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "the SHA length field must fit inside a block",
            ));
        }

        let mut total_len = 0u128;

        loop {
            let mut block = [0u8; BLOCK_SIZE];
            let size_read = Self::read_block(&mut stream, &mut block)?;
            total_len = total_len
                .checked_add(size_read as u128)
                .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "message length overflow"))?;

            if size_read == BLOCK_SIZE {
                let words = Self::build_words(&block);
                self.hash_block(words);
                continue;
            }

            block[size_read] = 0x80;

            if size_read + 1 + LENGTH_SIZE > BLOCK_SIZE {
                let words = Self::build_words(&block);
                self.hash_block(words);
                block = [0u8; BLOCK_SIZE];
            }

            Self::write_message_len(total_len, &mut block[BLOCK_SIZE - LENGTH_SIZE..])?;
            let words = Self::build_words(&block);
            self.hash_block(words);
            break;
        }

        Ok(self.into())
    }

    fn build_words(buffer: &[u8]) -> [WordSize; WORD_COUNT];

    fn hash_block(&mut self, words: [WordSize; WORD_COUNT]);

    fn read_block(stream: &mut impl Read, block: &mut [u8]) -> io::Result<usize> {
        let mut filled = 0;

        while filled < block.len() {
            match stream.read(&mut block[filled..]) {
                Ok(0) => break,
                Ok(size_read) => filled += size_read,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }

        Ok(filled)
    }

    fn write_message_len(total_len_bytes: u128, output: &mut [u8]) -> io::Result<()> {
        if output.len() != LENGTH_SIZE {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "length output buffer has invalid size",
            ));
        }

        let bit_len = total_len_bytes
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "message length overflow"))?;

        if LENGTH_SIZE < 16 {
            let max_bit_len = 1u128 << (LENGTH_SIZE * 8);
            if bit_len >= max_bit_len {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "message length too large for configured SHA length field",
                ));
            }
        }

        let bit_len_bytes = bit_len.to_be_bytes();
        output.copy_from_slice(&bit_len_bytes[bit_len_bytes.len() - LENGTH_SIZE..]);
        Ok(())
    }
}
