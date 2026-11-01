//! SHA-family hashing support.
//!
//! This module contains:
//! - concrete hashers: [`SHA0`], [`SHA1`], [`SHA224`], [`SHA256`],
//!   [`SHA384`], [`SHA512`], [`SHA512_224`], and [`SHA512_256`]
//! - [`Digest`], a fixed-size byte container with hexadecimal formatting
//! - [`HashStream`], the shared stream driver for block-based hashing
//!
//! The hashers are one-shot state values: create the default state, call
//! [`HashStream::hash_stream`], and consume the state into a digest.

mod digest;
mod sha0;
mod sha1;
mod sha224;
mod sha256;
mod sha384;
mod sha512;
mod sha512224;
mod sha512256;

pub use digest::Digest;
pub use sha0::SHA0;
pub use sha1::SHA1;
pub use sha224::SHA224;
pub use sha256::SHA256;
pub use sha384::SHA384;
pub use sha512::SHA512;
pub use sha512224::SHA512224 as SHA512_224;
pub use sha512256::SHA512256 as SHA512_256;

use std::io::{self, ErrorKind, Read};

/// Word-size marker used by the shared SHA machinery.
///
/// Only `u32` and `u64` are valid SHA schedule words in this crate.
pub trait WordType {}

impl WordType for u32 {}
impl WordType for u64 {}

/// Common interface for block-based SHA hashers.
///
/// Implementors supply two pieces:
/// - `build_words`: expand one padded byte block into schedule words
/// - `hash_block`: compress one schedule into the current hash state
///
/// The default stream driver supplies the shared work: block reads, SHA padding,
/// length encoding, and final conversion into [`Digest`].
///
/// Const parameters:
/// - `BLOCK_SIZE`: input block size in bytes
/// - `WORD_COUNT`: expanded schedule length
/// - `DIGEST_SIZE`: digest size in bytes
/// - `LENGTH_SIZE`: encoded message length size in bytes
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
    /// Hashes every byte produced by `stream`.
    ///
    /// The method consumes the hasher state and returns the final digest.
    ///
    /// # Errors
    ///
    /// Returns stream I/O errors directly. It also returns `InvalidInput` for an
    /// impossible padding configuration and `InvalidData` when the message length
    /// cannot be represented by the selected SHA variant.
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

    /// Expands one padded byte block into schedule words.
    fn build_words(buffer: &[u8]) -> [WordSize; WORD_COUNT];

    /// Mixes one expanded schedule into the current hash state.
    fn hash_block(&mut self, words: [WordSize; WORD_COUNT]);

    /// Fills one block as far as the stream allows, retrying interrupted reads.
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

    /// Writes the message length, in bits, into the final padding field.
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
