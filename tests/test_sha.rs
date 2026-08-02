use std::io::Cursor;

use sha::{
    sha512224,
    utils::{Digest, DigestExt},
};
use utils::sha::{HashStream, SHA0, SHA1, SHA224, SHA256, SHA384, SHA512, SHA512_224, SHA512_256};

static TEST_FILES: [&'static str; 1] =
    ["Let me speak Piccolo Let me speak! Let me speak to the people"];

#[test]
fn verify_sha_0() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA0::default().hash_stream(x).unwrap().to_string();
        assert_eq!(string_digest, string_digest);
    }
}

#[test]
fn verify_sha_1() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA1::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha1::Sha1::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_256() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA256::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha256::Sha256::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_224() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA224::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha224::Sha224::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_512() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA512::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha512::Sha512::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_384() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA384::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha384::Sha384::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_512_224() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA512_224::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha512224::Sha512224::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}

#[test]
fn verify_sha_512_256() {
    for test_string in TEST_FILES {
        let x = Cursor::new(test_string.as_bytes());
        let string_digest = SHA512_256::default().hash_stream(x).unwrap().to_string();
        let mut hasher = sha::sha512256::Sha512256::default();
        let out = hasher.digest(test_string.as_bytes()).to_hex();
        assert_eq!(string_digest, out);
    }
}
