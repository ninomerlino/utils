use utils::rng::{PRNG, Xoshiro256};
use utils::Error;

#[test]
fn xoshiro256_generates_expected_sequence() {
    let seed = [
        0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 3,
        0, 0, 0, 0, 0, 0, 0, 4,
    ];

    let mut rng = Xoshiro256::new(&seed).unwrap();
    let actual: Vec<u64> = (0..10).map(|_| rng.random()).collect();

    let expected = [
        41_943_041,
        58_720_359,
        3_588_806_011_781_223,
        3_591_011_842_654_386,
        9_228_616_714_210_784_205,
        9_973_669_472_204_895_162,
        14_011_001_112_246_962_877,
        12_406_186_145_184_390_807,
        15_849_039_046_786_891_736,
        10_450_023_813_501_588_000,
    ];

    assert_eq!(actual, expected);
}

#[test]
fn xoshiro256_rejects_zero_state() {
    assert!(matches!(Xoshiro256::new(&[]), Err(Error::ZeroRngState)));
}
