use utils::linear_systems::Vector;

fn assert_approx_eq(left: f64, right: f64) {
    assert!(
        (left - right).abs() <= 1e-12,
        "left: {left}, right: {right}"
    );
}

fn assert_vector_approx_eq<const S: usize>(left: Vector<f64, S>, right: [f64; S]) {
    for (actual, expected) in left.iter().zip(right) {
        assert_approx_eq(*actual, expected);
    }
}

#[test]
fn default_creates_zero_vector() {
    assert_eq!(Vector::<i32, 3>::default(), Vector::from([0, 0, 0]));
}

#[test]
fn from_array_preserves_values() {
    assert_eq!(Vector::from([1, 2, 3]), Vector::from([1, 2, 3]));
}

#[test]
fn from_scalar_fills_every_component() {
    assert_eq!(Vector::<i32, 3>::from(7), Vector::from([7, 7, 7]));
}

#[test]
fn unit_fills_every_component_with_one() {
    assert_eq!(Vector::<i32, 3>::unit(), Vector::from([1, 1, 1]));
}

#[test]
fn len_returns_const_size() {
    assert_eq!(Vector::from([1, 2, 3, 4]).len(), 4);
}

#[test]
fn iter_visits_components_in_order() {
    let values: Vec<_> = Vector::from([1, 2, 3]).iter().copied().collect();
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn iter_mut_allows_component_updates() {
    let mut vector = Vector::from([1, 2, 3]);

    for value in vector.iter_mut() {
        *value *= 2;
    }

    assert_eq!(vector, Vector::from([2, 4, 6]));
}

#[test]
fn sum_adds_components() {
    assert_eq!(Vector::from([1, 2, 3]).sum(), 6);
}

#[test]
fn magnitude_squared_sums_squared_components() {
    assert_eq!(Vector::from([1, 2, 3]).magnitude_squared(), 14);
}

#[test]
fn dot_computes_inner_product() {
    assert_eq!(Vector::from([1, 2, 3]).dot(&Vector::from([4, 5, 6])), 32);
}

#[test]
fn manahattan_distance_sums_absolute_component_distances() {
    assert_eq!(
        Vector::from([1, 2, 3]).manahattan_distance(&Vector::from([4, 5, 6])),
        9
    );
}

#[test]
fn hamming_distance_counts_different_components() {
    assert_eq!(
        Vector::from([1, 2, 3]).hamming_distance(&Vector::from([1, 0, 3])),
        1
    );
}

#[test]
fn squared_euclidian_distance_sums_squared_component_distances() {
    assert_eq!(
        Vector::from([1, 2, 3]).squared_euclidian_distance(&Vector::from([4, 5, 6])),
        27
    );
}

#[test]
fn magnitude_returns_square_root_of_magnitude_squared() {
    assert_approx_eq(Vector::from([3.0, 4.0]).magnitude(), 5.0);
}

#[test]
fn normalized_divides_by_magnitude() {
    assert_vector_approx_eq(Vector::from([3.0, 4.0]).normalized(), [0.6, 0.8]);
}

#[test]
fn cosine_returns_normalized_dot_product() {
    let x_axis = Vector::from([1.0, 0.0]);
    let sixty_degrees = Vector::from([0.5, f64::sqrt(3.0) / 2.0]);

    assert_approx_eq(x_axis.cosine(&sixty_degrees), 0.5);
}

#[test]
fn sine_returns_angle_sine() {
    let x_axis = Vector::from([1.0, 0.0]);
    let sixty_degrees = Vector::from([0.5, f64::sqrt(3.0) / 2.0]);

    assert_approx_eq(x_axis.sine(&sixty_degrees), f64::sqrt(3.0) / 2.0);
}

#[test]
fn euclidian_distance_returns_distance_between_vectors() {
    assert_approx_eq(
        Vector::from([3.0, 4.0]).euclidian_distance(&Vector::from([0.0, 0.0])),
        5.0,
    );
}

#[test]
fn reciprocal_returns_component_reciprocals() {
    assert_vector_approx_eq(Vector::from([2.0, 4.0]).reciprocal(), [0.5, 0.25]);
}

#[test]
fn cross_returns_three_dimensional_cross_product() {
    assert_eq!(
        Vector::from([1, 0, 0]).cross(&Vector::from([0, 1, 0])),
        Vector::from([0, 0, 1])
    );
}

#[test]
fn index_operator_returns_component() {
    assert_eq!(Vector::from([1, 2, 3])[1], 2);
}

#[test]
fn index_mut_operator_updates_component() {
    let mut vector = Vector::from([1, 2, 3]);
    vector[1] = 20;
    assert_eq!(vector, Vector::from([1, 20, 3]));
}

#[test]
fn range_index_operator_returns_component_slice() {
    assert_eq!(&Vector::from([1, 2, 3, 4])[1..3], &[2, 3]);
}

#[test]
fn range_index_mut_operator_updates_component_slice() {
    let mut vector = Vector::from([1, 2, 3, 4]);
    vector[1..3].copy_from_slice(&[20, 30]);
    assert_eq!(vector, Vector::from([1, 20, 30, 4]));
}

#[test]
fn equality_operator_compares_components() {
    assert_eq!(Vector::from([1, 2, 3]), Vector::from([1, 2, 3]));
    assert_ne!(Vector::from([1, 2, 3]), Vector::from([1, 2, 4]));
}

#[test]
fn ordering_operator_compares_components_lexicographically() {
    assert!(Vector::from([1, 2, 3]) < Vector::from([1, 2, 4]));
}

#[test]
fn neg_operator_negates_components() {
    assert_eq!(-Vector::from([1, -2, 3]), Vector::from([-1, 2, -3]));
}

#[test]
fn vector_add_operator_adds_components() {
    assert_eq!(
        Vector::from([12, 15, 20]) + Vector::from([3, 5, 6]),
        Vector::from([15, 20, 26])
    );
}

#[test]
fn vector_sub_operator_subtracts_components() {
    assert_eq!(
        Vector::from([12, 15, 20]) - Vector::from([3, 5, 6]),
        Vector::from([9, 10, 14])
    );
}

#[test]
fn vector_mul_operator_multiplies_components() {
    assert_eq!(
        Vector::from([12, 15, 20]) * Vector::from([3, 5, 6]),
        Vector::from([36, 75, 120])
    );
}

#[test]
fn vector_div_operator_divides_components() {
    assert_eq!(
        Vector::from([12, 15, 20]) / Vector::from([3, 5, 6]),
        Vector::from([4, 3, 3])
    );
}

#[test]
fn vector_rem_operator_takes_component_remainders() {
    assert_eq!(
        Vector::from([12, 15, 20]) % Vector::from([3, 5, 6]),
        Vector::from([0, 0, 2])
    );
}

#[test]
fn vector_shl_operator_shifts_components_left() {
    assert_eq!(
        Vector::from([1_u32, 8, 32]) << Vector::from([1_u32, 2, 3]),
        Vector::from([2, 32, 256])
    );
}

#[test]
fn vector_shr_operator_shifts_components_right() {
    assert_eq!(
        Vector::from([1_u32, 8, 32]) >> Vector::from([1_u32, 2, 3]),
        Vector::from([0, 2, 4])
    );
}

#[test]
fn vector_bitand_operator_ands_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) & Vector::from([0b1010_u32, 0b0101, 0b0011]),
        Vector::from([0b1000, 0b0000, 0b0010])
    );
}

#[test]
fn vector_bitor_operator_ors_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) | Vector::from([0b1010_u32, 0b0101, 0b0011]),
        Vector::from([0b1110, 0b1111, 0b0111])
    );
}

#[test]
fn vector_bitxor_operator_xors_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) ^ Vector::from([0b1010_u32, 0b0101, 0b0011]),
        Vector::from([0b0110, 0b1111, 0b0101])
    );
}

#[test]
fn vector_add_assign_operator_adds_components() {
    let mut value = Vector::from([1, 2, 3]);
    value += Vector::from([3, 4, 5]);
    assert_eq!(value, Vector::from([4, 6, 8]));
}

#[test]
fn vector_sub_assign_operator_subtracts_components() {
    let mut value = Vector::from([10, 20, 30]);
    value -= Vector::from([3, 4, 5]);
    assert_eq!(value, Vector::from([7, 16, 25]));
}

#[test]
fn vector_mul_assign_operator_multiplies_components() {
    let mut value = Vector::from([2, 3, 4]);
    value *= Vector::from([3, 4, 5]);
    assert_eq!(value, Vector::from([6, 12, 20]));
}

#[test]
fn vector_div_assign_operator_divides_components() {
    let mut value = Vector::from([12, 20, 35]);
    value /= Vector::from([3, 4, 5]);
    assert_eq!(value, Vector::from([4, 5, 7]));
}

#[test]
fn vector_rem_assign_operator_takes_component_remainders() {
    let mut value = Vector::from([12, 20, 37]);
    value %= Vector::from([3, 4, 5]);
    assert_eq!(value, Vector::from([0, 0, 2]));
}

#[test]
fn vector_shl_assign_operator_shifts_components_left() {
    let mut value = Vector::from([1_u32, 8, 32]);
    value <<= Vector::from([1_u32, 2, 3]);
    assert_eq!(value, Vector::from([2, 32, 256]));
}

#[test]
fn vector_shr_assign_operator_shifts_components_right() {
    let mut value = Vector::from([1_u32, 8, 32]);
    value >>= Vector::from([1_u32, 2, 3]);
    assert_eq!(value, Vector::from([0, 2, 4]));
}

#[test]
fn vector_bitand_assign_operator_ands_components() {
    let mut value = Vector::from([0b1100_u32, 0b1010, 0b0110]);
    value &= Vector::from([0b1010_u32, 0b0101, 0b0011]);
    assert_eq!(value, Vector::from([0b1000, 0b0000, 0b0010]));
}

#[test]
fn vector_bitor_assign_operator_ors_components() {
    let mut value = Vector::from([0b1100_u32, 0b1010, 0b0110]);
    value |= Vector::from([0b1010_u32, 0b0101, 0b0011]);
    assert_eq!(value, Vector::from([0b1110, 0b1111, 0b0111]));
}

#[test]
fn vector_bitxor_assign_operator_xors_components() {
    let mut value = Vector::from([0b1100_u32, 0b1010, 0b0110]);
    value ^= Vector::from([0b1010_u32, 0b0101, 0b0011]);
    assert_eq!(value, Vector::from([0b0110, 0b1111, 0b0101]));
}

#[test]
fn scalar_add_operator_adds_to_components() {
    assert_eq!(Vector::from([12, 15, 20]) + 3, Vector::from([15, 18, 23]));
}

#[test]
fn scalar_sub_operator_subtracts_from_components() {
    assert_eq!(Vector::from([12, 15, 20]) - 3, Vector::from([9, 12, 17]));
}

#[test]
fn scalar_mul_operator_multiplies_components() {
    assert_eq!(Vector::from([12, 15, 20]) * 3, Vector::from([36, 45, 60]));
}

#[test]
fn scalar_div_operator_divides_components() {
    assert_eq!(Vector::from([12, 15, 20]) / 3, Vector::from([4, 5, 6]));
}

#[test]
fn scalar_rem_operator_takes_component_remainders() {
    assert_eq!(Vector::from([12, 15, 20]) % 3, Vector::from([0, 0, 2]));
}

#[test]
fn scalar_shl_operator_shifts_components_left() {
    assert_eq!(
        Vector::from([1_u32, 8, 32]) << 2,
        Vector::from([4, 32, 128])
    );
}

#[test]
fn scalar_shr_operator_shifts_components_right() {
    assert_eq!(Vector::from([1_u32, 8, 32]) >> 2, Vector::from([0, 2, 8]));
}

#[test]
fn scalar_bitand_operator_ands_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) & 0b1010,
        Vector::from([0b1000, 0b1010, 0b0010])
    );
}

#[test]
fn scalar_bitor_operator_ors_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) | 0b0011,
        Vector::from([0b1111, 0b1011, 0b0111])
    );
}

#[test]
fn scalar_bitxor_operator_xors_components() {
    assert_eq!(
        Vector::from([0b1100_u32, 0b1010, 0b0110]) ^ 0b0101,
        Vector::from([0b1001, 0b1111, 0b0011])
    );
}

#[test]
fn scalar_add_assign_operator_adds_to_components() {
    let mut value = Vector::from([1, 2, 3]);
    value += 3;
    assert_eq!(value, Vector::from([4, 5, 6]));
}

#[test]
fn scalar_sub_assign_operator_subtracts_from_components() {
    let mut value = Vector::from([4, 5, 6]);
    value -= 2;
    assert_eq!(value, Vector::from([2, 3, 4]));
}

#[test]
fn scalar_mul_assign_operator_multiplies_components() {
    let mut value = Vector::from([2, 3, 4]);
    value *= 4;
    assert_eq!(value, Vector::from([8, 12, 16]));
}

#[test]
fn scalar_div_assign_operator_divides_components() {
    let mut value = Vector::from([8, 12, 16]);
    value /= 2;
    assert_eq!(value, Vector::from([4, 6, 8]));
}

#[test]
fn scalar_rem_assign_operator_takes_component_remainders() {
    let mut value = Vector::from([4, 6, 8]);
    value %= 5;
    assert_eq!(value, Vector::from([4, 1, 3]));
}

#[test]
fn scalar_shl_assign_operator_shifts_components_left() {
    let mut value = Vector::from([1_u32, 8, 32]);
    value <<= 2;
    assert_eq!(value, Vector::from([4, 32, 128]));
}

#[test]
fn scalar_shr_assign_operator_shifts_components_right() {
    let mut value = Vector::from([4_u32, 32, 128]);
    value >>= 2;
    assert_eq!(value, Vector::from([1, 8, 32]));
}

#[test]
fn scalar_bitand_assign_operator_ands_components() {
    let mut value = Vector::from([0b1100_u32, 0b1010, 0b0110]);
    value &= 0b1010;
    assert_eq!(value, Vector::from([0b1000, 0b1010, 0b0010]));
}

#[test]
fn scalar_bitor_assign_operator_ors_components() {
    let mut value = Vector::from([0b1000_u32, 0b1010, 0b0010]);
    value |= 0b0101;
    assert_eq!(value, Vector::from([0b1101, 0b1111, 0b0111]));
}

#[test]
fn scalar_bitxor_assign_operator_xors_components() {
    let mut value = Vector::from([0b1101_u32, 0b1111, 0b0111]);
    value ^= 0b0011;
    assert_eq!(value, Vector::from([0b1110, 0b1100, 0b0100]));
}
