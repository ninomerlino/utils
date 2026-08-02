use utils::linear_systems::{Matrix, Vector};

#[test]
fn from_array_preserves_rows_and_columns() {
    let matrix: Matrix<i32, 2, 3> = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(matrix[0], Vector::from([1, 2, 3]));
    assert_eq!(matrix[1], Vector::from([4, 5, 6]));
}

#[test]
fn from_vector_array_preserves_rows() {
    let matrix = Matrix::from([Vector::from([1, 2]), Vector::from([3, 4])]);

    assert_eq!(matrix, Matrix::from([[1, 2], [3, 4]]));
}

#[test]
fn from_scalar_fills_every_component() {
    assert_eq!(
        Matrix::<i32, 2, 3>::from(7),
        Matrix::from([[7, 7, 7], [7, 7, 7]])
    );
}

#[test]
fn default_creates_zero_matrix() {
    assert_eq!(
        Matrix::<i32, 2, 3>::default(),
        Matrix::from([[0, 0, 0], [0, 0, 0]])
    );
}

#[test]
fn rows_returns_const_row_count() {
    assert_eq!(Matrix::<i32, 2, 3>::default().rows(), 2);
}

#[test]
fn cols_returns_const_column_count() {
    assert_eq!(Matrix::<i32, 2, 3>::default().cols(), 3);
}

#[test]
fn is_square_detects_square_matrix() {
    assert!(Matrix::<i32, 3, 3>::default().is_square());
}

#[test]
fn is_square_rejects_rectangular_matrix() {
    assert!(!Matrix::<i32, 2, 3>::default().is_square());
}

#[test]
fn iter_visits_rows_in_order() {
    let matrix = Matrix::from([[1, 2], [3, 4]]);
    let rows: Vec<_> = matrix.iter().copied().collect();

    assert_eq!(rows, vec![Vector::from([1, 2]), Vector::from([3, 4])]);
}

#[test]
fn iter_mut_allows_row_updates() {
    let mut matrix = Matrix::from([[1, 2], [3, 4]]);

    for row in matrix.iter_mut() {
        *row += Vector::from([10, 20]);
    }

    assert_eq!(matrix, Matrix::from([[11, 22], [13, 24]]));
}

#[test]
fn get_column_returns_column_vector() {
    let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(matrix.get_column(1), Vector::from([2, 5]));
}

#[test]
fn set_column_writes_column_vector() {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    matrix.set_column(1, &Vector::from([20, 50, 80]));

    assert_eq!(matrix, Matrix::from([[1, 20, 3], [4, 50, 6], [7, 80, 9]]));
}

#[test]
fn get_row_returns_row_vector() {
    let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    assert_eq!(matrix.get_row(1), Vector::from([4, 5, 6]));
}

#[test]
fn set_row_writes_row_vector() {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    matrix.set_row(0, &Vector::from([10, 20, 30]));

    assert_eq!(matrix, Matrix::from([[10, 20, 30], [4, 5, 6]]));
}

#[test]
fn dot_vector_multiplies_matrix_by_vector() {
    let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    let vector = Vector::from([7, 8, 9]);

    assert_eq!(matrix.dot_vector(&vector), Vector::from([50, 122]));
}

#[test]
fn dot_matrix_multiplies_matrix_by_matrix() {
    let left = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    let right = Matrix::from([[7, 8], [9, 10], [11, 12]]);

    assert_eq!(
        left.dot_matrix(&right),
        Matrix::from([[58, 64], [139, 154]])
    );
}

#[test]
fn swap_rows_exchanges_entire_rows() {
    let mut matrix = Matrix::from([[1, 2], [3, 4], [5, 6]]);

    matrix.swap_rows(0, 2);

    assert_eq!(matrix, Matrix::from([[5, 6], [3, 4], [1, 2]]));
}

#[test]
fn swap_rows_partial_exchanges_selected_columns_between_rows() {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    matrix.swap_rows_partial(0, 1, 1..3);

    assert_eq!(matrix, Matrix::from([[1, 5, 6], [4, 2, 3]]));
}

#[test]
fn swap_cols_exchanges_entire_columns() {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);

    matrix.swap_cols(0, 2);

    assert_eq!(matrix, Matrix::from([[3, 2, 1], [6, 5, 4]]));
}

#[test]
fn swap_cols_partial_exchanges_selected_rows_between_columns() {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

    matrix.swap_cols_partial(0, 2, 1..3);

    assert_eq!(matrix, Matrix::from([[1, 2, 3], [6, 5, 4], [9, 8, 7]]));
}

#[test]
fn max_from_row_returns_largest_row_value_and_column_index() {
    let matrix = Matrix::from([[1, 5, 3], [4, 2, 6]]);

    assert_eq!(matrix.max_from_row(0), Some((5, 1)));
}

#[test]
fn max_from_col_returns_largest_column_value_and_row_index() {
    let matrix = Matrix::from([[1, 5], [4, 2], [3, 6]]);

    assert_eq!(matrix.max_from_col(0), Some((4, 1)));
}

#[test]
fn abs_max_from_row_returns_largest_absolute_row_value_and_column_index() {
    let matrix = Matrix::from([[-7, 5, 3], [4, 2, 6]]);

    assert_eq!(matrix.abs_max_from_row(0), Some((7, 0)));
}

#[test]
fn abs_max_from_col_returns_largest_absolute_column_value_and_row_index() {
    let matrix = Matrix::from([[-7, 5], [4, 2], [3, 6]]);

    assert_eq!(matrix.abs_max_from_col(0), Some((7, 0)));
}

#[test]
fn abs_max_from_col_partial_searches_only_selected_rows() {
    let matrix = Matrix::from([[100, 0], [-7, 0], [9, 0]]);

    assert_eq!(matrix.abs_max_from_col_partial(0, 1..3), Some((9, 2)));
}

#[test]
fn index_operator_returns_row() {
    let matrix = Matrix::from([[1, 2], [3, 4]]);

    assert_eq!(matrix[1], Vector::from([3, 4]));
}

#[test]
fn index_mut_operator_updates_row() {
    let mut matrix = Matrix::from([[1, 2], [3, 4]]);

    matrix[1] = Vector::from([30, 40]);

    assert_eq!(matrix, Matrix::from([[1, 2], [30, 40]]));
}

#[test]
fn range_index_operator_returns_rows() {
    let matrix = Matrix::from([[1, 2], [3, 4], [5, 6]]);
    let rows = &matrix[1..3];

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], Vector::from([3, 4]));
    assert_eq!(rows[1], Vector::from([5, 6]));
}

#[test]
fn range_index_mut_operator_updates_rows() {
    let mut matrix = Matrix::from([[1, 2], [3, 4], [5, 6]]);

    matrix[1..3].copy_from_slice(&[Vector::from([30, 40]), Vector::from([50, 60])]);

    assert_eq!(matrix, Matrix::from([[1, 2], [30, 40], [50, 60]]));
}

#[test]
fn tuple_index_operator_returns_component() {
    assert_eq!(Matrix::from([[1, 2], [3, 4]])[(1, 0)], 3);
}

#[test]
fn tuple_index_mut_operator_updates_component() {
    let mut matrix = Matrix::from([[1, 2], [3, 4]]);

    matrix[(1, 0)] = 30;

    assert_eq!(matrix, Matrix::from([[1, 2], [30, 4]]));
}

#[test]
fn neg_operator_negates_components() {
    assert_eq!(
        -Matrix::from([[1, -2], [3, -4]]),
        Matrix::from([[-1, 2], [-3, 4]])
    );
}

#[test]
fn matrix_add_operator_adds_components() {
    assert_eq!(
        Matrix::from([[1, 2], [3, 4]]) + Matrix::from([[5, 6], [7, 8]]),
        Matrix::from([[6, 8], [10, 12]])
    );
}

#[test]
fn matrix_sub_operator_subtracts_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) - Matrix::from([[5, 6], [7, 8]]),
        Matrix::from([[7, 9], [13, 17]])
    );
}

#[test]
fn matrix_mul_operator_multiplies_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) * Matrix::from([[5, 6], [7, 8]]),
        Matrix::from([[60, 90], [140, 200]])
    );
}

#[test]
fn matrix_div_operator_divides_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 24]]) / Matrix::from([[3, 5], [4, 6]]),
        Matrix::from([[4, 3], [5, 4]])
    );
}

#[test]
fn matrix_rem_operator_takes_component_remainders() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) % Matrix::from([[5, 6], [7, 8]]),
        Matrix::from([[2, 3], [6, 1]])
    );
}

#[test]
fn matrix_shl_operator_shifts_components_left() {
    assert_eq!(
        Matrix::from([[1_u32, 8], [32, 64]]) << Matrix::from([[1_u32, 2], [3, 4]]),
        Matrix::from([[2, 32], [256, 1024]])
    );
}

#[test]
fn matrix_shr_operator_shifts_components_right() {
    assert_eq!(
        Matrix::from([[1_u32, 8], [32, 64]]) >> Matrix::from([[1_u32, 2], [3, 4]]),
        Matrix::from([[0, 2], [4, 4]])
    );
}

#[test]
fn matrix_bitand_operator_ands_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]])
            & Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]),
        Matrix::from([[0b1000, 0b0000], [0b0010, 0b0011]])
    );
}

#[test]
fn matrix_bitor_operator_ors_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]])
            | Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]),
        Matrix::from([[0b1110, 0b1111], [0b0111, 0b1111]])
    );
}

#[test]
fn matrix_bitxor_operator_xors_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]])
            ^ Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]),
        Matrix::from([[0b0110, 0b1111], [0b0101, 0b1100]])
    );
}

#[test]
fn matrix_add_assign_operator_adds_components() {
    let mut value = Matrix::from([[1, 2], [3, 4]]);
    value += Matrix::from([[5, 6], [7, 8]]);
    assert_eq!(value, Matrix::from([[6, 8], [10, 12]]));
}

#[test]
fn matrix_sub_assign_operator_subtracts_components() {
    let mut value = Matrix::from([[12, 15], [20, 25]]);
    value -= Matrix::from([[5, 6], [7, 8]]);
    assert_eq!(value, Matrix::from([[7, 9], [13, 17]]));
}

#[test]
fn matrix_mul_assign_operator_multiplies_components() {
    let mut value = Matrix::from([[12, 15], [20, 25]]);
    value *= Matrix::from([[5, 6], [7, 8]]);
    assert_eq!(value, Matrix::from([[60, 90], [140, 200]]));
}

#[test]
fn matrix_div_assign_operator_divides_components() {
    let mut value = Matrix::from([[12, 15], [20, 24]]);
    value /= Matrix::from([[3, 5], [4, 6]]);
    assert_eq!(value, Matrix::from([[4, 3], [5, 4]]));
}

#[test]
fn matrix_rem_assign_operator_takes_component_remainders() {
    let mut value = Matrix::from([[12, 15], [20, 25]]);
    value %= Matrix::from([[5, 6], [7, 8]]);
    assert_eq!(value, Matrix::from([[2, 3], [6, 1]]));
}

#[test]
fn matrix_shl_assign_operator_shifts_components_left() {
    let mut value = Matrix::from([[1_u32, 8], [32, 64]]);
    value <<= Matrix::from([[1_u32, 2], [3, 4]]);
    assert_eq!(value, Matrix::from([[2, 32], [256, 1024]]));
}

#[test]
fn matrix_shr_assign_operator_shifts_components_right() {
    let mut value = Matrix::from([[1_u32, 8], [32, 64]]);
    value >>= Matrix::from([[1_u32, 2], [3, 4]]);
    assert_eq!(value, Matrix::from([[0, 2], [4, 4]]));
}

#[test]
fn matrix_bitand_assign_operator_ands_components() {
    let mut value = Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]);
    value &= Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]);
    assert_eq!(value, Matrix::from([[0b1000, 0b0000], [0b0010, 0b0011]]));
}

#[test]
fn matrix_bitor_assign_operator_ors_components() {
    let mut value = Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]);
    value |= Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]);
    assert_eq!(value, Matrix::from([[0b1110, 0b1111], [0b0111, 0b1111]]));
}

#[test]
fn matrix_bitxor_assign_operator_xors_components() {
    let mut value = Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]);
    value ^= Matrix::from([[0b1010_u32, 0b0101], [0b0011, 0b1111]]);
    assert_eq!(value, Matrix::from([[0b0110, 0b1111], [0b0101, 0b1100]]));
}

#[test]
fn scalar_add_operator_adds_to_components() {
    assert_eq!(
        Matrix::from([[1, 2], [3, 4]]) + 3,
        Matrix::from([[4, 5], [6, 7]])
    );
}

#[test]
fn scalar_sub_operator_subtracts_from_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) - 3,
        Matrix::from([[9, 12], [17, 22]])
    );
}

#[test]
fn scalar_mul_operator_multiplies_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) * 3,
        Matrix::from([[36, 45], [60, 75]])
    );
}

#[test]
fn scalar_div_operator_divides_components() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 24]]) / 3,
        Matrix::from([[4, 5], [6, 8]])
    );
}

#[test]
fn scalar_rem_operator_takes_component_remainders() {
    assert_eq!(
        Matrix::from([[12, 15], [20, 25]]) % 3,
        Matrix::from([[0, 0], [2, 1]])
    );
}

#[test]
fn scalar_shl_operator_shifts_components_left() {
    assert_eq!(
        Matrix::from([[1_u32, 8], [32, 64]]) << 2,
        Matrix::from([[4, 32], [128, 256]])
    );
}

#[test]
fn scalar_shr_operator_shifts_components_right() {
    assert_eq!(
        Matrix::from([[1_u32, 8], [32, 64]]) >> 2,
        Matrix::from([[0, 2], [8, 16]])
    );
}

#[test]
fn scalar_bitand_operator_ands_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]) & 0b1010,
        Matrix::from([[0b1000, 0b1010], [0b0010, 0b0010]])
    );
}

#[test]
fn scalar_bitor_operator_ors_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]) | 0b0011,
        Matrix::from([[0b1111, 0b1011], [0b0111, 0b0011]])
    );
}

#[test]
fn scalar_bitxor_operator_xors_components() {
    assert_eq!(
        Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]) ^ 0b0101,
        Matrix::from([[0b1001, 0b1111], [0b0011, 0b0110]])
    );
}

#[test]
fn scalar_add_assign_operator_adds_to_components() {
    let mut value = Matrix::from([[1, 2], [3, 4]]);
    value += 3;
    assert_eq!(value, Matrix::from([[4, 5], [6, 7]]));
}

#[test]
fn scalar_sub_assign_operator_subtracts_from_components() {
    let mut value = Matrix::from([[4, 5], [6, 7]]);
    value -= 2;
    assert_eq!(value, Matrix::from([[2, 3], [4, 5]]));
}

#[test]
fn scalar_mul_assign_operator_multiplies_components() {
    let mut value = Matrix::from([[2, 3], [4, 5]]);
    value *= 4;
    assert_eq!(value, Matrix::from([[8, 12], [16, 20]]));
}

#[test]
fn scalar_div_assign_operator_divides_components() {
    let mut value = Matrix::from([[8, 12], [16, 20]]);
    value /= 2;
    assert_eq!(value, Matrix::from([[4, 6], [8, 10]]));
}

#[test]
fn scalar_rem_assign_operator_takes_component_remainders() {
    let mut value = Matrix::from([[4, 6], [8, 10]]);
    value %= 5;
    assert_eq!(value, Matrix::from([[4, 1], [3, 0]]));
}

#[test]
fn scalar_shl_assign_operator_shifts_components_left() {
    let mut value = Matrix::from([[1_u32, 8], [32, 64]]);
    value <<= 2;
    assert_eq!(value, Matrix::from([[4, 32], [128, 256]]));
}

#[test]
fn scalar_shr_assign_operator_shifts_components_right() {
    let mut value = Matrix::from([[4_u32, 32], [128, 256]]);
    value >>= 2;
    assert_eq!(value, Matrix::from([[1, 8], [32, 64]]));
}

#[test]
fn scalar_bitand_assign_operator_ands_components() {
    let mut value = Matrix::from([[0b1100_u32, 0b1010], [0b0110, 0b0011]]);
    value &= 0b1010;
    assert_eq!(value, Matrix::from([[0b1000, 0b1010], [0b0010, 0b0010]]));
}

#[test]
fn scalar_bitor_assign_operator_ors_components() {
    let mut value = Matrix::from([[0b1000_u32, 0b1010], [0b0010, 0b0010]]);
    value |= 0b0101;
    assert_eq!(value, Matrix::from([[0b1101, 0b1111], [0b0111, 0b0111]]));
}

#[test]
fn scalar_bitxor_assign_operator_xors_components() {
    let mut value = Matrix::from([[0b1101_u32, 0b1111], [0b0111, 0b0111]]);
    value ^= 0b0011;
    assert_eq!(value, Matrix::from([[0b1110, 0b1100], [0b0100, 0b0100]]));
}
