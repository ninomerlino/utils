use utils::linear_systems::{Matrix, Vector};

fn assert_approx_eq(left: f64, right: f64) {
    assert!(
        (left - right).abs() <= 1e-10,
        "left: {left}, right: {right}"
    );
}

fn assert_vector_approx_eq<const S: usize>(left: Vector<f64, S>, right: [f64; S]) {
    for (actual, expected) in left.iter().zip(right) {
        assert_approx_eq(*actual, expected);
    }
}

fn assert_matrix_approx_eq<const ROWS: usize, const COLS: usize>(
    left: Matrix<f64, ROWS, COLS>,
    right: [[f64; COLS]; ROWS],
) {
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_approx_eq(left[(row, col)], right[row][col]);
        }
    }
}

fn multiply<const ROWS: usize, const INNER: usize, const COLS: usize>(
    left: Matrix<f64, ROWS, INNER>,
    right: Matrix<f64, INNER, COLS>,
) -> [[f64; COLS]; ROWS] {
    let mut result = [[0.0; COLS]; ROWS];

    for row in 0..ROWS {
        for col in 0..COLS {
            for inner in 0..INNER {
                result[row][col] += left[(row, inner)] * right[(inner, col)];
            }
        }
    }

    result
}

#[test]
fn identity_returns_identity_matrix() {
    assert_matrix_approx_eq(
        Matrix::<f64, 3, 3>::identity(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
}

#[test]
fn from_dot_returns_outer_product() {
    let left = Vector::from([2.0, 3.0]);
    let right = Vector::from([5.0, 7.0]);

    assert_matrix_approx_eq(
        Matrix::from_dot(&left, &right),
        [[10.0, 14.0], [15.0, 21.0]],
    );
}

#[test]
fn properties_describes_identity_matrix() {
    let props = Matrix::<f64, 2, 2>::identity().properties();

    assert!(props.simmetric);
    assert!(props.diagonal);
    assert!(props.upper_triangular);
    assert!(props.lower_triangular);
    assert!(props.all());
    assert!(props.any());
}

#[test]
fn properties_describes_general_non_triangular_matrix() {
    let props = Matrix::from([[1.0, 2.0], [3.0, 4.0]]).properties();

    assert!(!props.simmetric);
    assert!(!props.diagonal);
    assert!(!props.upper_triangular);
    assert!(!props.lower_triangular);
    assert!(!props.all());
    assert!(!props.any());
}

#[test]
fn lup_decomposition_reconstructs_permuted_matrix() {
    let matrix = Matrix::from([[0.0, 2.0], [1.0, 3.0]]);
    let (l, u, p) = matrix.lup_decomposition().unwrap();

    assert_matrix_approx_eq(l, [[1.0, 0.0], [0.0, 1.0]]);
    assert_matrix_approx_eq(u, [[1.0, 3.0], [0.0, 2.0]]);
    assert_matrix_approx_eq(p, [[0.0, 1.0], [1.0, 0.0]]);
    assert_matrix_approx_eq(Matrix::from(multiply(p, matrix)), multiply(l, u));
}

#[test]
fn lup_decomposition_returns_none_for_singular_matrix() {
    assert!(
        Matrix::from([[1.0, 2.0], [2.0, 4.0]])
            .lup_decomposition()
            .is_none()
    );
}

#[test]
fn solve_lower_triangular_solves_by_forward_substitution() {
    let lower = Matrix::from([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let b = Vector::from([2.0, 5.0, 1.0]);

    assert_vector_approx_eq(lower.solve_lower_triangular(&b), [1.0, 2.0, 2.0]);
}

#[test]
fn solve_upper_triangular_solves_by_back_substitution() {
    let upper = Matrix::from([[2.0, 1.0, -1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 4.0]]);
    let b = Vector::from([3.0, 7.0, 8.0]);

    assert_vector_approx_eq(upper.solve_upper_triangular(&b), [2.0, 1.0, 2.0]);
}

#[test]
fn solve_solves_linear_system() {
    let matrix = Matrix::from([[2.0, 1.0], [5.0, 7.0]]);
    let b = Vector::from([35.0, 146.0]);

    assert_vector_approx_eq(matrix.solve(&b), [11.0, 13.0]);
}
