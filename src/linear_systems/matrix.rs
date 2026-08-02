use std::{
    array::{self},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, Rem, RemAssign, Shl, ShlAssign,
        Shr, ShrAssign, Sub, SubAssign,
    },
};

use super::vector::{Vector, VectorType};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq)]
pub struct Matrix<T: VectorType, const ROWS: usize, const COLS: usize> {
    values: [Vector<T, COLS>; ROWS],
}

// ============================================================
// Inherent methods
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn rows(&self) -> usize {
        ROWS
    }

    pub fn cols(&self) -> usize {
        COLS
    }

    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Vector<T, COLS>> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Vector<T, COLS>> {
        self.values.iter_mut()
    }

    pub fn get_column(&self, index: usize) -> Vector<T, ROWS> {
        let mut vector = Vector::default();

        for row in 0..ROWS {
            vector[row] = self[row][index]
        }

        vector
    }

    pub fn set_column(&mut self, index: usize, vector: &Vector<T, ROWS>) {
        for row in 0..ROWS {
            self[row][index] = vector[row];
        }
    }

    pub fn get_row(&self, index: usize) -> Vector<T, COLS> {
        self[index]
    }

    pub fn set_row(&mut self, index: usize, vector: &Vector<T, COLS>) {
        self[index] = *vector;
    }

    pub fn dot_vector(&self, vector: &Vector<T, COLS>) -> Vector<T, ROWS> {
        let mut result = Vector::default();
        for row in 0..ROWS {
            result[row] = self.values[row].dot(vector);
        }
        result
    }

    pub fn dot_matrix<const OTHER_COLS: usize>(
        &self,
        other: &Matrix<T, COLS, OTHER_COLS>,
    ) -> Matrix<T, ROWS, OTHER_COLS> {
        let mut result = Matrix::default();

        for col in 0..OTHER_COLS {
            let vector = other.get_column(col);
            for row in 0..ROWS {
                result[row][col] = self[row].dot(&vector);
            }
        }

        result
    }

    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let t = self[row_a];
        self[row_a] = self[row_b];
        self[row_b] = t;
    }

    pub fn swap_rows_partial(&mut self, row_a: usize, row_b: usize, mask: Range<usize>) {
        for col in mask {
            let t = self[row_a][col];
            self[row_a][col] = self[row_b][col];
            self[row_b][col] = t;
        }
    }

    pub fn swap_cols(&mut self, col_a: usize, col_b: usize) {
        let a = self.get_column(col_a);
        let b = self.get_column(col_b);
        self.set_column(col_a, &b);
        self.set_column(col_b, &a);
    }

    pub fn swap_cols_partial(&mut self, col_a: usize, col_b: usize, mask: Range<usize>) {
        for row in mask {
            let t = self[row][col_a];
            self[row][col_a] = self[row][col_b];
            self[row][col_b] = t;
        }
    }

    pub fn max_from_row(&self, row: usize) -> Option<(T, usize)> {
        self[row].max()
    }

    pub fn max_from_col(&self, col: usize) -> Option<(T, usize)> {
        self.get_column(col).max()
    }

    pub fn abs_max_from_row(&self, row: usize) -> Option<(T, usize)> {
        self[row].abs().max()
    }

    pub fn abs_max_from_col(&self, col: usize) -> Option<(T, usize)> {
        self.get_column(col).abs().max()
    }

    pub fn abs_max_from_col_partial(&self, col: usize, mask: Range<usize>) -> Option<(T, usize)> {
        let mut max = Option::None;
        for row in mask {
            if let Some((max_val, _)) = max {
                if self[row][col].absolute_value() > max_val {
                    max = Some((self[row][col].absolute_value(), row));
                }
            } else {
                max = Some((self[row][col].absolute_value(), row));
            }
        }
        max
    }
}

// ============================================================
// From / Default Traits
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> From<[Vector<T, COLS>; ROWS]>
    for Matrix<T, ROWS, COLS>
{
    fn from(value: [Vector<T, COLS>; ROWS]) -> Self {
        Self { values: value }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]>
    for Matrix<T, ROWS, COLS>
{
    fn from(value: [[T; COLS]; ROWS]) -> Self {
        Self {
            values: value.map(Vector::from),
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> From<T> for Matrix<T, ROWS, COLS> {
    fn from(value: T) -> Self {
        Self {
            values: array::from_fn(|_| Vector::from(value)),
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Default for Matrix<T, ROWS, COLS> {
    fn default() -> Self {
        Self {
            values: array::from_fn(|_| Vector::default()),
        }
    }
}

// ============================================================
// Index Traits
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> Index<usize> for Matrix<T, ROWS, COLS> {
    type Output = Vector<T, COLS>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> IndexMut<usize>
    for Matrix<T, ROWS, COLS>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Index<Range<usize>>
    for Matrix<T, ROWS, COLS>
{
    type Output = [Vector<T, COLS>];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.values[range]
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> IndexMut<Range<usize>>
    for Matrix<T, ROWS, COLS>
{
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.values[range]
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Index<(usize, usize)>
    for Matrix<T, ROWS, COLS>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.values[index.0][index.1]
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)>
    for Matrix<T, ROWS, COLS>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.values[index.0][index.1]
    }
}

// ============================================================
// <op> Matrix
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> Neg for Matrix<T, ROWS, COLS>
where
    T: Neg<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn neg(self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = -self.values[i]
        }
        r
    }
}

// ============================================================
// Matrix <op> Matrix
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> Add for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] + rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Sub for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] - rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Mul for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    /// Element-wise (Hadamard) product, consistent with `Vector`'s `Mul`.
    /// For classic row-by-column matrix multiplication use `matmul`.
    fn mul(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] * rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Div for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] / rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Rem for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] % rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Shl for Matrix<T, ROWS, COLS>
where
    T: Shl<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn shl(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] << rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Shr for Matrix<T, ROWS, COLS>
where
    T: Shr<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn shr(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] >> rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitAnd for Matrix<T, ROWS, COLS>
where
    T: BitAnd<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] & rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitOr for Matrix<T, ROWS, COLS>
where
    T: BitOr<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] | rhs.values[i]
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitXor for Matrix<T, ROWS, COLS>
where
    T: BitXor<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] ^ rhs.values[i]
        }
        r
    }
}

// ============================================================
// Matrix <op>= Matrix
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> AddAssign for Matrix<T, ROWS, COLS> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] += rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> SubAssign for Matrix<T, ROWS, COLS> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] -= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> MulAssign for Matrix<T, ROWS, COLS> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] *= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> DivAssign for Matrix<T, ROWS, COLS> {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] /= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> RemAssign for Matrix<T, ROWS, COLS> {
    fn rem_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] %= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> ShlAssign for Matrix<T, ROWS, COLS>
where
    T: ShlAssign,
{
    fn shl_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] <<= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> ShrAssign for Matrix<T, ROWS, COLS>
where
    T: ShrAssign,
{
    fn shr_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] >>= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitAndAssign for Matrix<T, ROWS, COLS>
where
    T: BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] &= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitOrAssign for Matrix<T, ROWS, COLS>
where
    T: BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] |= rhs.values[i]
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitXorAssign for Matrix<T, ROWS, COLS>
where
    T: BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..ROWS {
            self.values[i] ^= rhs.values[i]
        }
    }
}

// ============================================================
// Matrix <op> Scalar (T)
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> Add<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn add(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] + rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Sub<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] - rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] * rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Div<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] / rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Rem<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;

    fn rem(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] % rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Shl<T> for Matrix<T, ROWS, COLS>
where
    T: Shl<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn shl(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] << rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> Shr<T> for Matrix<T, ROWS, COLS>
where
    T: Shr<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn shr(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] >> rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitAnd<T> for Matrix<T, ROWS, COLS>
where
    T: BitAnd<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitand(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] & rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitOr<T> for Matrix<T, ROWS, COLS>
where
    T: BitOr<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitor(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] | rhs
        }
        r
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitXor<T> for Matrix<T, ROWS, COLS>
where
    T: BitXor<Output = T>,
{
    type Output = Matrix<T, ROWS, COLS>;

    fn bitxor(self, rhs: T) -> Self::Output {
        let mut r = Self::Output::default();
        for i in 0..ROWS {
            r.values[i] = self.values[i] ^ rhs
        }
        r
    }
}

// ============================================================
// Matrix <op>= Scalar (T)
// ============================================================

impl<T: VectorType, const ROWS: usize, const COLS: usize> AddAssign<T> for Matrix<T, ROWS, COLS> {
    fn add_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] += rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> SubAssign<T> for Matrix<T, ROWS, COLS> {
    fn sub_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] -= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> MulAssign<T> for Matrix<T, ROWS, COLS> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] *= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> DivAssign<T> for Matrix<T, ROWS, COLS> {
    fn div_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] /= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> RemAssign<T> for Matrix<T, ROWS, COLS> {
    fn rem_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] %= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> ShlAssign<T> for Matrix<T, ROWS, COLS>
where
    T: ShlAssign,
{
    fn shl_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] <<= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> ShrAssign<T> for Matrix<T, ROWS, COLS>
where
    T: ShrAssign,
{
    fn shr_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] >>= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitAndAssign<T> for Matrix<T, ROWS, COLS>
where
    T: BitAndAssign,
{
    fn bitand_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] &= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitOrAssign<T> for Matrix<T, ROWS, COLS>
where
    T: BitOrAssign,
{
    fn bitor_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] |= rhs
        }
    }
}

impl<T: VectorType, const ROWS: usize, const COLS: usize> BitXorAssign<T> for Matrix<T, ROWS, COLS>
where
    T: BitXorAssign,
{
    fn bitxor_assign(&mut self, rhs: T) {
        for i in 0..ROWS {
            self.values[i] ^= rhs
        }
    }
}
