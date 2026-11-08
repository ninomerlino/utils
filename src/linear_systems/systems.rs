use super::{Matrix, Vector, vector::VectorType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Structural properties detected for a square matrix.
pub struct MatrixProperties {
    /// Whether the matrix equals its transpose within the numeric tolerance.
    pub simmetric: bool,
    /// Whether all off-diagonal entries are approximately zero.
    pub diagonal: bool,
    /// Whether all entries below the diagonal are approximately zero.
    pub upper_triangular: bool,
    /// Whether all entries above the diagonal are approximately zero.
    pub lower_triangular: bool,
}

impl Default for MatrixProperties {
    fn default() -> Self {
        Self {
            simmetric: true,
            diagonal: true,
            upper_triangular: true,
            lower_triangular: true,
        }
    }
}

type LuprTuple<T, const S: usize> = (Matrix<T, S, S>, Matrix<T, S, S>, Matrix<T, S, S>, usize);

impl MatrixProperties {
    /// Returns `true` if all four properties are present.
    pub fn all(&self) -> bool {
        self.simmetric && self.diagonal && self.upper_triangular && self.lower_triangular
    }

    /// Returns `true` if none of the four properties are present.
    pub fn none(&self) -> bool {
        !self.any()
    }

    /// Returns `true` if at least one of the four properties is present.
    pub fn any(&self) -> bool {
        self.simmetric || self.diagonal || self.upper_triangular || self.lower_triangular
    }
}

impl<T: VectorType, const S: usize> Matrix<T, S, S> {
    /// Constructs the `S`-by-`S` identity matrix.
    pub fn identity() -> Self {
        let mut mat: Matrix<T, S, S> = Matrix::default();
        for row in 0..S {
            mat[row][row] = T::unit();
        }
        mat
    }

    /// Constructs the outer product `a * bᵀ`.
    pub fn from_dot(a: &Vector<T, S>, b: &Vector<T, S>) -> Self {
        let mut mat: Matrix<T, S, S> = Matrix::default();
        for (row_i, row) in mat.iter_mut().enumerate() {
            *row = *b * a[row_i];
        }
        mat
    }

    /// Checks symmetry, diagonal, and triangular structure using the numeric
    /// type's approximate-zero tolerance.
    pub fn properties(&self) -> MatrixProperties {
        let mut prop = MatrixProperties::default();

        for row in 0..S {
            for col in 0..S {
                if !(self[row][col] - self[col][row]).approximate_zero() {
                    prop.simmetric = false;
                }
                if row != col && !self[row][col].approximate_zero() {
                    prop.diagonal = false;
                }
                if row > col && !self[row][col].approximate_zero() {
                    prop.upper_triangular = false;
                }
                if row < col && !self[row][col].approximate_zero() {
                    prop.lower_triangular = false;
                }

                if prop.none() {
                    return prop;
                }
            }
        }

        prop
    }

    /// Returns the transpose of this square matrix.
    pub fn transpose(&self) -> Matrix<T, S, S> {
        let mut result = *self;
        for row in 0..S {
            for col in 0..row {
                let t = self[row][col];
                result[row][col] = result[col][row];
                result[col][row] = t;
            }
        }
        result
    }
}

impl<T: VectorType, const S: usize> Matrix<T, S, S> {
    /// Factors the matrix as `P * self = L * U` using partial pivoting.
    /// Returns `None` when a pivot is zero or approximately zero.
    pub fn lup_decomposition(&self) -> Option<(Matrix<T, S, S>, Matrix<T, S, S>, Matrix<T, S, S>)> {
        let (l, u, p, _) = self.lupr_decomposition()?;
        Some((l, u, p))
    }

    /// Factors the matrix as `P * self = L * U` using partial pivoting.
    /// In addition to LUP factors, returns the number of row swaps performed.
    /// Returns `None` when a pivot is zero or approximately zero.
    pub fn lupr_decomposition(&self) -> Option<LuprTuple<T, S>> {
        let mut p: Matrix<T, S, S> = Matrix::identity();
        let mut l: Matrix<T, S, S> = Matrix::identity();
        let mut u: Matrix<T, S, S> = *self;
        let mut r = 0;

        for i in 0..S {
            let (_, k) = u.abs_max_from_col_partial(i, i..S)?;
            if i != k {
                r += 1;
            }
            p.swap_rows(i, k);
            l.swap_rows_partial(i, k, 0..i);
            u.swap_rows(i, k);

            let d = u[i][i];
            if d.approximate_zero() {
                return None;
            }

            for row in (i + 1)..S {
                let n = u[row][i];

                let factor = n / d;
                l[row][i] = factor;
                for col in i..S {
                    let a = u[i][col];
                    u[row][col] -= factor * a;
                }
            }
        }
        Some((l, u, p, r))
    }

    /// Solves a lower-triangular system by forward substitution.
    pub fn solve_lower_triangular(&self, b: &Vector<T, S>) -> Vector<T, S> {
        let mut x = Vector::default();
        for row in 0..S {
            let r = &self[row];
            let mut sum = T::default();
            for col in 0..row {
                sum += r[col] * x[col];
            }
            x[row] = (b[row] - sum) / r[row];
        }
        x
    }

    /// Solves an upper-triangular system by back substitution.
    pub fn solve_upper_triangular(&self, b: &Vector<T, S>) -> Vector<T, S> {
        let mut x = Vector::default();
        for row in (0..S).rev() {
            let r = &self[row];
            let mut sum = T::default();
            for col in (row + 1)..S {
                sum += r[col] * x[col];
            }
            x[row] = (b[row] - sum) / r[row];
        }
        x
    }

    /// Solves a diagonal system by dividing each right-hand-side entry by the
    /// matching diagonal entry.
    pub fn solve_diagonal(&self, b: &Vector<T, S>) -> Vector<T, S> {
        let mut x = Vector::default();
        for row in 0..S {
            x[row] = b[row] / self[row][row];
        }
        x
    }

    /// Computes the Cholesky factor `L` such that `self = L * Lᵀ` for a
    /// symmetric positive-definite matrix. Returns `None` if a diagonal pivot
    /// is not positive.
    pub fn cholesky_decomposition(&self) -> Option<Matrix<T, S, S>> {
        let mut l = *self;
        for row in 0..S {
            for col in 0..=row {
                let mut sum = T::default();

                if col == row {
                    for k in 0..col {
                        sum += l[row][k] * l[col][k];
                    }

                    if sum > self[row][row] {
                        return None;
                    }

                    l[row][col] = (self[row][col] - sum).sqrt();
                } else {
                    for k in 0..col {
                        sum += l[row][k] * l[col][k];
                    }
                    l[row][col] = (self[row][col] - sum) / l[col][col];
                }
            }
        }
        Some(l)
    }

    /// Computes an `L * D * Lᵀ` factorization for a symmetric matrix, where
    /// `L` is unit lower-triangular and `D` is diagonal.
    pub fn ldl_decomposition(&self) -> Option<(Matrix<T, S, S>, Matrix<T, S, S>)> {
        let mut l = Matrix::identity();
        let mut d = Matrix::default();

        for row in 0..S {
            let mut sum = T::default();
            for col in 0..row {
                sum += l[row][col] * l[col][col];
            }
            d[row][row] = self[row][row] - sum;

            for i in (row + 1)..S {
                sum = T::default();
                for k in 0..row {
                    sum += l[i][k] * l[k][k];
                }
                l[i][row] = (self[i][row] - sum) / d[row][row];
            }
        }

        Some((l, d))
    }

    /// Solves `self * x = b`, selecting a triangular, Cholesky, LDL, or LUP
    /// method based on the matrix structure. Returns `None` if factorization
    /// fails.
    pub fn solve(&self, b: &Vector<T, S>) -> Option<Vector<T, S>> {
        let properties = self.properties();

        if properties.diagonal {
            return Some(self.solve_diagonal(b));
        }
        if properties.upper_triangular {
            return Some(self.solve_upper_triangular(b));
        }
        if properties.lower_triangular {
            return Some(self.solve_lower_triangular(b));
        }
        if properties.simmetric {
            if let Some(l) = self.cholesky_decomposition() {
                let y = l.solve_lower_triangular(b);
                let x = l.solve_upper_triangular(&y);
                return Some(x);
            } else if let Some((l, d)) = self.ldl_decomposition() {
                let y = l.solve_lower_triangular(b);
                let z = d.solve_upper_triangular(&y);
                let x = l.transpose().solve_upper_triangular(&z);
                return Some(x);
            }
        }
        let (l, u, p) = self.lup_decomposition()?;
        let b = p.dot_vector(b);
        let y = l.solve_lower_triangular(&b);
        let x = u.solve_upper_triangular(&y);
        Some(x)
    }

    /// Copies the main diagonal into a vector.
    pub fn extract_diagonal(&self) -> Vector<T, S> {
        let mut v = Vector::default();
        for i in 0..S {
            v[i] = self[i][i]
        }
        v
    }

    /// Computes the determinant using LUP factorization and permutation parity.
    pub fn determinant(&self) -> T {
        if let Some((_, u, _, r)) = self.lupr_decomposition() {
            let diagonal_product = u.extract_diagonal().product();
            return if r % 2 == 0 {
                diagonal_product
            } else {
                -diagonal_product
            };
        }
        T::default()
    }

    pub fn inverse(&self) -> Option<Matrix<T, S, S>> {
        let (l, u, p) = self.lup_decomposition()?;
        let mut inverse = Matrix::default();

        for col in 0..S {
            let mut e = Vector::<T, S>::default();
            e[col] = T::unit();

            let permuted_e = p.dot_vector(&e);
            let z = l.solve_lower_triangular(&permuted_e);
            let x = u.solve_upper_triangular(&z);

            inverse.set_column(col, &x);
        }

        Some(inverse)
    }
}

impl<T: VectorType, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Computes the reduced Householder QR factorization `self = Q * R`.
    /// `Q` has shape `R`-by-`C` and orthonormal columns; the returned upper-
    /// triangular factor has shape `C`-by-`C`. Returns `None` when the matrix
    /// is wider than it is tall or lacks full column rank.
    pub fn qr_decomposition(&self) -> Option<(Matrix<T, R, C>, Matrix<T, C, C>)> {
        if R < C {
            return None;
        }

        let mut work = *self;
        let mut q_full: Matrix<T, R, R> = Matrix::identity();

        for step in 0..C.min(R) {
            // Build a reflector for the active part of this column.
            let mut v = Vector::<T, R>::default();
            for row in step..R {
                v[row] = work[row][step];
            }
            let norm = v.magnitude();
            if norm.approximate_zero() {
                return None;
            }

            // Choose the sign to avoid cancellation when forming v.
            let alpha = if v[step] >= T::default() { -norm } else { norm };
            v[step] -= alpha;

            let v_norm_squared = v.magnitude_squared();
            if v_norm_squared.approximate_zero() {
                return None;
            }
            let two = T::unit() + T::unit();

            // Apply H = I - 2vvᵀ/(vᵀv) to the remaining columns of work.
            for col in step..C {
                let projection = v.dot(&work.get_column(col));
                let scale = two * projection / v_norm_squared;
                for row in step..R {
                    work[row][col] -= scale * v[row];
                }
            }

            // Accumulate Q = H₀ H₁ ... by applying each reflector on the right.
            for row in 0..R {
                let projection = q_full[row].dot(&v);
                let scale = two * projection / v_norm_squared;
                for col in step..R {
                    q_full[row][col] -= scale * v[col];
                }
            }
        }

        let mut q = Matrix::default();
        let mut r = Matrix::default();
        for row in 0..R {
            for col in 0..C {
                q[row][col] = q_full[row][col];
            }
        }
        for row in 0..C {
            for col in row..C {
                r[row][col] = work[row][col];
            }
        }
        Some((q, r))
    }

    /// Fits the least-squares coefficients for `self * coefficients ≈ y` and
    /// returns `(coefficients, sum_squared_residuals)`. Requires at least as
    /// many rows as columns and full column rank.
    pub fn least_squares(&self, y: &Vector<T, R>) -> Option<(Vector<T, C>, T)> {
        let (q, r) = self.qr_decomposition()?;
        let mut projected = Vector::<T, C>::default();
        for col in 0..C {
            projected[col] = q.get_column(col).dot(y);
        }
        let sol = r.solve_upper_triangular(&projected);
        let residual = *y - q.dot_vector(&projected);
        let loss = residual.magnitude_squared();
        Some((sol, loss))
    }
}
