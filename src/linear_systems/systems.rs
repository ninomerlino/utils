use crate::linear_systems::vector::RealVector;

use super::{Matrix, Vector, vector::VectorType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixProperties {
    pub simmetric: bool,
    pub diagonal: bool,
    pub upper_triangular: bool,
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

impl MatrixProperties {
    pub fn all(&self) -> bool {
        self.simmetric && self.diagonal && self.upper_triangular && self.lower_triangular
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn any(&self) -> bool {
        self.simmetric || self.diagonal || self.upper_triangular || self.lower_triangular
    }
}

impl<T: VectorType, const S: usize> Matrix<T, S, S> {
    pub fn identity() -> Self {
        let mut mat: Matrix<T, S, S> = Matrix::default();
        for row in 0..S {
            mat[row][row] = T::unit();
        }
        mat
    }

    pub fn from_dot(a: &Vector<T, S>, b: &Vector<T, S>) -> Self {
        let mut mat: Matrix<T, S, S> = Matrix::default();
        for (row_i, row) in mat.iter_mut().enumerate() {
            *row = *b * a[row_i];
        }
        mat
    }

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

    pub fn transpose(&self) -> Matrix<T, S, S> {
        let mut result = self.clone()
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

impl<T: RealVector, const S: usize> Matrix<T, S, S> {
    pub fn lup_decomposition(&self) -> Option<(Matrix<T, S, S>, Matrix<T, S, S>, Matrix<T, S, S>)> {
        let mut p: Matrix<T, S, S> = Matrix::identity();
        let mut l: Matrix<T, S, S> = Matrix::identity();
        let mut u: Matrix<T, S, S> = self.clone();

        for i in 0..S {
            let (_, k) = u.abs_max_from_col_partial(i, i..S)?;
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
        Some((l, u, p))
    }

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

    pub fn solve_diagonal(&self, b: &Vector<T, S>) -> Vector<T, S> {
        let mut x = Vector::default();
        for row in 0..S {
            x[row] = b[row] / self[row][row];
        }
        x
    }

    pub fn cholesky_decomposition(&self) -> Option<Matrix<T, S, S>> {
        let mut l = self.clone();
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
                } else {
                    for k in 0..col {
                        sum += l[row][k] * l[col][k];
                    }
                    l[row][col] = (self[row][col] - sum) / l[col][col];
                }
            }
        }
        Some(l)
    }

    pub fn ldl_decomposition(&self) -> Option<(Matrix<T, S, S>, Matrix<T, S, S>)> {
        let mut l = Matrix::identity();
        let mut d = Matrix::default();

        for row in 0..S {

            let mut sum = T::default();
            for col in 0..row {
                sum += l[row][col] * l[col][col];
            }
            d[row][row] = self[row][row] - sum;

            for i in (row+1)..S {
                sum = T::default();
                for k in 0..row {
                    sum += l[i][k] * l[k][k];
                }
                l[i][row] = (self[i][row] - sum) / d[row][row];
            }

        }

        Some((l, d))
    }

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
}

impl<T: RealVector, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn qr_decomposition(&self) -> Option<(Matrix<T, R, R>, Matrix<T, R, C>)> {
        todo!()
    }
    pub fn least_squares(&self, teta: &Vector<T, R>) -> Vector<T, C> {
        todo!()
    }
}
