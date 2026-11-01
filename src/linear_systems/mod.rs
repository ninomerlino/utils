//! Fixed-size vectors and matrices, plus common linear algebra operations.
//!
//! Dimensions are part of each type. For example, `Matrix<f64, 3, 2>` has
//! three rows and two columns, and multiplying it by `Vector<f64, 2>` produces
//! `Vector<f64, 3>`. Matrix and vector operators such as `*` are element-wise;
//! use [`Matrix::dot_matrix`] or [`Matrix::dot_vector`] for linear algebra
//! products.
//!
//! Square matrices provide linear-system solvers and decompositions. Rectangular
//! matrices provide reduced QR decomposition and least-squares fitting.
//!
//! ```
//! use utils::linear_systems::{Matrix, Vector};
//!
//! let a = Matrix::from([[2.0_f64, 1.0], [1.0, 3.0]]);
//! let b = Vector::from([5.0, 7.0]);
//! let x = a.solve(&b).unwrap();
//! ```

mod matrix;
mod systems;
mod vector;

pub use matrix::Matrix;
pub use vector::Vector;
