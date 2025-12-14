//! Sparse matrix utilities.
//!
//! Helper functions for working with nalgebra-sparse matrices.

use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Create a CSC matrix from triplets (row, col, value).
///
/// Duplicates are summed together.
pub fn csc_from_triplets(
    nrows: usize,
    ncols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
) -> CscMatrix<f64> {
    if rows.is_empty() {
        return CscMatrix::zeros(nrows, ncols);
    }

    // Build COO matrix first
    let mut coo = CooMatrix::new(nrows, ncols);
    for ((row, col), val) in rows.into_iter().zip(cols).zip(vals) {
        if row < nrows && col < ncols {
            coo.push(row, col, val);
        }
    }

    // Convert to CSC
    CscMatrix::from(&coo)
}

/// Create a CSC identity matrix.
pub fn csc_identity(n: usize) -> CscMatrix<f64> {
    CscMatrix::identity(n)
}

/// Convert a dense matrix to CSC format.
pub fn dense_to_csc(dense: &DMatrix<f64>) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for j in 0..dense.ncols() {
        for i in 0..dense.nrows() {
            let v = dense[(i, j)];
            if v.abs() > 1e-15 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
    }

    csc_from_triplets(dense.nrows(), dense.ncols(), rows, cols, vals)
}

/// Convert CSC to dense matrix.
pub fn csc_to_dense(sparse: &CscMatrix<f64>) -> DMatrix<f64> {
    let mut dense = DMatrix::zeros(sparse.nrows(), sparse.ncols());
    for (row, col, val) in sparse.triplet_iter() {
        dense[(row, col)] = *val;
    }
    dense
}

/// Stack two CSC matrices vertically.
pub fn csc_vstack(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in a.triplet_iter() {
        rows.push(r);
        cols.push(c);
        vals.push(*v);
    }
    for (r, c, v) in b.triplet_iter() {
        rows.push(r + a.nrows());
        cols.push(c);
        vals.push(*v);
    }

    csc_from_triplets(
        a.nrows() + b.nrows(),
        a.ncols().max(b.ncols()),
        rows,
        cols,
        vals,
    )
}

/// Add two CSC matrices.
/// Uses nalgebra_sparse's built-in addition which is more efficient.
pub fn csc_add(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    a + b
}

/// Negate a CSC matrix.
pub fn csc_neg(a: &CscMatrix<f64>) -> CscMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|v| -v).collect();
    let col_offsets: Vec<usize> = a.col_offsets().to_vec();
    let row_indices: Vec<usize> = a.row_indices().to_vec();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), col_offsets, row_indices, values)
        .unwrap_or_else(|_| CscMatrix::zeros(a.nrows(), a.ncols()))
}

/// Scale a CSC matrix.
pub fn csc_scale(a: &CscMatrix<f64>, scalar: f64) -> CscMatrix<f64> {
    let values: Vec<f64> = a.values().iter().map(|v| v * scalar).collect();
    let col_offsets: Vec<usize> = a.col_offsets().to_vec();
    let row_indices: Vec<usize> = a.row_indices().to_vec();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), col_offsets, row_indices, values)
        .unwrap_or_else(|_| CscMatrix::zeros(a.nrows(), a.ncols()))
}

/// Multiply sparse matrix by dense matrix on the right: A_sparse @ B_dense
pub fn sparse_dense_matmul(a: &CscMatrix<f64>, b: &DMatrix<f64>) -> CscMatrix<f64> {
    // Convert sparse to dense, multiply, convert back
    // This is not optimal but correct
    let a_dense = csc_to_dense(a);
    let result = &a_dense * b;
    dense_to_csc(&result)
}

/// Stack two CSC matrices horizontally.
pub fn csc_hstack(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in a.triplet_iter() {
        rows.push(r);
        cols.push(c);
        vals.push(*v);
    }
    for (r, c, v) in b.triplet_iter() {
        rows.push(r);
        cols.push(c + a.ncols());
        vals.push(*v);
    }

    csc_from_triplets(
        a.nrows().max(b.nrows()),
        a.ncols() + b.ncols(),
        rows,
        cols,
        vals,
    )
}

/// Repeat rows of a CSC matrix.
pub fn csc_repeat_rows(m: &CscMatrix<f64>, times: usize) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (r, c, v) in m.triplet_iter() {
        for t in 0..times {
            rows.push(t * m.nrows() + r);
            cols.push(c);
            vals.push(*v);
        }
    }

    csc_from_triplets(m.nrows() * times, m.ncols(), rows, cols, vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_from_triplets() {
        let m = csc_from_triplets(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_csc_identity() {
        let m = csc_identity(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_dense_to_csc() {
        let dense = DMatrix::identity(3, 3);
        let sparse = dense_to_csc(&dense);
        assert_eq!(sparse.nrows(), 3);
    }

    #[test]
    fn test_csc_add() {
        let a = CscMatrix::<f64>::identity(3);
        let b = CscMatrix::<f64>::identity(3);
        let c = csc_add(&a, &b);
        // Identity + Identity = 2*Identity, so diagonal values should be 2.0
        assert_eq!(c.values().len(), 3);
        for v in c.values() {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }
}
