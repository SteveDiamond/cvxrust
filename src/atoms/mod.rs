//! Atom functions for building expressions.
//!
//! Atoms are the building blocks of optimization problems. They include:
//!
//! - **Affine atoms**: Operations that preserve linearity (add, mul, sum, reshape, etc.)
//! - **Nonlinear atoms**: Operations with specific curvature (norms, quadratic forms, etc.)

pub mod affine;
pub mod nonlinear;

// Re-export affine operations
pub use affine::{
    dot, flatten, hstack, index, matmul, reshape, slice, sum, sum_axis, trace, transpose, vstack,
};

// Re-export nonlinear atoms
pub use nonlinear::{
    abs, max2, maximum, min2, minimum, neg_part, norm, norm1, norm2, norm_inf, pos, quad_form,
    quad_over_lin, sum_squares,
};
