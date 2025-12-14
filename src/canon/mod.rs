//! Canonicalization transforms expressions into standard form.
//!
//! This module converts DCP expressions into:
//! - Linear expressions (LinExpr) for affine parts
//! - Quadratic expressions (QuadExpr) for QP objectives
//! - Cone constraints (ConeConstraint) for nonlinear atoms

pub mod canonicalizer;
pub mod lin_expr;

pub use canonicalizer::{canonicalize, CanonExpr, CanonResult, ConeConstraint};
pub use lin_expr::{LinExpr, QuadExpr};
