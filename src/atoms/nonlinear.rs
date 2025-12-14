//! Nonlinear atoms for convex optimization.
//!
//! These atoms have specific curvature properties (convex or concave)
//! and require DCP composition rules to be applied correctly.

use std::sync::Arc;

use crate::expr::Expr;

// ============================================================================
// Norms (all convex)
// ============================================================================

/// L1 norm: ||x||_1 = sum(|x_i|).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm1(x: &Expr) -> Expr {
    Expr::Norm1(Arc::new(x.clone()))
}

/// L2 norm: ||x||_2 = sqrt(sum(x_i^2)).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm2(x: &Expr) -> Expr {
    Expr::Norm2(Arc::new(x.clone()))
}

/// Infinity norm: ||x||_inf = max(|x_i|).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn norm_inf(x: &Expr) -> Expr {
    Expr::NormInf(Arc::new(x.clone()))
}

/// General p-norm: ||x||_p = (sum(|x_i|^p))^(1/p).
///
/// For p >= 1, this is convex. We support:
/// - p = 1: L1 norm
/// - p = 2: L2 norm
/// - p = f64::INFINITY: Infinity norm
pub fn norm(x: &Expr, p: f64) -> Expr {
    if p == 1.0 {
        norm1(x)
    } else if p == 2.0 {
        norm2(x)
    } else if p.is_infinite() {
        norm_inf(x)
    } else {
        // For other p values, we'd need a general pnorm atom
        // For now, only support 1, 2, inf
        panic!("Only p=1, 2, or inf supported for now");
    }
}

// ============================================================================
// Element-wise atoms
// ============================================================================

/// Absolute value: |x| (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
pub fn abs(x: &Expr) -> Expr {
    Expr::Abs(Arc::new(x.clone()))
}

/// Positive part: max(x, 0) (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing
pub fn pos(x: &Expr) -> Expr {
    Expr::Pos(Arc::new(x.clone()))
}

/// Negative part: max(-x, 0) (element-wise).
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Decreasing
pub fn neg_part(x: &Expr) -> Expr {
    Expr::NegPart(Arc::new(x.clone()))
}

// ============================================================================
// Maximum and minimum
// ============================================================================

/// Maximum of expressions (element-wise if same shape).
///
/// Properties:
/// - Curvature: Convex (when all arguments are convex)
/// - Sign: Depends on arguments
/// - Monotonicity: Increasing in all arguments
pub fn maximum(exprs: Vec<Expr>) -> Expr {
    if exprs.len() == 1 {
        return exprs.into_iter().next().unwrap();
    }
    Expr::Maximum(exprs.into_iter().map(Arc::new).collect())
}

/// Maximum of two expressions.
pub fn max2(a: &Expr, b: &Expr) -> Expr {
    maximum(vec![a.clone(), b.clone()])
}

/// Minimum of expressions (element-wise if same shape).
///
/// Properties:
/// - Curvature: Concave (when all arguments are concave)
/// - Sign: Depends on arguments
/// - Monotonicity: Increasing in all arguments
pub fn minimum(exprs: Vec<Expr>) -> Expr {
    if exprs.len() == 1 {
        return exprs.into_iter().next().unwrap();
    }
    Expr::Minimum(exprs.into_iter().map(Arc::new).collect())
}

/// Minimum of two expressions.
pub fn min2(a: &Expr, b: &Expr) -> Expr {
    minimum(vec![a.clone(), b.clone()])
}

// ============================================================================
// Quadratic atoms
// ============================================================================

/// Quadratic form: x' P x.
///
/// Properties:
/// - Curvature: Convex if P is PSD, Concave if P is NSD
/// - Sign: Non-negative if P is PSD, Non-positive if P is NSD
/// - Arguments: x must be affine, P must be constant symmetric
pub fn quad_form(x: &Expr, p: &Expr) -> Expr {
    Expr::QuadForm(Arc::new(x.clone()), Arc::new(p.clone()))
}

/// Sum of squares: ||x||_2^2 = x' x.
///
/// Properties:
/// - Curvature: Convex
/// - Sign: Non-negative
/// - Monotonicity: Increasing for x >= 0, decreasing for x <= 0
///
/// This is equivalent to quad_form(x, I) where I is identity.
pub fn sum_squares(x: &Expr) -> Expr {
    Expr::SumSquares(Arc::new(x.clone()))
}

/// Quadratic over linear: ||x||_2^2 / y.
///
/// Properties:
/// - Curvature: Convex (when x is affine and y is concave and positive)
/// - Sign: Non-negative
/// - Domain: y > 0
///
/// This is a perspective function and is jointly convex in (x, y).
pub fn quad_over_lin(x: &Expr, y: &Expr) -> Expr {
    Expr::QuadOverLin(Arc::new(x.clone()), Arc::new(y.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcp::Curvature;
    use crate::expr::{constant, variable};

    #[test]
    fn test_norm2_convex() {
        let x = variable(5);
        let n = norm2(&x);
        assert_eq!(n.curvature(), Curvature::Convex);
        assert!(n.is_nonneg());
    }

    #[test]
    fn test_norm1_convex() {
        let x = variable(5);
        let n = norm1(&x);
        assert_eq!(n.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_abs_convex() {
        let x = variable(5);
        let a = abs(&x);
        assert_eq!(a.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_pos_convex() {
        let x = variable(5);
        let p = pos(&x);
        assert_eq!(p.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_maximum_convex() {
        let x = variable(5);
        let y = variable(5);
        let m = maximum(vec![x, y]);
        assert_eq!(m.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_minimum_concave() {
        let x = variable(5);
        let y = variable(5);
        let m = minimum(vec![x, y]);
        assert_eq!(m.curvature(), Curvature::Concave);
    }

    #[test]
    fn test_sum_squares_convex() {
        let x = variable(5);
        let s = sum_squares(&x);
        assert_eq!(s.curvature(), Curvature::Convex);
        assert!(s.is_nonneg());
    }

    #[test]
    fn test_quad_form_psd() {
        use nalgebra::DMatrix;
        let x = variable(2);
        // Identity matrix is PSD
        let p = crate::expr::constant_dmatrix(DMatrix::identity(2, 2));
        let q = quad_form(&x, &p);
        assert_eq!(q.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_norm_of_affine_is_convex() {
        let x = variable(5);
        let y = variable(5);
        // x + y is affine
        let z = &x + &y;
        let n = norm2(&z);
        // norm2 of affine is convex
        assert_eq!(n.curvature(), Curvature::Convex);
    }

    #[test]
    fn test_norm_of_convex_is_unknown() {
        let x = variable(5);
        let n1 = norm2(&x);
        // norm2(norm2(x)) - norm of convex is unknown (not DCP)
        let n2 = norm2(&n1);
        assert_eq!(n2.curvature(), Curvature::Unknown);
    }
}
