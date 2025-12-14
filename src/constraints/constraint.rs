//! Constraint types for optimization problems.
//!
//! Constraints map to cone constraints in the solver:
//! - Zero: Ax + b = 0 (zero cone / equality)
//! - NonNeg: Ax + b >= 0 (nonnegative orthant)
//! - SOC: ||x|| <= t (second-order cone)

use std::sync::Arc;

use crate::expr::Expr;

/// A constraint in an optimization problem.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Equality constraint: expr == 0.
    /// Maps to the zero cone.
    Zero(Arc<Expr>),

    /// Inequality constraint: expr >= 0.
    /// Maps to the nonnegative orthant cone.
    NonNeg(Arc<Expr>),

    /// Second-order cone constraint: ||x||_2 <= t.
    /// The t argument must be scalar, x can be a vector.
    SOC {
        /// The scalar upper bound.
        t: Arc<Expr>,
        /// The vector argument.
        x: Arc<Expr>,
    },
}

impl Constraint {
    /// Create an equality constraint: lhs == rhs.
    pub fn eq(lhs: Expr, rhs: Expr) -> Self {
        Constraint::Zero(Arc::new(Expr::Add(
            Arc::new(lhs),
            Arc::new(Expr::Neg(Arc::new(rhs))),
        )))
    }

    /// Create an inequality constraint: lhs <= rhs.
    pub fn leq(lhs: Expr, rhs: Expr) -> Self {
        // lhs <= rhs  <=>  rhs - lhs >= 0
        Constraint::NonNeg(Arc::new(Expr::Add(
            Arc::new(rhs),
            Arc::new(Expr::Neg(Arc::new(lhs))),
        )))
    }

    /// Create an inequality constraint: lhs >= rhs.
    pub fn geq(lhs: Expr, rhs: Expr) -> Self {
        // lhs >= rhs  <=>  lhs - rhs >= 0
        Constraint::NonNeg(Arc::new(Expr::Add(
            Arc::new(lhs),
            Arc::new(Expr::Neg(Arc::new(rhs))),
        )))
    }

    /// Create a SOC constraint: ||x||_2 <= t.
    pub fn soc(t: Expr, x: Expr) -> Self {
        Constraint::SOC {
            t: Arc::new(t),
            x: Arc::new(x),
        }
    }

    /// Check if this constraint is DCP-compliant.
    ///
    /// DCP rules for constraints:
    /// - Zero: expression must be affine (equality of affine expressions)
    /// - NonNeg: expression must be concave (concave >= 0)
    /// - SOC: both t and x must be affine
    pub fn is_dcp(&self) -> bool {
        match self {
            Constraint::Zero(expr) => expr.is_affine(),
            Constraint::NonNeg(expr) => expr.is_concave(),
            Constraint::SOC { t, x } => t.is_affine() && x.is_affine(),
        }
    }

    /// Get all expressions in this constraint.
    pub fn expressions(&self) -> Vec<&Expr> {
        match self {
            Constraint::Zero(e) => vec![e.as_ref()],
            Constraint::NonNeg(e) => vec![e.as_ref()],
            Constraint::SOC { t, x } => vec![t.as_ref(), x.as_ref()],
        }
    }

    /// Get all variable IDs in this constraint.
    pub fn variables(&self) -> Vec<crate::expr::ExprId> {
        let mut vars = Vec::new();
        for expr in self.expressions() {
            vars.extend(expr.variables());
        }
        vars.sort_by_key(|id| id.raw());
        vars.dedup();
        vars
    }
}

/// Extension trait for creating constraints from expressions.
pub trait ConstraintExt {
    /// Create equality constraint: self == rhs.
    fn equals(&self, rhs: &Expr) -> Constraint;

    /// Create inequality constraint: self <= rhs.
    fn leq(&self, rhs: &Expr) -> Constraint;

    /// Create inequality constraint: self >= rhs.
    fn geq(&self, rhs: &Expr) -> Constraint;
}

impl ConstraintExt for Expr {
    fn equals(&self, rhs: &Expr) -> Constraint {
        Constraint::eq(self.clone(), rhs.clone())
    }

    fn leq(&self, rhs: &Expr) -> Constraint {
        Constraint::leq(self.clone(), rhs.clone())
    }

    fn geq(&self, rhs: &Expr) -> Constraint {
        Constraint::geq(self.clone(), rhs.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, variable};

    #[test]
    fn test_equality_constraint() {
        let x = variable(5);
        let c = constant(1.0);
        let constr = Constraint::eq(x, c);

        assert!(constr.is_dcp());
        if let Constraint::Zero(_) = constr {
            // OK
        } else {
            panic!("Expected Zero constraint");
        }
    }

    #[test]
    fn test_inequality_constraint() {
        let x = variable(5);
        let c = constant(0.0);
        let constr = Constraint::geq(x, c);

        assert!(constr.is_dcp());
        if let Constraint::NonNeg(_) = constr {
            // OK
        } else {
            panic!("Expected NonNeg constraint");
        }
    }

    #[test]
    fn test_soc_constraint() {
        let t = variable(());
        let x = variable(5);
        let constr = Constraint::soc(t, x);

        assert!(constr.is_dcp());
    }

    #[test]
    fn test_non_dcp_constraint() {
        let x = variable(5);
        // norm(x) >= 1 is NOT DCP (convex >= constant)
        let norm_x = Expr::Norm2(Arc::new(x));
        let c = constant(1.0);
        let constr = Constraint::geq(norm_x, c);

        // norm_x is convex, so norm_x - 1 is convex, not concave
        assert!(!constr.is_dcp());
    }

    #[test]
    fn test_constraint_ext() {
        let x = variable(5);
        let c = constant(1.0);

        let eq_constr = x.equals(&c);
        assert!(eq_constr.is_dcp());

        let x = variable(5);
        let leq_constr = x.leq(&c);
        assert!(leq_constr.is_dcp());
    }
}
