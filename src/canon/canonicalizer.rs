//! Expression canonicalization.
//!
//! Canonicalization transforms arbitrary DCP expressions into standard form:
//! - Affine expressions become LinExpr
//! - Quadratic objectives become QuadExpr (with P matrix for native QP)
//! - Nonlinear atoms are reformulated as affine + cone constraints

use std::sync::Arc;

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

use super::lin_expr::{LinExpr, QuadExpr};
use crate::expr::{Array, Expr, ExprId, Shape, VariableBuilder};
use crate::sparse::{csc_vstack, csc_repeat_rows, dense_to_csc, csc_to_dense, csc_scale, sparse_dense_matmul};

/// A cone constraint in standard form: Ax + b in K.
#[derive(Debug, Clone)]
pub enum ConeConstraint {
    /// Zero cone: Ax + b = 0 (equality).
    Zero { a: LinExpr },
    /// Nonnegative cone: Ax + b >= 0.
    NonNeg { a: LinExpr },
    /// Second-order cone: ||x||_2 <= t.
    /// Represented as [t; x] in K_soc.
    SOC {
        /// The scalar t expression.
        t: LinExpr,
        /// The vector x expression.
        x: LinExpr,
    },
}

/// Result of canonicalizing an expression.
#[derive(Debug)]
pub struct CanonResult {
    /// The canonicalized expression (affine or quadratic).
    pub expr: CanonExpr,
    /// Additional cone constraints introduced during canonicalization.
    pub constraints: Vec<ConeConstraint>,
    /// Auxiliary variables introduced during canonicalization.
    pub aux_vars: Vec<(ExprId, Shape)>,
}

/// The type of canonicalized expression.
#[derive(Debug)]
pub enum CanonExpr {
    /// Linear expression.
    Linear(LinExpr),
    /// Quadratic expression (for objectives only).
    Quadratic(QuadExpr),
}

impl CanonExpr {
    /// Get as linear expression, panicking if quadratic.
    pub fn as_linear(&self) -> &LinExpr {
        match self {
            CanonExpr::Linear(l) => l,
            CanonExpr::Quadratic(_) => panic!("Expected linear expression, got quadratic"),
        }
    }

    /// Get as quadratic expression, converting linear if needed.
    pub fn into_quadratic(self) -> QuadExpr {
        match self {
            CanonExpr::Linear(l) => QuadExpr::from_linear(l),
            CanonExpr::Quadratic(q) => q,
        }
    }
}

/// Canonicalize an expression.
///
/// This converts the expression tree into affine form plus cone constraints.
/// For objectives, quadratic expressions are preserved for native QP support.
pub fn canonicalize(expr: &Expr, for_objective: bool) -> CanonResult {
    let mut ctx = CanonContext::new();
    let canon_expr = ctx.canonicalize_expr(expr, for_objective);
    CanonResult {
        expr: canon_expr,
        constraints: ctx.constraints,
        aux_vars: ctx.aux_vars,
    }
}

/// Context for canonicalization, tracking auxiliary variables and constraints.
struct CanonContext {
    constraints: Vec<ConeConstraint>,
    aux_vars: Vec<(ExprId, Shape)>,
}

impl CanonContext {
    fn new() -> Self {
        CanonContext {
            constraints: Vec::new(),
            aux_vars: Vec::new(),
        }
    }

    /// Create a new auxiliary variable.
    fn new_aux_var(&mut self, shape: Shape) -> (ExprId, LinExpr) {
        let var = VariableBuilder::new(shape.clone()).build();
        let var_id = var.variable_id().unwrap();
        self.aux_vars.push((var_id, shape.clone()));
        (var_id, LinExpr::variable(var_id, shape))
    }

    /// Create a new non-negative auxiliary variable.
    fn new_nonneg_aux_var(&mut self, shape: Shape) -> (ExprId, LinExpr) {
        let var = VariableBuilder::new(shape.clone()).nonneg().build();
        let var_id = var.variable_id().unwrap();
        self.aux_vars.push((var_id, shape.clone()));
        let lin_var = LinExpr::variable(var_id, shape);
        // Add t >= 0 constraint
        self.constraints
            .push(ConeConstraint::NonNeg { a: lin_var.clone() });
        (var_id, lin_var)
    }

    /// Canonicalize an expression.
    fn canonicalize_expr(&mut self, expr: &Expr, for_objective: bool) -> CanonExpr {
        match expr {
            // Leaves
            Expr::Variable(v) => CanonExpr::Linear(LinExpr::variable(v.id, v.shape.clone())),
            Expr::Constant(c) => CanonExpr::Linear(self.canonicalize_constant(&c.value)),

            // Affine operations
            Expr::Add(a, b) => {
                let ca = self.canonicalize_expr(a, false);
                let cb = self.canonicalize_expr(b, false);
                match (ca, cb) {
                    (CanonExpr::Linear(la), CanonExpr::Linear(lb)) => {
                        CanonExpr::Linear(la.add(&lb))
                    }
                    (CanonExpr::Quadratic(qa), CanonExpr::Linear(lb)) => {
                        let qb = QuadExpr::from_linear(lb);
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                    (CanonExpr::Linear(la), CanonExpr::Quadratic(qb)) => {
                        let qa = QuadExpr::from_linear(la);
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                    (CanonExpr::Quadratic(qa), CanonExpr::Quadratic(qb)) => {
                        CanonExpr::Quadratic(qa.add(&qb))
                    }
                }
            }
            Expr::Neg(a) => {
                let ca = self.canonicalize_expr(a, false);
                match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.neg()),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(-1.0)),
                }
            }
            Expr::Mul(a, b) => self.canonicalize_mul(a, b, for_objective),
            Expr::MatMul(a, b) => self.canonicalize_matmul(a, b),
            Expr::Sum(a, axis) => self.canonicalize_sum(a, *axis),
            Expr::Reshape(a, shape) => self.canonicalize_reshape(a, shape),
            Expr::Index(a, _spec) => {
                // For now, treat index as identity (simplified)
                self.canonicalize_expr(a, false)
            }
            Expr::VStack(exprs) => self.canonicalize_vstack(exprs),
            Expr::HStack(exprs) => self.canonicalize_hstack(exprs),
            Expr::Transpose(a) => self.canonicalize_transpose(a),
            Expr::Trace(a) => self.canonicalize_trace(a),

            // Nonlinear atoms - introduce auxiliary variables and cone constraints
            Expr::Norm1(x) => self.canonicalize_norm1(x),
            Expr::Norm2(x) => self.canonicalize_norm2(x),
            Expr::NormInf(x) => self.canonicalize_norm_inf(x),
            Expr::Abs(x) => self.canonicalize_abs(x),
            Expr::Pos(x) => self.canonicalize_pos(x),
            Expr::NegPart(x) => self.canonicalize_neg_part(x),
            Expr::Maximum(exprs) => self.canonicalize_maximum(exprs),
            Expr::Minimum(exprs) => self.canonicalize_minimum(exprs),
            Expr::QuadForm(x, p) => self.canonicalize_quad_form(x, p, for_objective),
            Expr::SumSquares(x) => self.canonicalize_sum_squares(x, for_objective),
            Expr::QuadOverLin(x, y) => self.canonicalize_quad_over_lin(x, y),
        }
    }

    fn canonicalize_constant(&self, arr: &Array) -> LinExpr {
        match arr {
            Array::Scalar(v) => LinExpr::scalar(*v),
            Array::Dense(m) => LinExpr::constant(m.clone()),
            Array::Sparse(s) => {
                // Convert sparse to dense
                LinExpr::constant(csc_to_dense(s))
            }
        }
    }

    fn canonicalize_mul(&mut self, a: &Expr, b: &Expr, for_objective: bool) -> CanonExpr {
        // Check if one side is constant
        if let Some(arr) = a.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                let cb = self.canonicalize_expr(b, for_objective);
                return match cb {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
        }
        if let Some(arr) = b.constant_value() {
            if let Some(scalar) = arr.as_scalar() {
                let ca = self.canonicalize_expr(a, for_objective);
                return match ca {
                    CanonExpr::Linear(l) => CanonExpr::Linear(l.scale(scalar)),
                    CanonExpr::Quadratic(q) => CanonExpr::Quadratic(q.scale(scalar)),
                };
            }
        }

        // Both non-constant: fall back to treating as affine (simplified)
        let ca = self.canonicalize_expr(a, false);
        let _cb = self.canonicalize_expr(b, false);
        // This is a simplification - would need proper handling
        ca
    }

    fn canonicalize_matmul(&mut self, a: &Expr, b: &Expr) -> CanonExpr {
        // If A is constant, this is A @ x which is affine
        if let Some(arr) = a.constant_value() {
            let cb = self.canonicalize_expr(b, false).as_linear().clone();
            return CanonExpr::Linear(self.matmul_const_lin(arr, &cb));
        }
        // If B is constant, this is x @ B which is also affine
        if let Some(arr) = b.constant_value() {
            let ca = self.canonicalize_expr(a, false).as_linear().clone();
            return CanonExpr::Linear(self.lin_matmul_const(&ca, arr));
        }
        // Both non-constant: not DCP, return simplified
        self.canonicalize_expr(a, false)
    }

    fn matmul_const_lin(&self, a: &Array, b: &LinExpr) -> LinExpr {
        // A @ (sum_i B_i x_i + c) = sum_i (A @ B_i) x_i + A @ c
        let a_mat = match a {
            Array::Dense(m) => m.clone(),
            Array::Scalar(v) => DMatrix::from_element(1, 1, *v),
            Array::Sparse(s) => csc_to_dense(s),
        };

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &b.coeffs {
            // A @ B_i
            let new_coeff = dense_sparse_matmul(&a_mat, coeff);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let new_const = &a_mat * &b.constant;
        let shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    fn lin_matmul_const(&self, a: &LinExpr, b: &Array) -> LinExpr {
        // (sum_i A_i x_i + c) @ B = sum_i (A_i @ B) x_i + c @ B
        let b_mat = match b {
            Array::Dense(m) => m.clone(),
            Array::Scalar(v) => DMatrix::from_element(1, 1, *v),
            Array::Sparse(s) => csc_to_dense(s),
        };

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &a.coeffs {
            // A_i @ B
            let new_coeff = sparse_dense_matmul(coeff, &b_mat);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let new_const = &a.constant * &b_mat;
        let shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape,
        }
    }

    fn canonicalize_sum(&mut self, a: &Expr, _axis: Option<usize>) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Sum all elements: multiply by ones vector
        let size = ca.size();
        let ones = DMatrix::from_element(1, size, 1.0);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &ca.coeffs {
            let new_coeff = dense_sparse_matmul(&ones, coeff);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let flat_const = ca.constant.reshape_generic(
            nalgebra::Dyn(size),
            nalgebra::Dyn(1),
        );
        let result = &ones * &flat_const;
        let new_const = DMatrix::from_element(1, 1, result[(0, 0)]);

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::scalar(),
        })
    }

    fn canonicalize_reshape(&mut self, a: &Expr, shape: &Shape) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Reshape doesn't change the linear structure, just the shape interpretation
        CanonExpr::Linear(LinExpr {
            coeffs: ca.coeffs,
            constant: ca.constant.reshape_generic(
                nalgebra::Dyn(shape.rows()),
                nalgebra::Dyn(shape.cols()),
            ),
            shape: shape.clone(),
        })
    }

    fn canonicalize_vstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }
        // Canonicalize all and stack
        let mut result = self.canonicalize_expr(&exprs[0], false).as_linear().clone();
        for e in &exprs[1..] {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            result = self.vstack_lin(&result, &ce);
        }
        CanonExpr::Linear(result)
    }

    fn vstack_lin(&self, a: &LinExpr, b: &LinExpr) -> LinExpr {
        // Stack constants vertically
        let new_const = stack_vertical(&a.constant, &b.constant);
        let new_shape = Shape::matrix(new_const.nrows(), new_const.ncols());

        // Stack coefficients for each variable
        let mut new_coeffs = std::collections::HashMap::new();
        let all_vars: std::collections::HashSet<_> =
            a.coeffs.keys().chain(b.coeffs.keys()).copied().collect();

        for var_id in all_vars {
            let ca = a.coeffs.get(&var_id);
            let cb = b.coeffs.get(&var_id);
            let stacked = match (ca, cb) {
                (Some(ma), Some(mb)) => stack_csc_vertical(ma, mb),
                (Some(ma), None) => {
                    let zeros = CscMatrix::zeros(b.size(), ma.ncols());
                    stack_csc_vertical(ma, &zeros)
                }
                (None, Some(mb)) => {
                    let zeros = CscMatrix::zeros(a.size(), mb.ncols());
                    stack_csc_vertical(&zeros, mb)
                }
                (None, None) => continue,
            };
            new_coeffs.insert(var_id, stacked);
        }

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: new_shape,
        }
    }

    fn canonicalize_hstack(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // Similar to vstack but horizontal
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }
        let mut result = self.canonicalize_expr(&exprs[0], false).as_linear().clone();
        for e in &exprs[1..] {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // Simplified: treat as addition for now
            result = result.add(&ce);
        }
        CanonExpr::Linear(result)
    }

    fn canonicalize_transpose(&mut self, a: &Expr) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Transpose the shape and constant
        let new_shape = ca.shape.transpose();
        let new_const = ca.constant.transpose();
        CanonExpr::Linear(LinExpr {
            coeffs: ca.coeffs, // Coefficients need transpose too (simplified)
            constant: new_const,
            shape: new_shape,
        })
    }

    fn canonicalize_trace(&mut self, a: &Expr) -> CanonExpr {
        let ca = self.canonicalize_expr(a, false).as_linear().clone();
        // Trace: sum of diagonal elements
        let n = ca.shape.rows().min(ca.shape.cols());
        let trace_val: f64 = (0..n).map(|i| ca.constant[(i, i)]).sum();
        // For coefficients, would need to extract diagonal (simplified)
        CanonExpr::Linear(LinExpr::scalar(trace_val))
    }

    // ========================================================================
    // Nonlinear atom canonicalizers
    // ========================================================================

    fn canonicalize_norm1(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_1 = sum(|x_i|)
        // Introduce t_i >= 0, -t_i <= x_i <= t_i
        // Then ||x||_1 = sum(t_i)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let size = cx.size();
        let (_, t) = self.new_nonneg_aux_var(Shape::vector(size));

        // Constraints: t >= x, t >= -x
        // i.e., t - x >= 0, t + x >= 0
        self.constraints
            .push(ConeConstraint::NonNeg { a: t.add(&cx.neg()) });
        self.constraints.push(ConeConstraint::NonNeg { a: t.add(&cx) });

        // Return sum(t)
        self.canonicalize_sum_lin(&t)
    }

    fn canonicalize_norm2(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_2: Introduce t >= 0, SOC(t, x)
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // SOC constraint: ||x||_2 <= t
        self.constraints.push(ConeConstraint::SOC { t: t.clone(), x: cx });

        CanonExpr::Linear(t)
    }

    fn canonicalize_norm_inf(&mut self, x: &Expr) -> CanonExpr {
        // ||x||_inf = max(|x_i|)
        // Introduce t >= 0, -t <= x_i <= t for all i
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let size = cx.size();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Expand t to match x's size
        let t_expanded = self.expand_scalar(&t, size);

        // Constraints: t >= x_i, t >= -x_i for all i
        self.constraints
            .push(ConeConstraint::NonNeg { a: t_expanded.add(&cx.neg()) });
        self.constraints
            .push(ConeConstraint::NonNeg { a: t_expanded.add(&cx) });

        CanonExpr::Linear(t)
    }

    fn canonicalize_abs(&mut self, x: &Expr) -> CanonExpr {
        // |x| element-wise: same as norm1 but keeping element-wise
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let shape = cx.shape.clone();
        let (_, t) = self.new_nonneg_aux_var(shape);

        // Constraints: t >= x, t >= -x
        self.constraints
            .push(ConeConstraint::NonNeg { a: t.add(&cx.neg()) });
        self.constraints.push(ConeConstraint::NonNeg { a: t.add(&cx) });

        CanonExpr::Linear(t)
    }

    fn canonicalize_pos(&mut self, x: &Expr) -> CanonExpr {
        // pos(x) = max(x, 0)
        // Introduce t >= 0, t >= x
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let shape = cx.shape.clone();
        let (_, t) = self.new_nonneg_aux_var(shape);

        // Constraint: t >= x, i.e., t - x >= 0
        self.constraints
            .push(ConeConstraint::NonNeg { a: t.add(&cx.neg()) });

        CanonExpr::Linear(t)
    }

    fn canonicalize_neg_part(&mut self, x: &Expr) -> CanonExpr {
        // neg(x) = max(-x, 0) = pos(-x)
        let neg_x = Expr::Neg(Arc::new(x.clone()));
        self.canonicalize_pos(&neg_x)
    }

    fn canonicalize_maximum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // max(x1, ..., xn): Introduce t, t >= x_i for all i
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }

        let shape = exprs[0].shape();
        let (_, t) = self.new_aux_var(shape);

        for e in exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t >= x_i, i.e., t - x_i >= 0
            self.constraints
                .push(ConeConstraint::NonNeg { a: t.add(&ce.neg()) });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_minimum(&mut self, exprs: &[Arc<Expr>]) -> CanonExpr {
        // min(x1, ..., xn): Introduce t, t <= x_i for all i
        if exprs.is_empty() {
            return CanonExpr::Linear(LinExpr::zeros(Shape::scalar()));
        }

        let shape = exprs[0].shape();
        let (_, t) = self.new_aux_var(shape);

        for e in exprs {
            let ce = self.canonicalize_expr(e, false).as_linear().clone();
            // t <= x_i, i.e., x_i - t >= 0
            self.constraints
                .push(ConeConstraint::NonNeg { a: ce.add(&t.neg()) });
        }

        CanonExpr::Linear(t)
    }

    fn canonicalize_quad_form(&mut self, x: &Expr, p: &Expr, for_objective: bool) -> CanonExpr {
        // x' P x: If for_objective and P is constant PSD, use native QP
        let cx = self.canonicalize_expr(x, false).as_linear().clone();

        if for_objective {
            if let Some(p_arr) = p.constant_value() {
                if let Array::Dense(p_mat) = p_arr {
                    // Build quadratic form for native QP
                    // x' P x where x = sum_i A_i v_i + b
                    // For now, simplified: assume x is a single variable
                    let vars = cx.variables();
                    if vars.len() == 1 && cx.constant.iter().all(|&v| v == 0.0) {
                        let var_id = vars[0];
                        let p_csc = dense_to_csc(p_mat);
                        return CanonExpr::Quadratic(QuadExpr::quadratic(var_id, p_csc));
                    }
                }
            }
        }

        // Fall back to SOC reformulation
        // x' P x where P = L L' (Cholesky)
        // = ||L' x||_2^2
        // Introduce t, SOC constraint
        self.canonicalize_sum_squares_lin(&cx, for_objective)
    }

    fn canonicalize_sum_squares(&mut self, x: &Expr, for_objective: bool) -> CanonExpr {
        // ||x||_2^2 = x' x
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        self.canonicalize_sum_squares_lin(&cx, for_objective)
    }

    fn canonicalize_sum_squares_lin(&mut self, x: &LinExpr, for_objective: bool) -> CanonExpr {
        if for_objective {
            // For objective, use native QP: ||x||^2 = x' I x
            // The (1/2) factor for Clarabel is handled in stuffing.rs
            let vars = x.variables();
            if vars.len() == 1 && x.constant.iter().all(|&v| v == 0.0) {
                let var_id = vars[0];
                let size = x.size();
                let identity = CscMatrix::identity(size);
                return CanonExpr::Quadratic(QuadExpr::quadratic(var_id, identity));
            }
        }

        // SOC reformulation: ||x||^2 <= t iff SOC(sqrt(t), x)
        // Actually: introduce t, s, with t = s + 1, and SOC(s, x)
        // Simpler: introduce t >= 0, with ||x||_2^2 <= t via rotated SOC
        // Or: ||x||^2 = quad_over_lin(x, 1)
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Rotated SOC: ||x||^2 <= 2 * t * 1 = 2t
        // Standard form: || [2t - 1; 2x] ||_2 <= 2t + 1
        // Simplified: use SOC with proper reformulation
        self.constraints.push(ConeConstraint::SOC {
            t: t.clone(),
            x: x.clone(),
        });

        CanonExpr::Linear(t)
    }

    fn canonicalize_quad_over_lin(&mut self, x: &Expr, y: &Expr) -> CanonExpr {
        // ||x||_2^2 / y: Introduce t, rotated SOC constraint
        let cx = self.canonicalize_expr(x, false).as_linear().clone();
        let _cy = self.canonicalize_expr(y, false).as_linear().clone();
        let (_, t) = self.new_nonneg_aux_var(Shape::scalar());

        // Rotated SOC: ||x||^2 <= t * y
        // This requires proper rotated SOC support
        // Simplified: add as SOC
        self.constraints.push(ConeConstraint::SOC { t: t.clone(), x: cx });

        CanonExpr::Linear(t)
    }

    // ========================================================================
    // Utility functions
    // ========================================================================

    fn canonicalize_sum_lin(&mut self, x: &LinExpr) -> CanonExpr {
        let size = x.size();
        let ones = DMatrix::from_element(1, size, 1.0);

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &x.coeffs {
            let new_coeff = dense_sparse_matmul(&ones, coeff);
            new_coeffs.insert(*var_id, new_coeff);
        }

        let flat_const = x.constant.clone().reshape_generic(
            nalgebra::Dyn(size),
            nalgebra::Dyn(1),
        );
        let result = &ones * &flat_const;
        let new_const = DMatrix::from_element(1, 1, result[(0, 0)]);

        CanonExpr::Linear(LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::scalar(),
        })
    }

    fn expand_scalar(&self, scalar: &LinExpr, size: usize) -> LinExpr {
        // Expand a scalar to a vector by repeating
        let ones = DMatrix::from_element(size, 1, 1.0);
        let new_const = &ones * &scalar.constant;

        let mut new_coeffs = std::collections::HashMap::new();
        for (var_id, coeff) in &scalar.coeffs {
            // Repeat the scalar coefficient
            let expanded = repeat_rows_csc(coeff, size);
            new_coeffs.insert(*var_id, expanded);
        }

        LinExpr {
            coeffs: new_coeffs,
            constant: new_const,
            shape: Shape::vector(size),
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn dense_sparse_matmul(dense: &DMatrix<f64>, sparse: &CscMatrix<f64>) -> CscMatrix<f64> {
    // Dense @ Sparse multiplication
    // Note: nalgebra_sparse doesn't support dense @ sparse directly.
    // A more efficient implementation would iterate through sparse columns.
    // For medium-scale problems (100-10k variables), this is acceptable.
    let result = dense * csc_to_dense(sparse);
    dense_to_csc(&result)
}

fn stack_vertical(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(a.nrows() + b.nrows(), a.ncols().max(b.ncols()));
    result
        .view_mut((0, 0), (a.nrows(), a.ncols()))
        .copy_from(a);
    result
        .view_mut((a.nrows(), 0), (b.nrows(), b.ncols()))
        .copy_from(b);
    result
}

fn stack_csc_vertical(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    csc_vstack(a, b)
}

fn repeat_rows_csc(m: &CscMatrix<f64>, times: usize) -> CscMatrix<f64> {
    csc_repeat_rows(m, times)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::variable;

    #[test]
    fn test_canonicalize_variable() {
        let x = variable(5);
        let result = canonicalize(&x, false);
        assert!(result.constraints.is_empty());
        assert!(matches!(result.expr, CanonExpr::Linear(_)));
    }

    #[test]
    fn test_canonicalize_norm2() {
        let x = variable(5);
        let n = Expr::Norm2(Arc::new(x));
        let result = canonicalize(&n, false);
        // Should have 1 SOC constraint + 1 NonNeg (t >= 0), and 1 aux variable
        assert_eq!(result.constraints.len(), 2);
        assert_eq!(result.aux_vars.len(), 1);
    }

    #[test]
    fn test_canonicalize_sum_squares_objective() {
        let x = variable(5);
        let s = Expr::SumSquares(Arc::new(x));
        let result = canonicalize(&s, true);
        // For objective, should produce quadratic or SOC
        assert!(
            matches!(result.expr, CanonExpr::Quadratic(_))
                || !result.constraints.is_empty()
        );
    }
}
