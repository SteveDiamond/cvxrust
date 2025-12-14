//! Clarabel solver integration.
//!
//! This module provides the interface to the Clarabel conic solver.

use std::collections::HashMap;

use clarabel::algebra::CscMatrix as ClarabelCsc;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT,
};

use super::stuffing::{ConeDims, StuffedProblem, VariableMap};
use crate::expr::{Array, ExprId};

/// Solution status from the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Optimal solution found.
    Optimal,
    /// Problem is infeasible.
    Infeasible,
    /// Problem is unbounded.
    Unbounded,
    /// Maximum iterations reached.
    MaxIterations,
    /// Numerical difficulties.
    NumericalError,
    /// Unknown status.
    Unknown,
}

impl From<SolverStatus> for SolveStatus {
    fn from(status: SolverStatus) -> Self {
        match status {
            SolverStatus::Solved => SolveStatus::Optimal,
            SolverStatus::PrimalInfeasible => SolveStatus::Infeasible,
            SolverStatus::DualInfeasible => SolveStatus::Unbounded,
            SolverStatus::MaxIterations => SolveStatus::MaxIterations,
            SolverStatus::MaxTime => SolveStatus::MaxIterations,
            _ => SolveStatus::Unknown,
        }
    }
}

/// Solver settings.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Print solver output.
    pub verbose: bool,
    /// Maximum iterations.
    pub max_iter: u32,
    /// Time limit in seconds.
    pub time_limit: f64,
    /// Absolute tolerance.
    pub tol_gap_abs: f64,
    /// Relative tolerance.
    pub tol_gap_rel: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            verbose: false,
            max_iter: 100,
            time_limit: f64::INFINITY,
            tol_gap_abs: 1e-8,
            tol_gap_rel: 1e-8,
        }
    }
}

/// Solution from the solver.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solution status.
    pub status: SolveStatus,
    /// Optimal value (if solved).
    pub value: Option<f64>,
    /// Primal variable values (if solved).
    pub primal: Option<HashMap<ExprId, Array>>,
    /// Dual variable values (if solved).
    pub dual: Option<Vec<f64>>,
    /// Solve time in seconds.
    pub solve_time: f64,
    /// Number of iterations.
    pub iterations: u32,
}

impl Solution {
    /// Get the value of a variable.
    pub fn get_value(&self, var_id: ExprId) -> Option<&Array> {
        self.primal.as_ref().and_then(|p| p.get(&var_id))
    }
}

/// Solve the stuffed problem using Clarabel.
pub fn solve(problem: &StuffedProblem, settings: &Settings) -> Solution {
    // Convert to Clarabel format
    let p = to_clarabel_csc(&problem.p);
    let a = to_clarabel_csc(&problem.a);
    let cones = to_clarabel_cones(&problem.cone_dims);

    // Build Clarabel settings
    let clarabel_settings = DefaultSettingsBuilder::default()
        .verbose(settings.verbose)
        .max_iter(settings.max_iter)
        .time_limit(settings.time_limit)
        .tol_gap_abs(settings.tol_gap_abs)
        .tol_gap_rel(settings.tol_gap_rel)
        .build()
        .unwrap();

    // Create and run solver
    let mut solver = DefaultSolver::new(&p, &problem.q, &a, &problem.b, &cones, clarabel_settings);
    solver.solve();

    // Extract solution
    let status: SolveStatus = solver.solution.status.into();
    let solve_time = solver.solution.solve_time;
    let iterations = solver.info.iterations;

    if status == SolveStatus::Optimal {
        let primal = unpack_primal(&solver.solution.x, &problem.var_map);
        let value = compute_objective(&solver.solution.x, &problem.p, &problem.q);

        Solution {
            status,
            value: Some(value),
            primal: Some(primal),
            dual: Some(solver.solution.z.clone()),
            solve_time,
            iterations,
        }
    } else {
        Solution {
            status,
            value: None,
            primal: None,
            dual: None,
            solve_time,
            iterations,
        }
    }
}

/// Convert nalgebra CSC to Clarabel CSC.
fn to_clarabel_csc(m: &nalgebra_sparse::CscMatrix<f64>) -> ClarabelCsc<f64> {
    ClarabelCsc::new(
        m.nrows(),
        m.ncols(),
        m.col_offsets().to_vec(),
        m.row_indices().to_vec(),
        m.values().to_vec(),
    )
}

/// Convert cone dimensions to Clarabel cones.
fn to_clarabel_cones(dims: &ConeDims) -> Vec<SupportedConeT<f64>> {
    let mut cones = Vec::new();

    if dims.zero > 0 {
        cones.push(SupportedConeT::ZeroConeT(dims.zero));
    }

    if dims.nonneg > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(dims.nonneg));
    }

    for &soc_dim in &dims.soc {
        cones.push(SupportedConeT::SecondOrderConeT(soc_dim));
    }

    cones
}

/// Unpack primal solution into variable values.
fn unpack_primal(x: &[f64], var_map: &VariableMap) -> HashMap<ExprId, Array> {
    let mut result = HashMap::new();

    for (&var_id, &(start, size)) in &var_map.id_to_col {
        let values: Vec<f64> = x[start..start + size].to_vec();
        let arr = if size == 1 {
            Array::Scalar(values[0])
        } else {
            Array::from_vec(values)
        };
        result.insert(var_id, arr);
    }

    result
}

/// Compute objective value: (1/2) x' P x + q' x.
fn compute_objective(x: &[f64], p: &nalgebra_sparse::CscMatrix<f64>, q: &[f64]) -> f64 {
    // q' x
    let linear: f64 = q.iter().zip(x.iter()).map(|(qi, xi)| qi * xi).sum();

    // (1/2) x' P x
    let mut quadratic = 0.0;
    for (row, col, val) in p.triplet_iter() {
        if row == col {
            quadratic += 0.5 * *val * x[row] * x[col];
        } else {
            // Off-diagonal (stored as upper triangle, so count once)
            quadratic += *val * x[row] * x[col];
        }
    }

    linear + quadratic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert!(!settings.verbose);
        assert_eq!(settings.max_iter, 100);
    }

    #[test]
    fn test_to_clarabel_cones() {
        let dims = ConeDims {
            zero: 2,
            nonneg: 3,
            soc: vec![4],
        };
        let cones = to_clarabel_cones(&dims);
        assert_eq!(cones.len(), 3);
    }
}
