//! Solver interface for cvxrust.
//!
//! This module provides:
//! - Matrix stuffing to convert canonicalized problems to solver format
//! - Clarabel solver integration

pub mod clarabel;
pub mod stuffing;

pub use self::clarabel::{solve, Settings, Solution, SolveStatus};
pub use stuffing::{stuff_problem, ConeDims, StuffedProblem, VariableMap};
