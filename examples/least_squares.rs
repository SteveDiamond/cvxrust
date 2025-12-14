//! Least Squares Regression Example
//!
//! This example demonstrates linear regression:
//!
//! minimize    ||Ax - b||_2^2

use cvxrust::prelude::*;

fn main() {
    println!("=== Least Squares Regression ===\n");

    // Fit a line y = w0 + w1*x to data
    let a = constant_matrix(vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
        1.0, 5.0,
    ], 5, 2);
    let b = constant_vec(vec![3.1, 5.2, 6.8, 9.1, 10.9]);

    println!("Fitting y = w0 + w1*x to 5 data points\n");

    // Unconstrained least squares
    let w = variable(2);
    let residual = matmul(&a, &w) - &b;

    println!("Solving unconstrained least squares...");
    let solution = Problem::minimize(sum_squares(&residual))
        .solve()
        .expect("Failed to solve");

    println!("\nUnconstrained Results:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal value: {:.6}", solution.value.unwrap());

    let w_val = solution.get_value(w.variable_id().unwrap()).unwrap();
    let w_mat = match w_val {
        Array::Dense(m) => m,
        _ => panic!("Expected dense array"),
    };
    println!("  w0 (intercept) = {:.6}", w_mat[(0, 0)]);
    println!("  w1 (slope) = {:.6}", w_mat[(1, 0)]);

    // Constrained least squares (w >= 0)
    println!("\n--- Constrained Least Squares (w >= 0) ---\n");

    let w2 = variable(2);
    let residual2 = matmul(&a, &w2) - &b;

    let solution2 = Problem::minimize(sum_squares(&residual2))
        .constraint(w2.clone().geq(&zeros(2)))
        .solve()
        .expect("Failed to solve");

    println!("Constrained Results:");
    println!("  Status: {:?}", solution2.status);
    println!("  Optimal value: {:.6}", solution2.value.unwrap());

    let w2_val = solution2.get_value(w2.variable_id().unwrap()).unwrap();
    let w2_mat = match w2_val {
        Array::Dense(m) => m,
        _ => panic!("Expected dense array"),
    };
    println!("  w0 (intercept) = {:.6}", w2_mat[(0, 0)]);
    println!("  w1 (slope) = {:.6}", w2_mat[(1, 0)]);
}
