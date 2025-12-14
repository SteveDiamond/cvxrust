//! Basic Linear Programming Example
//!
//! This example demonstrates solving a simple linear program:
//!
//! minimize    c'x
//! subject to  Ax <= b, x >= 0

use cvxrust::prelude::*;

fn main() {
    println!("=== Basic Linear Program ===\n");

    // Problem: Maximize profit = 3*x1 + 2*x2 + 5*x3
    // subject to:
    //   2*x1 + x2 + 3*x3 <= 100  (resource constraint 1)
    //   x1 + 2*x2 + x3 <= 80     (resource constraint 2)
    //   x >= 0

    println!("Problem: Maximize 3*x1 + 2*x2 + 5*x3");
    println!("Subject to:");
    println!("  2*x1 + x2 + 3*x3 <= 100");
    println!("  x1 + 2*x2 + x3 <= 80");
    println!("  x >= 0\n");

    // Create variables
    let x = variable(3);

    // Coefficients
    let c = constant_vec(vec![3.0, 2.0, 5.0]);
    let a = constant_matrix(vec![
        2.0, 1.0, 3.0,
        1.0, 2.0, 1.0,
    ], 2, 3);
    let b = constant_vec(vec![100.0, 80.0]);

    // Objective: maximize c'x (minimize -c'x)
    let objective = -dot(&c, &x);

    // Solve
    println!("Solving...");
    let solution = Problem::minimize(objective)
        .subject_to([
            matmul(&a, &x).leq(&b),
            x.clone().geq(&zeros(3)),
        ])
        .solve()
        .expect("Failed to solve");

    // Display results
    println!("\nResults:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal profit: {:.4}", -solution.value.unwrap());

    let x_val = solution.get_value(x.variable_id().unwrap()).unwrap();
    let x_mat = match x_val {
        Array::Dense(m) => m,
        _ => panic!("Expected dense array"),
    };
    println!("  x1 = {:.4}", x_mat[(0, 0)]);
    println!("  x2 = {:.4}", x_mat[(1, 0)]);
    println!("  x3 = {:.4}", x_mat[(2, 0)]);
}
