//! Quadratic Programming Example
//!
//! This example demonstrates solving a quadratic program:
//!
//! minimize    (1/2) x'Px + q'x
//! subject to  Ax = b, x >= 0

use cvxrust::prelude::*;

fn main() {
    println!("=== Quadratic Programming ===\n");

    // Problem: Find point closest to [3, 2] satisfying x1 + x2 = 4, x >= 0
    // minimize (x1 - 3)^2 + (x2 - 2)^2

    println!("Problem: Find point closest to [3, 2]");
    println!("Subject to: x1 + x2 = 4, x >= 0\n");

    let x = variable(2);

    // Objective: ||x - [3, 2]||^2
    let target = constant_vec(vec![3.0, 2.0]);
    let objective = sum_squares(&(&x - &target));

    // Solve
    println!("Solving...");
    let solution = Problem::minimize(objective)
        .subject_to([
            sum(&x).equals(&constant(4.0)),
            x.clone().geq(&zeros(2)),
        ])
        .solve()
        .expect("Failed to solve");

    println!("\nResults:");
    println!("  Status: {:?}", solution.status);
    println!("  Optimal value: {:.6}", solution.value.unwrap());

    let x_val = solution.get_value(x.variable_id().unwrap()).unwrap();
    let x_mat = match x_val {
        Array::Dense(m) => m,
        _ => panic!("Expected dense array"),
    };
    println!("  x1 = {:.6}", x_mat[(0, 0)]);
    println!("  x2 = {:.6}", x_mat[(1, 0)]);
    println!("  Sum: {:.6}", x_mat[(0, 0)] + x_mat[(1, 0)]);
    println!("  Distance: {:.6}", solution.value.unwrap().sqrt());
}
