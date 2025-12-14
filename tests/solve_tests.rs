//! Comprehensive solve tests for all atoms.
//!
//! Pattern inspired by CVXPY's test_constant_atoms.py:
//! Define test cases as data, then run them programmatically.

use cvxrust::prelude::*;

/// Tolerance for comparing floating point results
const TOL: f64 = 1e-4;

/// A test case definition
struct TestCase {
    name: &'static str,
    /// Function that builds the problem and returns (problem, expected_value)
    build: fn() -> (Problem, f64),
}

/// All minimize test cases
fn minimize_test_cases() -> Vec<TestCase> {
    vec![
        // ========== Linear Programs ==========
        TestCase {
            name: "sum_nonneg_constraint",
            build: || {
                // minimize sum(x) s.t. x >= 1, x in R^5
                // optimal: x = [1,1,1,1,1], value = 5
                let x = variable(5);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.geq(&constant(1.0))])
                    .build();
                (prob, 5.0)
            },
        },
        TestCase {
            name: "sum_equality_constraint",
            build: || {
                // minimize sum(x) s.t. x == 2, x in R^3
                // optimal: x = [2,2,2], value = 6
                let x = variable(3);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.equals(&constant(2.0))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "sum_upper_bound",
            build: || {
                // minimize -sum(x) s.t. x <= 3, x in R^4
                // optimal: x = [3,3,3,3], value = -12
                let x = variable(4);
                let neg_sum = &constant(-1.0) * &sum(&x);
                let prob = Problem::minimize(neg_sum)
                    .subject_to([x.leq(&constant(3.0))])
                    .build();
                (prob, -12.0)
            },
        },
        TestCase {
            name: "weighted_sum",
            build: || {
                // minimize 2*x + 3*y s.t. x >= 1, y >= 2
                // optimal: x=1, y=2, value = 2 + 6 = 8
                let x = variable(1);
                let y = variable(1);
                let obj = &(&constant(2.0) * &x) + &(&constant(3.0) * &y);
                let prob = Problem::minimize(obj)
                    .subject_to([x.geq(&constant(1.0)), y.geq(&constant(2.0))])
                    .build();
                (prob, 8.0)
            },
        },

        // ========== Norms (SOCP) ==========
        TestCase {
            name: "norm2_equality",
            build: || {
                // minimize ||x||_2 s.t. sum(x) = 5, x in R^5
                // optimal: x = [1,1,1,1,1], value = sqrt(5)
                let x = variable(5);
                let prob = Problem::minimize(norm2(&x))
                    .subject_to([sum(&x).equals(&constant(5.0))])
                    .build();
                (prob, 5.0_f64.sqrt())
            },
        },
        TestCase {
            name: "norm2_zero",
            build: || {
                // minimize ||x||_2 + 1 s.t. x == 0, x in R^3
                // optimal: x = [0,0,0], value = 1
                let x = variable(3);
                let obj = &norm2(&x) + &constant(1.0);
                let prob = Problem::minimize(obj)
                    .subject_to([x.equals(&constant(0.0))])
                    .build();
                (prob, 1.0)
            },
        },
        TestCase {
            name: "norm1_equality",
            build: || {
                // minimize ||x||_1 s.t. sum(x) = 3, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(norm1(&x))
                    .subject_to([sum(&x).equals(&constant(3.0))])
                    .build();
                (prob, 3.0)
            },
        },
        TestCase {
            name: "norm_inf_equality",
            build: || {
                // minimize ||x||_inf s.t. sum(x) = 4, x in R^4
                // optimal: x = [1,1,1,1], value = 1
                let x = variable(4);
                let prob = Problem::minimize(norm_inf(&x))
                    .subject_to([sum(&x).equals(&constant(4.0))])
                    .build();
                (prob, 1.0)
            },
        },

        // ========== Quadratic (QP) ==========
        TestCase {
            name: "sum_squares_equality",
            build: || {
                // minimize ||x||_2^2 s.t. sum(x) = 2, x in R^2
                // optimal: x = [1,1], value = 2
                let x = variable(2);
                let prob = Problem::minimize(sum_squares(&x))
                    .subject_to([sum(&x).equals(&constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },
        TestCase {
            name: "sum_squares_nonneg",
            build: || {
                // minimize ||x||_2^2 s.t. x >= 1, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(sum_squares(&x))
                    .subject_to([x.geq(&constant(1.0))])
                    .build();
                (prob, 3.0)
            },
        },

        // ========== Element-wise convex ==========
        TestCase {
            name: "abs_equality",
            build: || {
                // minimize sum(|x|) s.t. x == [-1, 2, -3]
                // value = 1 + 2 + 3 = 6
                let x = variable(3);
                let prob = Problem::minimize(sum(&abs(&x)))
                    .subject_to([x.equals(&constant_vec(vec![-1.0, 2.0, -3.0]))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "pos_nonneg",
            build: || {
                // minimize sum(pos(x)) s.t. x >= -2, sum(x) = 1, x in R^2
                // optimal: x = [-2, 3] gives pos(x) = [0, 3], sum = 3... no
                // Actually with sum(x) = 1, to minimize sum(pos(x)):
                // optimal is x = [0.5, 0.5], pos = [0.5, 0.5], value = 1
                let x = variable(2);
                let prob = Problem::minimize(sum(&pos(&x)))
                    .subject_to([
                        x.geq(&constant(-2.0)),
                        sum(&x).equals(&constant(1.0)),
                    ])
                    .build();
                (prob, 1.0)
            },
        },

        // ========== Maximum/Minimum ==========
        TestCase {
            name: "maximum_bound",
            build: || {
                // minimize max(x, y) s.t. x >= 1, y >= 2
                // optimal: x=1, y=2, value = 2
                let x = variable(1);
                let y = variable(1);
                let prob = Problem::minimize(max2(&x, &y))
                    .subject_to([x.geq(&constant(1.0)), y.geq(&constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },

        // ========== Multiple constraints ==========
        TestCase {
            name: "box_constraints",
            build: || {
                // minimize sum(x) s.t. x >= 1, x <= 2, x in R^3
                // optimal: x = [1,1,1], value = 3
                let x = variable(3);
                let prob = Problem::minimize(sum(&x))
                    .subject_to([x.geq(&constant(1.0)), x.leq(&constant(2.0))])
                    .build();
                (prob, 3.0)
            },
        },
        TestCase {
            name: "mixed_constraints",
            build: || {
                // minimize ||x||_2 s.t. x >= 0, sum(x) = 3, x in R^3
                // optimal: x = [1,1,1], value = sqrt(3)
                let x = variable(3);
                let prob = Problem::minimize(norm2(&x))
                    .subject_to([
                        x.geq(&constant(0.0)),
                        sum(&x).equals(&constant(3.0)),
                    ])
                    .build();
                (prob, 3.0_f64.sqrt())
            },
        },
    ]
}

/// All maximize test cases
fn maximize_test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "maximize_sum_upper_bound",
            build: || {
                // maximize sum(x) s.t. x <= 2, x in R^3
                // optimal: x = [2,2,2], value = 6
                let x = variable(3);
                let prob = Problem::maximize(sum(&x))
                    .subject_to([x.leq(&constant(2.0))])
                    .build();
                (prob, 6.0)
            },
        },
        TestCase {
            name: "maximize_minimum",
            build: || {
                // maximize min(x, y) s.t. x <= 3, y <= 2
                // optimal: x=2, y=2, value = 2
                let x = variable(1);
                let y = variable(1);
                let prob = Problem::maximize(min2(&x, &y))
                    .subject_to([x.leq(&constant(3.0)), y.leq(&constant(2.0))])
                    .build();
                (prob, 2.0)
            },
        },
        TestCase {
            name: "maximize_neg_norm",
            build: || {
                // maximize -||x||_2 s.t. sum(x) = 0, x in R^2
                // optimal: x = [0,0], value = 0
                let x = variable(2);
                let neg_norm = &constant(-1.0) * &norm2(&x);
                let prob = Problem::maximize(neg_norm)
                    .subject_to([sum(&x).equals(&constant(0.0))])
                    .build();
                (prob, 0.0)
            },
        },
    ]
}

/// Test cases expected to be infeasible
fn infeasible_test_cases() -> Vec<(&'static str, Problem)> {
    vec![
        (
            "infeasible_bounds",
            {
                // x >= 1 and x <= 0 is infeasible
                let x = variable(3);
                Problem::minimize(sum(&x))
                    .subject_to([x.geq(&constant(1.0)), x.leq(&constant(0.0))])
                    .build()
            },
        ),
        (
            "infeasible_equality",
            {
                // x == 1 and x == 2 is infeasible
                let x = variable(1);
                Problem::minimize(sum(&x))
                    .subject_to([x.equals(&constant(1.0)), x.equals(&constant(2.0))])
                    .build()
            },
        ),
    ]
}

/// Test cases expected to be unbounded
fn unbounded_test_cases() -> Vec<(&'static str, Problem)> {
    vec![
        (
            "unbounded_below",
            {
                // minimize sum(x) with only upper bound
                let x = variable(3);
                Problem::minimize(sum(&x))
                    .subject_to([x.leq(&constant(1.0))])
                    .build()
            },
        ),
        (
            "unbounded_above",
            {
                // maximize sum(x) with only lower bound
                let x = variable(3);
                Problem::maximize(sum(&x))
                    .subject_to([x.geq(&constant(1.0))])
                    .build()
            },
        ),
    ]
}

// ============================================================================
// Test runner
// ============================================================================

#[test]
fn test_minimize_atoms() {
    for case in minimize_test_cases() {
        let (prob, expected) = (case.build)();

        assert!(prob.is_dcp(), "Problem '{}' should be DCP", case.name);

        let result = prob.solve();
        assert!(result.is_ok(), "Problem '{}' should solve: {:?}", case.name, result.err());

        let solution = result.unwrap();
        assert_eq!(
            solution.status, SolveStatus::Optimal,
            "Problem '{}' should be optimal, got {:?}", case.name, solution.status
        );

        let value = solution.value.expect("should have value");
        let rel_err = (value - expected).abs() / (1.0 + expected.abs());
        assert!(
            rel_err < TOL,
            "Problem '{}': expected {}, got {} (rel_err={})",
            case.name, expected, value, rel_err
        );
    }
}

#[test]
fn test_maximize_atoms() {
    for case in maximize_test_cases() {
        let (prob, expected) = (case.build)();

        assert!(prob.is_dcp(), "Problem '{}' should be DCP", case.name);

        let result = prob.solve();
        assert!(result.is_ok(), "Problem '{}' should solve: {:?}", case.name, result.err());

        let solution = result.unwrap();
        assert_eq!(
            solution.status, SolveStatus::Optimal,
            "Problem '{}' should be optimal, got {:?}", case.name, solution.status
        );

        let value = solution.value.expect("should have value");
        let rel_err = (value - expected).abs() / (1.0 + expected.abs());
        assert!(
            rel_err < TOL,
            "Problem '{}': expected {}, got {} (rel_err={})",
            case.name, expected, value, rel_err
        );
    }
}

#[test]
fn test_infeasible() {
    for (name, prob) in infeasible_test_cases() {
        let result = prob.solve();
        match result {
            Err(CvxError::SolverError(msg)) if msg.contains("infeasible") => {
                // Expected
            }
            Ok(solution) if solution.status == SolveStatus::Infeasible => {
                // Also acceptable
            }
            other => {
                panic!("Problem '{}' should be infeasible, got {:?}", name, other);
            }
        }
    }
}

#[test]
fn test_unbounded() {
    for (name, prob) in unbounded_test_cases() {
        let result = prob.solve();
        match result {
            Err(CvxError::SolverError(msg)) if msg.contains("unbounded") => {
                // Expected
            }
            Ok(solution) if solution.status == SolveStatus::Unbounded => {
                // Also acceptable
            }
            other => {
                panic!("Problem '{}' should be unbounded, got {:?}", name, other);
            }
        }
    }
}

// ============================================================================
// Primal value verification tests
// ============================================================================

#[test]
fn test_primal_values() {
    // Test that we can recover the optimal x values, not just objective
    let x = variable(3);
    let result = Problem::minimize(sum(&x))
        .subject_to([x.geq(&constant(2.0))])
        .solve()
        .expect("should solve");

    assert_eq!(result.status, SolveStatus::Optimal);

    let x_val = result.get_value(x.variable_id().unwrap());
    assert!(x_val.is_some(), "should have primal value for x");

    let arr = x_val.unwrap();
    // Each component should be approximately 2.0
    match arr {
        Array::Dense(m) => {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    assert!(
                        (m[(i, j)] - 2.0).abs() < TOL,
                        "x[{},{}] should be ~2.0, got {}",
                        i, j, m[(i, j)]
                    );
                }
            }
        }
        _ => panic!("expected dense array"),
    }
}
