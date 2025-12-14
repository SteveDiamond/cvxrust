# cvxrust

A Disciplined Convex Programming (DCP) library for Rust, inspired by [CVXPY](https://www.cvxpy.org/).

cvxrust provides a domain-specific language for specifying convex optimization problems with automatic convexity verification and solving via the [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) solver.

## Features

- **DCP Verification**: Automatically verifies problem convexity using disciplined convex programming rules
- **Rich Atom Library**: Norms, quadratic forms, element-wise operations, and more
- **Native QP Support**: Quadratic programs are solved directly (not reformulated as SOCPs)
- **Flexible API**: Builder patterns for variables, constraints, and problems
- **Sparse Matrix Support**: Efficient handling of large-scale problems

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
cvxrust = { git = "https://github.com/your-username/cvxrust" }
```

## Quick Start

```rust
use cvxrust::prelude::*;

// Minimize ||x||_2 subject to sum(x) = 1, x >= 0
let x = variable(5);
let objective = norm2(&x);
let constraints = [
    sum(&x).equals(&constant(1.0)),
    x.clone().geq(&zeros(5)),
];

let solution = Problem::minimize(objective)
    .subject_to(constraints)
    .solve()
    .unwrap();

println!("Optimal value: {}", solution.value);
println!("x = {:?}", x.value(&solution));
```

## Supported Problem Classes

| Problem Type | Objective | Constraints |
|-------------|-----------|-------------|
| **LP** | Linear | Linear equality/inequality |
| **QP** | Quadratic (convex) | Linear equality/inequality |
| **SOCP** | Linear | Second-order cone |
| **Mixed** | Any convex | Combination of above |

## Expression Atoms

### Affine Operations
- Arithmetic: `+`, `-`, `*` (scalar), `/` (scalar)
- Aggregation: `sum`, `sum_axis`
- Structural: `reshape`, `flatten`, `vstack`, `hstack`, `transpose`
- Linear algebra: `matmul`, `dot`, `trace`
- Indexing: `index`, `slice`

### Convex Atoms
- Norms: `norm1`, `norm2`, `norm_inf`, `norm`
- Element-wise: `abs`, `pos`, `neg_part`
- Aggregation: `maximum`, `max2`
- Quadratic: `quad_form` (PSD), `sum_squares`, `quad_over_lin`

### Concave Atoms
- Aggregation: `minimum`, `min2`
- Quadratic: `quad_form` (NSD)

## Variable Construction

```rust
// Simple vector variable
let x = variable(5);

// Matrix variable
let X = variable((3, 4));

// Named variable with constraints
let x = VariableBuilder::vector(5)
    .name("x")
    .nonneg()  // x >= 0
    .build();
```

## Constraint Types

```rust
// Equality constraint
expr.equals(&rhs)

// Inequality constraints
expr.leq(&rhs)  // expr <= rhs
expr.geq(&rhs)  // expr >= rhs

// Second-order cone constraint
Constraint::soc(t, x)  // ||x||_2 <= t
```

## Examples

### Least Squares
```rust
let x = variable(n);
let objective = sum_squares(&(A * x.clone() - b));
let solution = Problem::minimize(objective).solve()?;
```

### L1 Regularized Regression (Lasso)
```rust
let x = variable(n);
let loss = sum_squares(&(A * x.clone() - b));
let reg = norm1(&x);
let objective = loss + lambda * reg;
let solution = Problem::minimize(objective).solve()?;
```

### Portfolio Optimization
```rust
let w = VariableBuilder::vector(n).nonneg().build();
let ret = dot(&mu, &w);
let risk = quad_form(&w, &Sigma);
let solution = Problem::maximize(ret - gamma * risk)
    .subject_to([sum(&w).equals(&constant(1.0))])
    .solve()?;
```

## DCP Rules

cvxrust enforces disciplined convex programming rules:

- **Minimization** requires a **convex** objective
- **Maximization** requires a **concave** objective
- **Equality constraints** require **affine** expressions
- **Inequality constraints** (`>=`) require **concave** expressions on the left

Problems that violate these rules will return a `DcpError`.

## Architecture

```
Expression → DCP Verification → Canonicalization → Matrix Stuffing → Clarabel → Solution
```

1. Build expression trees using atoms and operator overloads
2. Verify convexity via curvature and sign propagation
3. Transform to canonical form (LinExpr/QuadExpr + cone constraints)
4. Stuff into sparse matrices P, q, A, b
5. Solve with Clarabel and map solution back to variables

## License

Apache 2.0
