# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build              # Build the library
cargo test               # Run all tests (159 tests)
cargo test test_name     # Run a specific test
cargo test -- --nocapture  # Run tests with stdout visible
cargo clippy             # Run linter
cargo doc --open         # Generate and view documentation
```

## Running Examples

```bash
cargo run --example portfolio        # Portfolio optimization with dual variables
cargo run --example least_squares    # Least squares regression
cargo run --example quadratic_program # Basic QP
cargo run --example basic_lp         # Linear programming
cargo run --example lasso            # L1-regularized regression
```

## Architecture

cvxrust is a Disciplined Convex Programming (DCP) library that transforms optimization problems into standard form and solves them via Clarabel.

### Core Data Flow

1. **Expression Building**: Users construct `Expr` enum trees using atoms (`sum`, `norm2`, etc.) and operator overloads
2. **DCP Verification**: `Curvature` (convex/concave/affine) and `Sign` (nonneg/nonpos) are computed via composition rules
3. **Canonicalization**: `Expr` → `LinExpr`/`QuadExpr` + `ConeConstraint` (Zero, NonNeg, SOC, ExpCone, PowerCone)
4. **Matrix Stuffing**: Build sparse matrices P, q, A, b for Clarabel's standard form
5. **Solve**: Clarabel returns solution, mapped back to user variables

### Module Structure

```
src/
├── lib.rs              # Main library entry point and prelude
├── expr/               # Expression types and constructors
│   ├── expression.rs   # Core Expr enum with all expression variants
│   ├── variable.rs     # Variable creation with builder pattern
│   ├── constant.rs     # Constant creation helpers
│   └── shape.rs        # Shape/dimension tracking
├── atoms/              # Building blocks for expressions
│   ├── affine.rs       # Affine operations and operator overloading
│   └── nonlinear.rs    # Convex/concave atoms (norms, exp, log, power, etc.)
├── dcp/                # Disciplined Convex Programming analysis
│   ├── curvature.rs    # Curvature tracking (convex/concave/affine/constant)
│   └── sign.rs         # Sign tracking (nonnegative/nonpositive)
├── constraints/        # Constraint types
│   └── constraint.rs   # Constraint definitions, DCP verification, constraint! macro
├── canon/              # Canonicalization (expression → standard form)
│   ├── canonicalizer.rs # Main canonicalization logic
│   └── lin_expr.rs     # Linear and quadratic expression forms
├── solver/             # Solver interface
│   ├── clarabel.rs     # Clarabel solver wrapper, Solution type
│   └── stuffing.rs     # Matrix stuffing for standard form
├── problem.rs          # Problem definition and builder
├── sparse.rs           # Sparse matrix utilities
└── error.rs            # Error types (CvxError)
```

### Key Types

- `Expr` (expr/expression.rs): Enum with all expression variants (Variable, Constant, Add, Norm2, Exp, Log, etc.)
- `LinExpr` (canon/lin_expr.rs): Affine form `sum_i(A_i * x_i) + b` with sparse coefficient matrices
- `QuadExpr` (canon/lin_expr.rs): Quadratic form `(1/2) x' P x + q' x + r` for native QP support
- `ConeConstraint` (canon/canonicalizer.rs): Zero (equality), NonNeg (inequality), SOC, ExpCone, PowerCone
- `Solution` (solver/clarabel.rs): Contains primal values, dual values, status, and objective value

### Constraint API

Use the `constraint!` macro for natural syntax:
```rust
constraint!(x >= 0.0)        // x >= 0
constraint!(x <= 10.0)       // x <= 10
constraint!((sum(&x)) == 1.0) // sum(x) = 1
```

Or method syntax:
```rust
x.ge(0.0)    // x >= 0
x.le(10.0)   // x <= 10
sum(&x).eq(1.0)
```

### DCP Composition Rules

Curvature propagation in `dcp/curvature.rs`:
- `convex + convex = convex`, `concave + concave = concave`
- `nonneg * convex = convex`, `nonpos * convex = concave`
- `convex(affine) = convex`, `convex(concave) = unknown`

Sign propagation in `dcp/sign.rs`:
- `nonneg + nonneg = nonneg`, `nonpos + nonpos = nonpos`
- `nonneg * nonneg = nonneg`, `nonneg * nonpos = nonpos`

### Implemented Atoms

**Affine** (preserve curvature): `sum`, `sum_axis`, `cumsum`, `reshape`, `flatten`, `vstack`, `hstack`, `transpose`, `diag`, `matmul`, `dot`, `trace`, `index`, `slice`, `+`, `-`, `*scalar`, `/scalar`

**Convex**: `norm1`, `norm2`, `norm_inf`, `norm`, `abs`, `pos`, `neg_part`, `maximum`, `max2`, `quad_form` (PSD), `sum_squares`, `quad_over_lin`, `exp`, `power` (p >= 1 or p < 0)

**Concave**: `minimum`, `min2`, `quad_form` (NSD), `log`, `entropy`, `sqrt`, `power` (0 < p < 1)

### Clarabel Sign Convention

In `solver/stuffing.rs`, constraints use Clarabel's form `Ax + s = b, s ∈ K`:
- **Zero cone** (equality): `Ax = b` where `b = -constant`
- **NonNeg cone**: `expr >= 0` becomes `-Ax <= constant` (negate coefficients)
- **SOC**: `||x||_2 <= t` becomes `[t; x] ∈ K_soc` (also negated)
- **ExpCone**: `(x, y, z) ∈ K_exp` means `y * exp(x/y) <= z`
- **PowerCone**: `(x, y, z) ∈ K_pow(α)` means `x^α * y^(1-α) >= |z|`

## Solver Integration

Only Clarabel is supported. The solver expects:
- P: Quadratic cost matrix (upper triangular, symmetric)
- q: Linear cost vector
- A: Constraint matrix
- b: Constraint vector
- Cones specified as `SupportedConeT` variants

## Common Patterns

### Detecting constant expressions
Use `expr.variables().is_empty()` to check if an expression is constant (has no variables), rather than `expr.constant_value()` which only matches the `Constant` variant directly.

### Accessing dual variables
```rust
let solution = problem.solve()?;
let duals = solution.duals();           // All duals as slice
let dual_i = solution.constraint_dual(i); // Dual for constraint i
```

### Error handling
- `CvxError::NotDcp` - Problem violates DCP rules
- `CvxError::SolverError` - Clarabel failed (infeasible, unbounded, etc.)
- `CvxError::InvalidProblem` - Malformed problem (dimension mismatch, etc.)
