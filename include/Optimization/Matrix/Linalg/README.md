# Linear Algebra (Linalg) Module

This directory contains a suite of high-performance, strictly zero-allocation linear algebra solvers and decomposition algorithms. Like the rest of the Matrix module, it is deeply optimized for real-time NMPC execution using explicit hardware SIMD intrinsics.

## Design Philosophy & Performance

The algorithms in this module are tailored to solve the dense linear systems that arise iteratively in Sequential Quadratic Programming (SQP) and Non-Linear Programming (NLP).
- **In-Place Execution**: All solvers (Cholesky, LDLT, LU, QR) are designed to perform forward and backward substitutions *in-place*. They aggressively reuse input and output memory buffers, completely eliminating dynamic memory allocations.
- **Hardware-Native Vectorization**: The heavy $O(N^3)$ loops within the matrix decompositions are explicitly unrolled and mapped to **ARM NEON** and **AVX2** FMA (Fused Multiply-Add) instructions. This bypasses the unpredictability of compiler auto-vectorizers and guarantees deterministic, peak hardware throughput.
- **Numerical Robustness**: Explicit stability checks (such as NaN propagation guards and tiny-pivot thresholding in `LDLT`) are implemented to prevent matrix explosion and floating-point exceptions during aggressive real-time autonomous driving maneuvers.

## Core Components

### 1. `Core.hpp`
Provides foundational matrix-matrix multiplication routines.
- Implements highly vectorized `multiply` ($C = A \times B$).
- Implements `multiply_AT_B` ($C = A^T \times B$), which calculates the product of a transposed matrix directly. This explicitly avoids the $O(N^2)$ memory copying penalty of performing an actual matrix transpose in memory.

### 2. `Cholesky.hpp` ($A = LL^T$)
- Extremely fast Cholesky decomposition for symmetric positive-definite matrices (e.g., Hessian approximations).
- The `cholesky_solver` performs forward and backward substitutions utilizing contiguous memory access patterns (SAXPY and Dot products) directly mapped to SIMD registers.

### 3. `LDLT.hpp` ($A = LDL^T$)
- Robust LDLT decomposition capable of handling symmetric indefinite systems (which frequently appear in KKT matrices).
- Includes strict numerical pivot checking (`1e-9` thresholding and `std::isnan` guards) to safely catch singularities before they corrupt the controller's steering or acceleration commands.

### 4. `LU.hpp` ($PA = LU$)
- General-purpose LU decomposition featuring **Partial Pivoting** for numerical stability.
- The `lu_solve` routine applies the row-permutation array seamlessly during the zero-allocation forward substitution phase.

### 5. `QR.hpp`
- Implements a highly stable **Modified Gram-Schmidt (MGS)** QR decomposition.
- Features complex SIMD horizontal-add (`hadd`) logic to drastically optimize the inner dot-products of the orthogonalization process. It is exceptionally well-suited for badly conditioned or highly rectangular Jacobian matrices.

### 6. `LinearAlgebra.hpp`
- A clean aggregator header that exposes all the linear algebra routines (`Core`, `Cholesky`, `LDLT`, `LU`, `QR`) to the broader NMPC solver ecosystem.
