# Automatic Differentiation (AD) Module

This directory contains a highly optimized, from-scratch Automatic Differentiation (AD) library tailored for the NMPC solver. It utilizes forward-mode AD with dual numbers to evaluate functions and their exact gradients simultaneously without any finite-difference approximations.

## Design Philosophy & Performance

The AD module is explicitly engineered for **hard real-time execution** on both x86_64 architectures and edge devices (e.g., ARM-based Jetson Nano). 

1. **Zero Dynamic Allocation**: All gradient arrays are stack-allocated and statically sized at compile time, guaranteeing deterministic execution times.
2. **Cache Line Optimization**: Arrays are explicitly aligned (`alignas(64)`) to perfectly fit 64-byte CPU cache lines, preventing cache misses and false sharing.
3. **SIMD Hardware Acceleration**: Loops for vector operations are aggressively unrolled and mapped directly to hardware vector instructions (Intrinsics), bypassing the compiler's auto-vectorizer to ensure maximum throughput.

## Core Components

### 1. `DualScalar.hpp` (`Dual<T>`)
Provides a scalar dual number implementation for 1-dimensional differentiation. 
- Tracks a primal value (`v`) and a single derivative (`d`).
- Overloads standard arithmetic operators (`+`, `-`, `*`, `/`) with exact analytic derivative chain rules.

### 2. `DualVec.hpp` (`DualVec<T, N>`)
A multi-dimensional dual number structure designed for computing Jacobians and gradients of $f: \mathbb{R}^N \rightarrow \mathbb{R}$.
- **Hardware Intrinsics**: Uses explicit compiler intrinsics for gradient array operations (addition, subtraction, multiplication, quotient rules).
  - **ARM NEON**: Optimized for ARM edge architectures (`float32x4_t`, `float64x2_t`).
  - **AVX2**: Optimized for x86_64 architectures (`__m256`, `__m256d`).
- **SIMD Zeroing**: Instantiation and zero-initialization bottlenecks are eliminated using SIMD zero-set instructions.

### 3. `DualMath.hpp`
Provides AD support for transcendental and non-linear mathematical functions.
- Implements exact gradient propagation for functions such as `abs`, `sin`, `cos`, `atan2`, `sqrt`, and `atan`.
- Complex gradient chain rules (e.g., the quotient rule inside `atan2`) are also fully vectorized using AVX2 and NEON to maintain real-time performance.

### 4. `Dual.hpp`
An aggregator header that cleanly exposes the primary types (`Dual`, `DualVec`) and namespace aliases (`Optimization::ad`) to the rest of the solver.

## Usage Example

```cpp
#include "Optimization/Matrix/AD/Dual.hpp"

using namespace Optimization;
constexpr std::size_t N = 4; // Number of variables
using DVec = DualVec<float, N>;

// Initialize variables, marking their index for the gradient array
DVec x = DVec::make_variable(1.5f, 0); 
DVec y = DVec::make_variable(2.0f, 1);

// Math operations automatically and exactly compute gradients
DVec result = ad::sin(x) * y + ad::atan2(y, x);

// Access results:
// result.v -> The evaluated primal value
// result.g -> The computed gradient vector [df/dx, df/dy, 0, 0]
```
