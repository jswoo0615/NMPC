# Matrix Core Module

This directory contains the foundational, statically-allocated matrix structures and mathematical traits for the NMPC solver. It is designed to entirely bypass dynamic memory allocation (e.g., `std::vector` or `malloc`) and relies exclusively on stack-allocated, SIMD-accelerated linear algebra operations.

## Design Philosophy & Performance

The core module prioritizes **predictability, cache-locality, and raw throughput** over dynamic flexibility. By fixing matrix dimensions at compile time and explicitly aligning data structures, the solver guarantees the hard real-time execution bounds critical for high-frequency autonomous vehicle control.

## Core Components

### 1. `StaticMatrix.hpp` (`StaticMatrix<T, Rows, Cols>`)
A fixed-size, column-major dense matrix class serving as the primary data container.
- **Cache-Line Alignment**: The underlying flat array is strictly aligned using `alignas(64)`, perfectly synchronizing with standard 64-byte CPU cache lines to prevent cache thrashing and false sharing.
- **Hardware SIMD Intrinsics**: Core operations (copy construction, assignment, `set_zero`, `saxpy`, `+=`, `-=`, `*=`) bypass compiler auto-vectorization. They utilize explicit hardware intrinsics for maximum throughput:
  - **ARM NEON** for edge and embedded processors (e.g., Jetson Nano).
  - **AVX2** for high-performance x86_64 CPUs.
- **Zero-Copy Enforcement**: The `transpose()` method is explicitly marked `[[deprecated]]` to prevent developers from accidentally invoking expensive $O(N^2)$ memory copies. Transposition is strictly handled at the operation level (e.g., using specialized `multiply_AT_B` kernels) to maintain zero-allocation guarantees.

### 2. `StaticMatrixView.hpp` (`StaticMatrixView<T, R, C>`)
A lightweight, non-owning reference into an existing memory block.
- Allows mathematical operations on sub-matrices, blocks, or slices without triggering any memory allocation or data duplication.
- Fully supports custom column strides, enabling seamless views into larger continuous or non-continuous matrix blocks.

### 3. `MathTraits.hpp`
A structural traits layer that elegantly bridges standard floating-point arithmetic with the Automatic Differentiation (AD) module (`Dual`, `DualVec`).
- Provides a unified interface for standard math operations (`abs`, `sqrt`, `max`, `min`, `near_zero`, `get_value`).
- Allows high-level algorithmic logic (such as KKT residual calculations or line search) to be written completely generically. The code complies down to either highly optimized scalar arithmetic or full gradient-tracking AD math depending on the template parameter.
- Employs `CUDA_CALLABLE` macros, architecturally future-proofing the codebase for heterogeneous (CPU/GPU) execution contexts.
