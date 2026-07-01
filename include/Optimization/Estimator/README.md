# Estimator Module

This directory contains the state estimation algorithms designed to fuse noisy sensor measurements and provide the NMPC controller with clean, accurate state feedback. These estimators are built with a strict focus on predictable Worst-Case Execution Time (WCET) and zero dynamic memory allocation, making them ideal for hard real-time execution on embedded edge devices (e.g., NVIDIA Jetson Nano).

## Core Components

### 1. `EKF.hpp` (Extended Kalman Filter)
A high-performance implementation of the Extended Kalman Filter for real-time state estimation.
- **Exact Linearization**: Leverages the `AD` (Automatic Differentiation) module to dynamically compute the exact Jacobian ($F$) of the highly non-linear vehicle dynamics through the RK4 integrator. This avoids the numerical drift common in finite-difference methods.
- **Zero-Allocation Covariance Update**: Features an "Architect's Update" that completely eliminates the need to allocate or compute a transposed matrix ($F^T$) during the heavy covariance prediction step ($P_{k|k-1} = F P_{k-1|k-1} F^T + Q$). Instead, it computes the matrix multiplication directly in-place using explicit **ARM NEON** and **AVX2** SIMD intrinsics.
- **Robust Inversion**: Uses the module's custom `LDLT` decomposition to invert the innovation covariance matrix ($S$), mathematically preventing division-by-zero crashes caused by numerical singularities.

### 2. `HistoryBuffer.hpp`
A heavily optimized circular buffer used to store past states, measurements, and control inputs for the Moving Horizon Estimator.
- **Branchless & Modulo-Free**: To eliminate ALU bottlenecks on low-power embedded processors, the buffer's index wrapping logic completely avoids the computationally expensive modulo (`%`) operator. It instead uses conditional arithmetic that compiles down to single-cycle branchless conditional-select (CSEL) instructions.
- **SIMD Aligned**: Built directly on top of `StaticVector`, guaranteeing that all historical data pushes and lookups leverage 64-byte aligned SIMD memory load/store operations.

### 3. `SparseMHE.hpp` (Moving Horizon Estimator)
A lightweight Sparse Moving Horizon Estimator (MHE) designed to handle severe non-linearities and bounded measurement noise over a sliding time window.
- **Bounded Iterations**: Strictly limits the internal Gauss-Newton SQP loop to a very small maximum (`MAX_ITER = 2`) to mathematically guarantee a bounded Worst-Case Execution Time (WCET) for real-time safety.
- **Comprehensive Cost Formulation**: Simultaneously optimizes across Arrival Cost (prior knowledge), Measurement Cost (sensor data), and Dynamic Feasibility Cost (physics model constraints).
- **Cache-Coherent AD Extraction**: Includes an "Architect's Update" to extract AD-generated Jacobians directly into a column-major `StaticMatrix`. This ensures the subsequent dense Hessian assembly ($H = J^T W^2 J$) perfectly hits the L1 data cache, drastically reducing CPU cycle waste.
- **Fault Tolerance**: Includes primal filtering that immediately aborts optimization if severe sensor faults (`NaN` or extreme outliers) are detected in the history buffer, preventing the estimator from diverging and corrupting the controller.
