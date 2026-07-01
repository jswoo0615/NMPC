# Utils Module

This directory contains the foundational utility functions and macros that support the core NMPC framework, bridging the gap between external autonomous driving stacks (Perception/Planning) and the internal low-level mathematical solver.

## Core Components

### 1. `EnvironmentEvaluator.hpp`
A highly optimized evaluation engine that translates dynamic objects into mathematical constraints for the Interior Point Method (IPM) solver.
- **Dynamic Obstacle Prediction**: Propagates moving obstacles in the Frenet coordinate frame using velocity vectors (`vs`, `vd`) across the NMPC's future prediction horizon (`time_future`).
- **Analytical Constraint Gradients**: Rather than forcing the solver to compute the derivative of the obstacle avoidance barrier function via Automatic Differentiation (AD), this class explicitly injects the exact analytical Jacobian (`J_x`) for the obstacle constraint $c(x)$. This "short-circuiting" significantly reduces the computational burden on the solver.
- **ROI (Region of Interest) Filtering**: Implements a hard spatial cutoff (`ds > 20.0`). Obstacles outside the vehicle's immediate trajectory are instantly dropped (`is_active = false`), strictly bounding the Worst-Case Execution Time (WCET) even in highly crowded environments.

### 2. `CoordinateConvert.cpp`
A mathematical utility designed for Cartesian-to-Frenet coordinate transformation.
- Projects the vehicle's global Cartesian `(x, y)` position onto a piecewise linear sequence of `Waypoint`s.
- Calculates the longitudinal track progression ($s$) and the lateral deviation ($d$). This is a critical pre-processing step required to convert raw Odometry/GPS data into the Frenet state-space expected by the `RealTimeDynamicsModel`.

### 3. `CUDAMacros.hpp`
A lightweight but architecturally vital macro header designed for Heterogeneous Computing.
- Defines the `CUDA_CALLABLE` macro, which resolves to `__host__ __device__` under the NVCC compiler.
- By systematically decorating all matrix operations, math traits, and physics models with this macro, the codebase guarantees that the exact same C++ logic can be executed on standard CPUs (via GCC/Clang) or massively parallel GPUs without writing a single line of duplicated code.
