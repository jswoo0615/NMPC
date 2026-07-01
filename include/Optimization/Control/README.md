# Controller Module

This directory contains the high-level control algorithms and optimization logic for the Non-Linear Model Predictive Control (NMPC) system. It orchestrates the vehicle dynamics, barrier constraints, line-search, and fail-safe mechanisms necessary for robust real-time autonomous driving.

## Core Components

### 1. `SparseNMPC_IPM.hpp` (`SparseNMPC_IPM`)
The primary controller class implementing a **Sparse Nonlinear Model Predictive Control** formulation solved via an **Interior Point Method (IPM)** and Sequential Quadratic Programming (SQP).

**Key Features & Engineering Highlights:**
- **Exact Automatic Differentiation**: Integrates directly with the `AD` module to automatically compute exact Jacobians of the vehicle dynamics along the prediction horizon (using RK4 integration), entirely avoiding the inaccuracy and latency of finite-differences.
- **Interior Point Method (IPM)**: Strictly enforces inequality constraints (actuator limits, road boundaries, dynamic obstacles) by transforming them into logarithmic barrier functions, allowing smooth navigation around obstacles.
- **Sparsity-Preserving Rate Control**: Implements an advanced gradient-injection technique ("Architect's Update") to perfectly account for control rate-of-change (Jerk and Steering Rate) across the entire prediction horizon. Crucially, this is achieved via the Gradient (`q`, `r`) rather than the Hessian (`Q`, `R`), completely preserving the strict block-diagonal sparsity required by the $O(N)$ Riccati solver.
- **Merit-Based Line Search**: Evaluates step candidates using an L1 exact penalty merit function (`evaluate_merit`) to guarantee global convergence and regulate step-size (`alpha`) during highly nonlinear or aggressive transient maneuvers.

### 2. `SafeBuffer.hpp` (`SafeTrajectoryBuffer`)
A critical fail-safe module designed specifically for the unpredictable nature of real-world autonomous driving. 
Because NMPC relies on iterative numerical optimization, there is always a non-zero probability that the solver may diverge, hit a real-time computation timeout, or encounter numerical singularities (NaNs) due to extreme sensor noise or aggressive disturbances.

**Key Features & Engineering Highlights:**
- **Feasible Trajectory Caching**: Continuously caches the full state and control trajectories of the most recently successful optimization cycle.
- **Warm-Starting**: Feeds the IPM solver with a guaranteed-valid initial guess for the next time step, dramatically reducing the required number of SQP iterations.
- **Deterministic Emergency Fallback**: If the solver's internal monitors detect divergence or NaNs, the controller safely aborts the current optimization. It immediately triggers `generate_fallback_control()`, outputting a deterministic fail-safe command (e.g., maintaining straight steering with maximum emergency braking), completely preventing undefined vehicle behavior.
