# SparseNMPC_IPM API Reference

!!! abstract "Overview"
    This document serves as the low-level API reference for the `SparseNMPC_IPM.hpp` header. While the [Sparse NMPC Overview](SparseNMPC_IPM.md) provides the system architecture and theoretical algorithm flow, this page details the exact C++ data structures, configuration parameters, and class methods.

## :material-tune-vertical: Configuration: `NMPCTuningConfig`

The `NMPCTuningConfig` struct holds all the tunable weights, limits, and reference parameters for the solver.

=== "State Penalties (Q)"
    | Parameter | Default | Description |
    | :--- | :--- | :--- |
    | `Q_D` | 200.0 | Penalty for lateral deviation from the reference path |
    | `Q_mu` | 50.0 | Penalty for heading error relative to the path |
    | `Q_Vx` | 50.0 | Penalty for longitudinal velocity error |
    | `Q_Vy` | 1.0 | Penalty for lateral velocity (sideslip) |
    | `Q_r` | 1.0 | Penalty for yaw rate deviation |
    | `Q_alpha_f`, `Q_alpha_r` | 10.0 | Penalty for front and rear tire slip angles |

=== "Input Penalties (R & Rate)"
    | Parameter | Default | Description |
    | :--- | :--- | :--- |
    | `R_Steer` | 5000.0 | Penalty for absolute steering magnitude |
    | `R_Accel` | 10.0 | Penalty for absolute acceleration magnitude |
    | `R_SteerRate` | 1500.0 | Penalty for steering rate (jerk/slew rate) |
    | `R_AccelRate` | 100.0 | Penalty for acceleration rate (jerk/slew rate) |

=== "Constraints & Bounds"
    | Parameter | Default | Description |
    | :--- | :--- | :--- |
    | `d_max`, `d_min` | 3.5, -3.5 | Maximum and minimum lateral boundaries (road limits) |
    | `u_max`, `u_min` | {0.6, 10}, {-0.6, -10} | Bounds for {Steer, Accel} |
    | `Obstacle_Margin` | 1.5 | Safety margin radius added around dynamic obstacles |

## :material-cube-outline: Core Class: `SparseNMPC_IPM`

```cpp
template <std::size_t H, typename PlantModel, std::size_t Nx_mem, std::size_t Nx_active, std::size_t Nu>
class SparseNMPC_IPM;
```

This is the main template class executing the Primal-Dual Interior-Point Method.

### 1. Template Parameters
- **`H`**: Prediction horizon length.
- **`PlantModel`**: The vehicle dynamics model (default: `RealTimeDynamicsModel`).
- **`Nx_mem`**: Total state dimension in memory.
- **`Nx_active`**: Active state dimension optimized by the solver.
- **`Nu`**: Number of control inputs.

### 2. Internal Structures
```cpp
struct ConstraintState {
    double s = 1.0;     // Slack variable (must be > 0)
    double lam = 1.0;   // Dual variable (multiplier)
    double ds = 0.0;    // Newton step for slack
    double dlam = 0.0;  // Newton step for dual
};
```
Each constraint (e.g., $u_{max}$, $d_{min}$, obstacles) maintains a `ConstraintState` instance. These are grouped into `IPMDuals` arrays along the horizon.

### 3. Key Public Methods

#### `shift_sequence()`
```cpp
inline void shift_sequence();
```
- **Purpose**: Shifts the prediction sequence forward by one time step $\Delta t$.
- **Behavior**: Copies trajectory data $k+1 \rightarrow k$, halves the final control inputs for stability, and predicts the new terminal state $X_{H}$ using Runge-Kutta 4 (RK4) integration.

#### `solve_ipm()`
```cpp
NMPCResult solve_ipm(const matrix::StaticVector<double, Nx_mem>& x_curr, const NMPCTuningConfig& config);
```
- **Purpose**: The main entry point to execute the NMPC optimization.
- **Flow**:
    1. Validates the initial state (`NaN check`).
    2. Runs a forward pass to initialize slacks and dual variables.
    3. Enters the main SQP loop (up to `ipm_max_iter`).
    4. Computes Jacobians via Automatic Differentiation (AD).
    5. Builds the KKT system (`Q, R, q, r`) and solves it using `RiccatiSolver`.
    6. Extracts dual steps and performs a Merit-based Backtracking Line Search.
    7. Checks KKT residual for convergence or divergence.
- **Returns**: `NMPCResult` containing convergence status, iterations taken, and final KKT error.

#### `execute_fallback()`
```cpp
inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason, const NMPCTuningConfig& config);
```
- **Purpose**: Triggered when the IPM diverges, encounters a `NaN`, or fails to factorize the Riccati system.
- **Behavior**: Automatically fetches the last known safe trajectory from `SafeTrajectoryBuffer` to guarantee continuous, collision-free control output.

#### `evaluate_merit()`
```cpp
double evaluate_merit(double alpha, const NMPCTuningConfig& config, const matrix::StaticVector<double, Nx_mem>& x_init, double current_mu);
```
- **Purpose**: Calculates the scalar Merit function value for a given step size `alpha`.
- **Composition**: It balances state/input tracking costs (quadratic), Log-Barrier penalties for inequality constraints, and Exact L1 penalties for dynamics constraint violations.
