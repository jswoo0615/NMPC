## SparseNMPC_IPM.hpp
### Overview
The file `SparseNMPC_IPM.hpp` implements a **Sparse Non-Linear Model Predictive Control (NMPC)** solver based on a **Primal-Dual Interior-Point Method (IPM)**
The solver is written as a header-only C++ class template so that it can be instantiated for different horizon lenghts, plant models and state / input dimensions.

The key idea is to solve the NMPC problem in a *sparse* form
- The dynamics are linearized only once per iteration (via AD) $\rightarrow$ a Riccati Recursion is used to solve the KKT system.
- Constraints (state limits, input limits, obstacle avoidance) are handled in a log-barrier fashion and their dual variables are updated in a simple "predict-correct" scheme.
- A **merit function** is evaluated to drive a line-search and to decide whether to accept a step or fall back to a safe control.

---
### 1. Includes & Namespaces
```c++
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "Optimization/Control/SafeBuffer.hpp"
#include "Optimization/Matrix/AD/DualVec.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Simulation/Integrator.hpp"
#include "Optimization/Solver/KKTMonitor.hpp"
#include "Optimization/Solver/MeritLineSearch.hpp"
#include "Optimization/Sovler/RiccatiSolver.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
```
- `SafeBuffer.hpp` - Keeps a recent safe trajectory that can be returned if the IPM diverges.
- `DualVec.hpp` - An automatic-differentiation (AD) variable used to compute Jacobians of the dynamics.
- `MathTraits.hpp` - Provides math utilities such as `max`, `min`, `isnan`, etc.
- `StaticMatrix.hpp` - Compile-time sized matrices/vectors (used for the Riccati Solver)
- `Integrator.hpp` - RK4 integration for the plant model.
- `KKTMonitor.hpp` - Computes infinity-norm of the KKT residual.
- `RiccatiSolver.hpp` - Solves the KKT linear system via a Riccati recursion.
- `RealTimeDynamicsModel.hpp` - Default plant model (e.g. bicycle model).

The main class is inside `namespace Optimization::controller`.

---
### 2. Helper Data STructures
#### 2.1. `ObstacleFrenet`
```c++
struct ObstacleFrenet {
    double s = 0.0;             // Longitudinal Position
    double d = 0.0;             // Lateral Offset
    double r = 0.5;             // Radius
    double vs = 0.0;            // Longitudinal Velocity
    double vd = 0.0;            // Lateral Velocity
};
```
- Holds a Frenet-like description of an obstacle (Position, Radius, Motion).
- The solver assumes at most 10 obstacles (`obstacles[10]`).

#### 2.2. `NMPCResult`
```C++
struct NMPCResult {
    bool success = false;
    bool fallback_triggered = false;
    double max_kkt_error = 0.0;
    int sqp_iterations = 0;
    std::string status_msg = "OK";
};
```
- Returned by `solve_ipm`.
- Stores convergence information and whether a fallback was performed.

#### 2.3. `NMPCTuningConfig`
Contains all tunable weights and constraints.
Some parameters are not used by the hard-PD IPM (e.g. `Obstacle_Penalty`, `W_slack`) but are kept for API compatibility.

##### Key  Fields
|**Field**|**Meaning**|
|:---|:---|
|`Q_D`, `Q_mu`, `Q_Vx`, `Q_Vy`, `Q_r`, `Q_alpha_f`, `Q_alpha_r`|State weighting terms|
|`R_Steer`, `R_Accel`|Input weighting terms|
|`R_Steer_Rate`, `R_Accel_Rate`|Not used - legacy|
|`Obstacle_Margin`|Safety margin added to obstacle radius|
|`W_slack`|Safety margin added to obstacle radius|
|`damping_Q`, `damping_R`|Regularization added to Hessian|
|`d_max`, `d_min`|Lateral limits|
|`u_min`, `u_max`|Input limits|
|`kappa`|Road curvature (affects dynamics)|
|`target_vx`|Target longitudinal speed|
|`target_d[100]`|Desired lateral offset over horizon (only first `H` values used)|
|`ipm_max_iter`|Max IPM iterations (default 8)|
|`kkt_tolerance`|Convergence tolerance (default 1e-2)|

---
### 3. Class Templates : `SparseNMPC_IPM`
```C++
template <size_t H, typename PlantModel = Dynamics::RealTimeDynamnicsModel, size_t Nx = 8, size_T Nu = 2>
class SparseNMPC_IPM {};
```
- `H` - Prediction horizon (steps)
- `PlantModel` - The dynamics model (default `RealTimeDynamicsModel`)
- `Nx` - State dimension (default 8)
- `Nu` - Input Dimension (default 2)

#### 3.1. Internal Types
```C++
struct ConstraintState {
    double s = 1.0;             // Slack Variable
    double lam = 1.0;           // Dual Varaible
    double ds = 0.0;            // Change of Slack
    double dlam = 0.0;          // Change of dual
};

struct IPMDuals {
    ConstraintState d_max;
    ConstraintState d_min;
    ConstraintState u_max[2];
    ConstraintState u_min[2];
    ConstraintState obs[10];
};
```
- Each constraint has a slack `s` $(\geq 0)$ and a dual $(\lambda)$
- The `du` members (`ds`, `dlam`) are used during the Newton step to update the duals.

#### 3.2. Member Variables
|**Variable**|**Purpose**|
|:---|:---|
|`double dt`|Time step between prediction steps.|
|`U_guess[H]`|Current control trajectory guess|
|`X_pred[H+1]`|Predicted state trajectory (including terminal state).|
|`duals[H]`|Dual variables for every time step.|
|`u_last`|Last applied control (used for fallback)|
|`obstacles[10]`|List of obstacles.|
|`mu`|Barrier parameter $(\mu)$|
|`riccati`|Instance of `RiccatiSolver` - Solves the KKT System.|
|`safe_buffer`|Stores a safe trajectory that can be used when the IPM fails.|

All containers and `std::array` with compile-time size which keeps the memory footprint small and cache-friendly.

#### 3.3. Constructor
Initializes :
- `dt = 0.05` (20Hz sampling)
- All control guesses to zero
- Duals to defaults $(s = 1, \lambda = 1)$
- Obstacles to far away (`s = d = 10000`) so they are initially inactive.
- `u_last` to zero.

---
### 4. Core Methods
#### 4.1. `shift_sequence()`
- Slides all trajectories one step forward (typical in MPC).
- Halves the last control command (to smooth the transition).
- Predicts the next terminal state via RK4.

#### 4.2. `execute_fallback()`
- Called when the IPM fails (`NaN`, KKT divergence, Riccati failure).
- Sets `success=false`, `fallback_trggered=true`, and a status message.
- If `safe_buffer` contains a valid trajectory, it is copied back.
- Otherwise it sets all controls to a "safe" control returned by `safe_buffer.extract_fallback_control`.
- Returns the updated `NMPCResult`.

#### 4.3. `evaluate_merit()`
A **merit function** used for line-search and convergence monitoring.
Components:
1. **State deviation** : From the initial state weighted by `L1_WEIGHT` (1000).
2. **Stage cost** : Weighted sum of tracking errors (`Q_*`) and input costs (`R_*`).
3. **Barrier terms** for every constraint:
    - Lateral limits `d_max / d_min`
    - Input limits `u_max / u_min`
    - Obstacle avoidance (distance to each obstacle)
4. **Simulation consistency** : L1 distance between predicted next state and RK4 simulation of the current state/input.
5. **Barrier penalty** : `-current_mu * std::log(s_cand) + L1_WEIGHT * std::abs(c + s_cand);` (Penalizes violating the constraint).
    - If slack becomes $\leq 1e-12$, a huge penalty (`1e9`) is added.

The function returns the scalar merit value for a given step size `alpha` (used in the line-search).
The code uses a lambda `eval_barrier` to avoid code duplication.

#### 4.4. `solve_ipm()`
The heart of the solver. Roughly the algorithm:
1. **Preliminary checks** - If the lateral position or speed is `NaN`, fallback immediately.
2. **Initialize** :  
    - Copy current state to `X_pred[0]` and ensure speed $\geq 0.1$.
    - Set `mu = 1.0`.
    - For each step `k`:
        - Predict next state with RK4 using current `U_guess[k]`.
        - Compute initial duals for constraints (slack `s` = max(previous s, max(0.1, -c))).
        - Compute $\lambda = \mu / s$
        - Store obstacle constraint slack and $\lambda$.
3. **IPM Loop** (`ipm_max_iter` iterations)
    - **(a) Jacobian update** every other step (even indices):
        - Build AD variables for `x` and `u` (using `DualVec`)
        - Compute `x_next_dual = step_rk4(...)`.
        - Extract Jacobians `A`, `B` and residual `d` from the AD gradients.
    - **(b) Build KKT Matrices** : 
        - Reset `Q`, `R`, `q`, `r` for every step.
        - Populate based on stage cost and barrier terms (using `apply_condensed`).
        - Add regularization `dmaping_Q`, `damping_R`.
    - **(c) Terminal cost** - Add weighted terminal penalties on `d`, `mu`, `v`. 
    - **(d) Solve KKT** using the Riccati Recursion
        - If failure $\rightarrow$ fallback.
    - **(e) Compute Newton step for duals** : 
        - `extract_dual` computes `ds`, `dlam` from the dual KKT equations.
        - Determines `alpha_max` (step size that keeps slacks and $\lambda$ positive).
    - **(f) Merit line-search** : 
        - Compute current merit (`alpha = 0`).
        - Evaluate merit for a candidate `alpha` up to `alpha_max`.
        - Reduce `alpha` by factor 0.5 until Armijo condition is satisfied (or max 10 reductions).
    - **(g) Apply step** : 
        - Update `X_pred`, `U_guess`, and all duals using `alpha`.
        - Update terminal state `X_pred[H]`.
    - **(h) Compute KKT residual** (infinity norm of `du_vec`) : 
        - If NaN or > 20 $\rightarrow$ fallback.
    - **(i) Update barrier parameter** : 
        - `mu = max(1e-4, 0.2 * average_gap)`
        - `average_gap` = sum($s * \lambda$)/num_constraints.
    - **(j) Convergence check** : 
        - If `kkt_residual < kkt_tolerance` and `average_gap <= 1e-3`, exit early.
        - Commit the trajectory to `safe_buffer`
4. Finalization
    - Set `u_last` to first control.
    - Mark `success=true`.
    - If status message unchanged, set to `"IPM Solved"`.
    - Commit the trajectory to `safe_buffer`.
    - Return the `NMPCResult`.

##### 4.4.1. Important Implementation Details
- **Barrier update** - The standard barrier method $\lambda = \mu / s$
    - Dual updates use `cs.dlam = (mu - cs.lam * cs.s - cs.lam * cs.ds) / cs.s;` (derived from the derivative of $\lambda$).
- **Constraint Jacobians** - Computed by AD only every other iteration (`update_jacobian` flag)
    - Saves computational time because Jacobian change slowly.
- **Regularization** - `damping_Q`, `damping_R` are added to the Hessian to keep it SPD.
- **Obstacle constraints** - Expressed as `c_obs = safety_margin^2 - dist^2`.
    - If the vehicle is inside the safety circle, `c_obs < 0` $\rightarrow$ active barrier.
- **Merit function** - Includes both cost and barrier terms, ensuring a decrease in both.
- **Line-search** - Uses a simple backtracking strategy (repeated halving).
    - The Armijo condition is very loose (`1e-4` relative reduction), which is typacal for IPM.


### 5. Performance Notes
- **Static sizing** - All matrices/vectors are compile-time sized(`StaticMatrix`, `StaticVector`), which eliminates dynamic allocation and improves cache locality.
- **AD Jacobian update every other step** - Reduces the number of expensive AD evaluations by 50%.
- **Riccati Recursion** - Solves the KKT system in `O(H * max(Nx, Nu)^2)` operations, which is optimal for sparse MPC.
- **Barrier terms** - The barrier cost is evaluated in `evaluate_merit` - this adds overhead but is necessary for line-search.
- **Line-search** - Backtracking with at most 10 evaluations is cheap relative to solving the Riccati recursion.

If the solver is used in a real-time loop at 20Hz, the current design should comfortably fit within the time budget for moderate horizons (`H <= 20`). For very long horizons or higher-dimensional models, one might consider:
    - **Parallelizing** the Jacobian computation across steps (each step is independent)
    - **Using a more advanced IPM** (e.g., primal-dual IPM with predictor/corrector).
    - **Exploiting sparsity** in the constraint Jacobian (e.g., using sparse matrices instead of dense `StaticMatrix`).

### 6. How to Solver Works - A Step-by-Step Flow
1. Receive current state `x_curr`
2. Check for NaNs - fallback if needed.
3. Initialize prediction and duals using current guess
4. For ipm_step = 1 ... ipm_max_iter
    a. Update Jacobians (every other step) via AD.
    b. Build Q, R, q, r for each stage (cost + barrier)
    c. Add terminal cost.
    d. Solve the KKT system (Riccati recursion).
    e. Compute dual step and alpha_max.
    f. Perform merit line-search to find optimal_alpha.
    g. Update states, inputs, and duals with optimal_alpha.
    h. Compute KKT residual -> fallback if too large.
    i. Update mu (barrier parameter) from average primal-dual gap.
    j. If KKT residual and gap below thresholds -> exit early.
5. Commit the latest trajectory to safe_buffer.
6. Return NMPCResult (success, interations, etc).

### 7. Summary
The `SparseNMPC_IPM` class is a compact, template-based implementation of a primal-dual interior-point method for NMPC.
- **Sparse** - Uses a Riccati Recursion, keeping the KKT system low-dimensional.
- **Interior-Point** - Handles inequality constraints via log-barrier terms, with duals updated analytically.
- **Adaptive** - Barrier parameter $\mu$ is updated every iteration based on the primal-dual gap.
- **Safe fallback** - Maintains a safe trajectory and can revert to a conservative control if the solver diverges.

This code is well-structured, heavily commented, and suitable for real-time control on embedded platforms (thanks to static sizing and no dynamic allocations). The main trade-off is the need to hand-craft many parameters (`NMPCTuningConfig`) and the rist of hard-coded array sizes. With a few tweaks (e.g., dynamic obstacle handling, more adaptive line-search), it could serve as a robust baseline for many autonomous driving or robotics applications.