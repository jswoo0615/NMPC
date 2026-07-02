# Merit Function Evaluation

!!! abstract "Overview"
    `evaluate_merit(alpha, ...)` evaluates a **scalar merit function** that is used in the line-search of the Interior Point Method (IPM). It balances tracking performance, control effort, constraint satisfaction (via log-barrier), and model fidelity.
    
    The IPM uses this value to decide how far along the Newton direction ($\alpha$) the solver can move while performing a backtracking line-search.

## :material-function: Merit Function Components

The merit function is a weighted sum of four main components. All penalty and barrier terms are multiplied by a large constant (`L1_WEIGHT = 1000.0`) so they dominate the line-search, while the quadratic tracking costs are weighted normally.

<div class="grid cards" markdown>

- :material-target:
    **State Deviation**
    
    L1-penalty from the initial state.

- :material-cash:
    **Stage Cost**
    
    Quadratic terms for tracking & control effort.

- :material-wall:
    **Barrier Penalty**
    
    Log-barrier for inequality constraints.

- :material-compare:
    **Dynamics Consistency**
    
    Prediction vs. actual simulation penalty.

</div>

## :material-math-integral: Full Merit Expression

!!! math "Complete Merit Function"
    $$
    \begin{aligned}
    \Phi(\alpha) =\;&
    \underbrace{\sum_{i=1}^{N_x}w_{\text{L1}}\bigl|\,x_{\text{curr},i}(\alpha)-x_{\text{init},i}\bigr|}_{\text{State Deviation}}\\
    &+\sum_{k=0}^{H-1}\Biggl[
    \underbrace{\frac{1}{2}\,x_{\text{curr},k}^\top Q_k x_{\text{curr},k}
    +\frac{1}{2}\,u_{\text{cand},k}^\top R_k u_{\text{cand},k}
    }_{\text{Quadratic Cost}}\\
    &\quad+\underbrace{\sum_{\text{constr}}\bigl(-\mu\,\log(s_{\text{cand}})+w_{\text{L1}}\bigl|c+s_{\text{cand}}\bigr|\bigr)}_{\text{Barrier Terms}}\\
    &\quad+\underbrace{w_{\text{L1}}\sum_{i=1}^{N_x}\bigl|(1-\alpha)d_k(i)\bigr|}_{\text{Dynamics Consistency}}
    \Biggr]
    \end{aligned}
    $$

| Variable | Description |
| -------- | ----------- |
| $x_{\text{curr},k}(\alpha)$ | $x_{\text{pred},k} + \alpha\,\Delta x_k$ (Candidate state) |
| $u_{\text{cand},k}(\alpha)$ | $u_{\text{pred},k} + \alpha\,\Delta u_k$ (Candidate control) |
| $s_{\text{cand}}$ | $s + \alpha\,ds$ (Candidate slack for each constraint) |
| $\mu$ | Barrier parameter (reduced during IPM iterations) |
| $w_{\text{L1}}$ | `L1_WEIGHT = 1000.0` |

## :material-cogs: Implementation Details

=== "1. Preparation"
    
    ```cpp
    matrix::StaticVector<double, Nx> x_curr = X_pred[0];
    x_curr.saxpy(alpha, riccati.dx[0]);
    
    for (std::size_t i = 0; i < Nx; ++i) { // (1)
        merit += L1_WEIGHT * std::abs(x_curr(i) - x_init(i));
    }
    ```
    
    1. **State Deviation Penalty**: Evaluates $\text{pen}_{\text{init}}(\alpha)=\sum_{i=1}^{N_x} w_{\text{L1}}\,\bigl|\,x_{\text{curr},i}-x_{\text{init},i}\bigr|$. The large weight forces the line-search to keep the trajectory close to the true initial state `x_init`.

=== "2. Stage Cost"

    ```cpp
    double err_d = x_curr(1) - config.target_d[k];
    double err_mu = x_curr(2);
    // ...
    merit += 0.5 * (config.Q_D * err_d * err_d + /*...*/); // (1)
    merit += 0.5 * (config.R_Steer * u_cand(0) * u_cand(0) + /*...*/); // (2)
    ```
    
    1. **State Error**: Penalizes deviations from the reference trajectory (e.g., lateral offset $d$, velocity $v$, yaw rate $r$).
    2. **Control Effort**: Penalizes steering and acceleration usage.

=== "3. Barrier Terms"

    ```cpp
    auto eval_barrier = [&](double c, double s, double ds) {
        double s_cand = s + alpha * ds;
        if (s_cand <= 1e-8) s_cand = 1e-8; // (1)
        return -current_mu * std::abs(s_cand) + L1_WEIGHT * std::abs(c + s_cand);
    };
    ```

    1. **Numerical Safety**: Prevents the slack variable from becoming zero or negative, which would cause $\log(s)$ to be undefined.

=== "4. Obstacle Avoidance"

    ```cpp
    double ds = x_curr(0) - obs_pred_s;
    if (std::abs(ds) > 20.0) continue; // (1)
    
    // ... calculate c_obs ...
    merit += eval_barrier(c_obs, duals[k].obs[i].s, duals[k].obs[i].ds);
    ```

    1. **Optimization**: Skips evaluating the barrier for obstacles that are too far away longitudinally ($> 20\text{m}$).

=== "5. Dynamics Consistency"

    ```cpp
    for (std::size_t i = 0; i < Nx; ++i) { // (1)
        merit += L1_WEIGHT * std::abs((1.0 - alpha) * riccati.d[k](i));
    }
    ```
    
    1. **Affine Residual**: $\text{pen}_{\text{dyn},k}(\alpha)= w_{\text{L1}}\;\bigl|(1-\alpha)\,d_k\bigr|$. Prevents the algorithm from taking a step that would violate the linearized dynamics prediction.

## :material-shield-alert: Evaluated Constraints

The following constraints are evaluated using the log-barrier function. For each, the corresponding `ds` is provided by the dual structure (`duals[k].*`).

| Constraint | Expression `c` | Slack `s` & Update `ds` |
| :--- | :--- | :--- |
| **Lateral limit upper** ($d \le d_{\max}$) | `x_curr(1) - config.d_max` | `duals[k].d_max` |
| **Lateral limit lower** ($d \ge d_{\min}$) | `config.d_min - x_curr(1)` | `duals[k].d_min` |
| **Input limits upper** ($u_{i} \le u_{\max, i}$) | `u_cand(i) - config.u_max[i]` | `duals[k].u_max[i]` |
| **Input limits lower** ($u_{i} \ge u_{\min, i}$) | `config.u_min[i] - u_cand(i)` | `duals[k].u_min[i]` |
| **Obstacle avoidance** | `safety_margin^2 - dist^2` | `duals[k].obs[i]` |

## :material-star: Design Philosophy

!!! success "Why this specific form?"
    - **Feasibility First**: The log-barrier and L1 terms dominate, guaranteeing that any accepted step keeps all constraints satisfied.
    - **Objective Preservation**: Quadratic terms for tracking and control effort are weighted normally, ensuring optimal behavior within feasible regions.
    - **Algorithmic Stability**: The dynamics-consistency penalty `(1 - alpha) * d_k` prevents stepping too far out of the linearized model's valid region.
    - **Computational Efficiency**: The merit function is extremely cheap to evaluate (only a few vector-scalar operations per step), which is crucial for real-time line-searches.
    - **Numerical Safety**: Hard clamps (`if (s_cand <= 1e-8)`) and spatial culling (`abs(ds) > 20`) prevent overflows, infinite barriers, and unnecessary computation.