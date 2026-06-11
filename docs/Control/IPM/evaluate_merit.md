### What `evalaute_merit` actually computes
`evaluate_merit(alpha, ...)` evaluates a **scalar merit function** that is used in the line-search of the IPM.

It is a weighted sum of
1. **State deviation from the initial state** (L1-penalty)
2. **Stage cost**  (quadratic terms for tracking, control effort).
3. **Barrier penalty** for every inequality constraint (state limits, input limits, obstacle avoidance).
4. **Prediction consistency penalty** (difference between the linearised dynamics prediction and the actual RK4 simulation).

All terms are multiplied by a large constant `L!_WEIGHT = 1000.0` expect for the quadratic cost terms, so that the barrier terms dominate the line-search.

---
#### 1. Preparation
```C++
matric::StaticVector<double, Nx> x_curr = X_pred[0];
x_curr.saxpy(alpha, riccati.dx[0]);
```
- `x_curr` is the candidate state at the beginning of the horizon, obtained by taking the current nominal prediction `X_pred[0` and adding the newton step $\alpha \cdot \Delta x_{0}$ (where $\Delta x_{0}$ = `riccati.dx[0]`)

The first line-by-line component is the **state-deviation penalty**
```C++
for (std::size_t i = 0; i < Nx; ++i) {
	merit += L1_WEIGHT * std::abs(x_curr(i) - x_init(i));
}
```

##### 1.1. State deviation penalty
$$\text{pen}_{\text{init}}(\alpha)=
\sum_{i=1}^{N_x} w_{\text{L1}}\,
\bigl|\,x_{\text{curr},i}-x_{\text{init},i}\bigr|
,\qquad w_{\text{L1}} = 1000$$
`x_init` is the true current state (the one that was passed to `solve_ipm`).
`x_curr` is the predicted state after taking a step of size $\alpha$
The large weight forces the line-search to keep the trajectory close to the true state.

---
#### 2. Loop over all horizon steps
```C++
for (std::size_t k = 0; k < H; ++k) {
	// Candidate control after the step
	matrix::StaticVector<double, Nu> u_cand = U_guess[k];
	u_cand.saxpy(alpha, riccati.du[k]);
	
	// Candidate next state
	matrix::StaticVector<double, Nx> x_next = X_pred[k+1];
	x_next.saxpy(alpha, riccati.dx[k+1]);
	
	// _ (Cost and barrier terms)
	x_curr = x_next;
}
```

`u_cand` and `x_next` are the candidate control and next state after moving $\alpha$ along the Newton direction.

Inside the loop the following sub-components are added to `merit`.

##### 2.1. Quadratic stage cost
```C++
double err_d = x_curr(1) - config.target_d[k];
double err_mu = x_curr(2);
double err_v = x_curr(3) - config.target_vx;

merit += 0.5 * (config.Q_D * err_d * err_d + 
				config.Q_mu * err_mu * err_mu + 
				config.Q_Vx * err_v * err_v);
merit += 0.5 * (config.Q_vy * x_curr(4) * x_curr(4));
merit += 0.5 * (config.Q_r * x_curr(5) * x_curr(5));
merit += 0.5 * (config.Q_alpha_f * x_curr(6) * x_curr(6));
merit += 0.5 * (config.Q_alpha_r * x_curr(7) * x_curr(7));
merit += 0.5 * (config.R_Steer * u_cand(0) * u_cand(0) +
				config.R_Accel * u_cand(1) * u_cand(1));
```

###### Mathematically : 
$$\begin{aligned}
\text{pen}_{\text{stage},k}(\alpha)=
&\frac12\bigl(Q_D\, (d_k - d_{\text{ref},k})^2 + Q_\mu\, \mu_k^2 + Q_{Vx}\,(v_k - v_{\text{ref}})^2 + Q_{Vy}\,v_{y,k}^2 + Q_{r}\,r_k^2 + Q_{\alpha_f}\,\alpha_{f,k}^2 + Q_{\alpha_r}\,\alpha_{r,k}^2 \bigr)\\ &+ \frac12 \bigl( R_{\text{Steer}}\;\delta_k^2 + R_{\text{Accel}}\;a_k^2\bigr) \end{aligned}$$
- $d$ : Lateral offset
- $\mu$ : Slip (or equivalent)
- $v$ : Longitudinal speed
- $v_{y}$ : Lateral speed
- $r$ : Yaw rate
- $\frac{\alpha_{f}}{\alpha_{r}}$ : Front/back slip angles
- $\delta, \ a$ : Steering, Acceleration (inputs).

##### 2.2. Barrier terms for inequality constraints
The code uses a log-barrier with a linear penalty on the constraint residual:
```C++
auto eval_barrier = [&](double c, double s, double ds) {
	double s_cand = s + alpha * ds;
	if (s_cand <= 1e-8) {
		s_cand = 1e-8;
	}
	return -current_mu * std::abs(s_cand) + L1_WEIGHT * std::abs(c + s_cand);
};
```

For a constraint of the form $(c(x, u) \le 0)$ we maintain a slack variable `s` and its Newton update `ds`.

The merit contribution is
$$\phi_{\text{barrier}}(c,s,ds;\alpha) =
-\,\mu \,\log\bigl(s + \alpha\,ds\bigr)
+ w_{\text{L1}}\;\bigl|\,c + s + \alpha\,ds\bigr|$$
where $\mu$ is the current barrier parameter (`current_mu`).
The large `L1_WEIGHT` forces the line-search to keep the constraints satisfied.

The following constraints are penalised:

| **Constraint**                           | **Expression `c`**                      | **Slack** `s`, **update** `ds` |
| :--------------------------------------- | :-------------------------------------- | :----------------------------- |
| Lateral limit upper : $(d \le d_{\max})$ | `c_d_max = x_curr(1) - config.d_max`    | `duals[k].d_max.s`             |
| Lateral limit over : $(d \ge d_{\min})$  | `c_d_min = config.d_min - x_curr(1)`    | `duals[k].d_min.s`             |
| Input limits : $(u_{i} \le u_{\max, i})$ | `c_u_max = u_cand(i) - config.u_max[i]` | `duals[k].u_max[i].s`          |
| Input limits : $(u_{i} \ge u_{\min, i})$ | `c_u_min = config.u_min[i] - u_cand(i)` | `duals[k].u_min[i].s`          |
| Obstacle avoidance                       | `c_obs = safety_margin^2 - dist^2`      | `duals[k].obs[i].s`            |
For each, the corresponding `ds` is provided by the dual structure (`duals[k].*`)

##### 2.3. Obstacle avoidance term
```C++
double time_future = k * dt;
for (std::size_t i = 0; i < 10; ++i) {
	double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
	double ds = x_curr(0) - obs_pred_s;
	if (std::abs(ds) > 20.0) {
		continue;                // Skip far obstacles
	}
	
	double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
	double dd = x_curr(1) - obs_pred_d;
	
	double dd_eff = dd;
	if (std::abs(dd_eff) < 0.1) {
		dd_eff = (dd_eff >= 0 ? 0.1 : -0.1)
	}
	
	double ds_scaled = ds * 0.5;
	double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
	double safety_margin = obstacles[i].r + config.Obstacle_Margin;
	double c_obs = safety_margin * safety_margin - dist_sq;
	
	merit += eval_barrier(c_obs, duals[k].obs[i].s, duals[k].obs[i].ds);
}
```

###### Mathematical form
$$c_{\text{obs},i} =
\bigl(r_i + \text{margin}\bigr)^2 -
\Bigl(\bigl(\tfrac12 (s_{\text{veh}}-s^{\text{pred}}_i)\bigr)^2
+ \bigl(d_{\text{veh}}-d^{\text{pred}}_i\bigr)^2\Bigr)$$
The barrier is applied only if the vehicle is within a horizontal range $(|ds| < 20)$
The small `dd_eff` clamping (`0.1`) ensures the distance never vanishes, preventing an infinite barrier.

##### 2.4. Dynamics consistency penalty
```C++
for (std::size_t i = 0; i < Nx; ++i) {
	merit += L1_WEIGHT * std::abs((1.0 - alpha) * riccati.d[k](i));
}
```

`riccati.d[k]` is the **affine residual** of the linearised dynamics : $d_{k} = x_{k+1}^{\text{lin}} - x_{k+1}^{\text{nom}}$

The penalty forces the candidate trajectory to stay close to the linearised dynamics:
$$\text{pen}_{\text{dyn},k}(\alpha)=
w_{\text{L1}}\;\bigl|(1-\alpha)\,d_k\bigr|$$

---
#### 3. Full merit expression
Putting everything together, the merit for a step size $\alpha$ is
$$\begin{aligned}

\Phi(\alpha) =\;&

\underbrace{\sum_{i=1}^{N_x}w_{\text{L1}}

\bigl|\,x_{\text{curr},i}(\alpha)-x_{\text{init},i}\bigr|}_{\text{state deviation}}\\

&+\sum_{k=0}^{H-1}\Bigl[

\underbrace{\tfrac12\,x_{\text{curr},k}^\top Q_k x_{\text{curr},k}

+\tfrac12\,u_{\text{candy},k}^\top R_k u_{\text{candy},k}

}_{\text{quadratic cost}}\\

&\quad+\underbrace{\sum_{\text{constr}}

\bigl(-\mu\,\log(s_{\text{cand}})+w_{\text{L1}}\bigl|c+s_{\text{cand}}\bigr|\bigr)}_{\text{barrier terms}}\\

&\quad+\underbrace{w_{\text{L1}}\sum_{i=1}^{N_x}\bigl|(1-\alpha)d_k(i)\bigr|}_{\text{dynamics consistency}}

\Bigr]

\end{aligned}$$

where : 
* $x_{\text{curr},k}(\alpha)=x_{\text{pred},k} + \alpha\,\Delta x_k$
* $u_{\text{candy},k}(\alpha)=u_{\text{pred},k} + \alpha\,\Delta u_k$
* $s_{\text{cand}} = s + \alpha\,ds$ is the candidate slack for each constraint.

The barrier weight $\mu$ is reduced during the IPM iterations, while the L1 weight `W_L1 = 1000` dominates the penalty terms so that the line-search keeps the trajectory feasible and close to the nominal trajectory.

---
#### 4. Why this form?
- **Feasibility first** - The barrier terms dominate, guaranteeing that nay accepted step keeps all constraints satisfied.
- **Objective still matters** - Quadratic terms for tracking and control effort are weighted normally.
- **Stability** - The dynamics-consistency penalty $(1 - \alpha) \cdot d_{k}$ prevents the algorithm from taking a step that would leave the linearised model's validity.
- **Fast line-search** - The merit is cheap to evaluate (only a few vector-scalar operations per step).
- **Numerical safety** - The `if (s_cand <= 1e-8)` clamp and the `abs(ds) > 20` skip for far obstacles avoid overflow and unnecessary computation.

---
#### Bottom line
`evaluate_merit` returns a scalar that balance **tracking performance, control effort, constraint satisfaction** (via log-barrier) and **model fidelity**
The IPM uses this value to decide how far along the Newton direction $(\alpha)$ the solver can move while still reducing the merit, i.e., performing a backtracking line-search that respects the barrier constraints.