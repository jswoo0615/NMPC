## Riccati Sovler
### 1. High-level structure
The solver implements the classic **discrete-time Riccati recursion** for a linear-quadratic problem that appears inside each IPM iteration of the NMPC controller.

#### Problem Statement
$$\min_{u_{k}, x_{k+1}} \frac{1}{2}\sum_{k=0}^{H-1}\Big[x_{k}^{T}Q_{k}x_{k} + q_{k}^{T}x_{k} + u_{k}^{T}R_{k}u_{k} + r_{k}^{T}u_{k} \Big] + \frac{1}{2}x_{H}^{T}Q_{H}x_{H} + q_{H}^{T}x_{H}$$
$$\text{s.t.} \ x_{k+1} = A_{k}x_{k} + B_{k}u_{k} + d_{k}$$

|**Section**|**What is contains**|**Equations**|
|:---|:---|:---|
|`A`, `B`, `d`|Linearised dynamics $x_{k+1} = A_{k}x_{k} + B_{k}u_{k} + d_{k}$|$x_{k+1} = A_{k}x_{k} + B_{k}u_{k} + d_{k}$|
|`Q`, `R`, `q`, `r`|Quadratic cost metrices/vectors for each stage and terminal stage|See objective above|
|`P`, `p`|Value-function (cost-to-go) parameters computed in the backward pass|$V_{k}(x) = \frac{1}{2}x^{T}P_{k}x + p_{k}^{T}x$|
|`K`, `k_ff`|Optimal policy $du_{k} = K_{k}dx_{k} + k_{ff, k}$|$\Delta u_{k} = K_{k}\Delta x_{k} + k_{ff, k}$|
|`dx`, `du`|State and input updates produced in the forward pass|$\Delta x_{k}$, $\Delta u_{k}$|
|`solve()`|Runs the backward and forward passes, returns `SolverStatus`|-|

All containers are `std::array` of `matrix::StaticMatrix/StaticVector`, meaning **no dynamic allocation** and the size is known at compile time.
The solver is parameterised by the horizon `H`, state dimension `Nx`, and input dimension `Nu`.

---
### 2. Details walk-through
#### 2.1. Member variables
```C++
std::array<matrix::StaticMatrix<double, Nx, Nx>, H> A;
std::array<matrix::StaticMatrix<double, Nx, Nu>, H> B;
std::array<matrix::StaticVector<double, Nx>, H> d;
```
- `A_k`, `B_k`, `d_k` are the linearised dynamics at every step.
- These are filled by the IPM before calling `solve()`.

```C++
std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1> Q;
std::array<matrix::StaticMatrix<double, Nu, Nu>, H> R;
std::array<matrix::StaticVector<double, Nx>, H + 1> q;
std::array<matrix::StaticVector<double, Nu>, H> r;
```
- Stage cost $\frac{1}{2}x^{T}Q_{k}x + q_{k}^{T}R_{k}u + r_{k}^{T}u$
- Terminal cost stored in `Q[H]`, `q[H]`

```C++
std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1> P;
std::array<matrix::StaticVector<double, Nx>, H + 1> p;
```
- Value-function parameters : $V_{k}(x) = \frac{1}{2}x^{T}P_{k}x + p_{k}^{T}x$
- Computed in the **backward pass**.

```C++
std::array<matrix::StaticMatrix<double, Nu, Nx>, H> K;
std::array<matrix::StaticVector<double, Nu>, H> k_ff;
```
- Feedback gain `K_k` and feed-forward term `k_ff_k`.
- Policy : `du_k = K_k dx_k + k_ff_k` (note the *plus* sign - the code later does `du = K * dx + k_ff` and `k_ff` already contains the negative sign).

```C++
std::array<matrix::StaticVector<double, Nx>, H + 1> dx;
std::array<matrix::StaticVector<double, Nu>, H> du;
```
- State and input updates produced in the **forward pass**.

#### 2.2. `solve(double reg_u = 1e-6, double reg_x = 0.0)`
> Returns SolverStatus (SUCCESS, MATH_ERROR, ...)

##### 2.2.1. Initialisation
$$P_{H} = Q_{H}, \quad p_{H} = q_{H}, \quad P_{H}(i, i) += reg_{x}$$
```C++
P[H] = Q[H];
p[H] = q[H];
for i : P[H](i, i) += reg_x;
```
- Terminal value function is set to the terminal cost.
- Optional diagonal regularisation (`reg_x`) is added to the Hessian to keep it positive-definite.

##### 2.2.2. Backward Pass (`k = H-1, ... 0`)
The backward pass follows the classic discrete-time Riccati recursion for a linear-quadratic problem, but it is written **without dynamic memory allocation**

|**Step**|**Mathematical expression**|**Code Snippet**|
|:---|:---|:---|
|1. Intermediate products|$P_{k+1}A_{k}, \\ P_{k+1}B_{k}$|`P_next_A = P[k+1] * A[k];` `P_next_B = P[k+1] * B[k]`|
|2. `Quu` & `Qux`|$Q_{uu} = R_{k}+B_{k}^{T}P_{k+1}B_{k} \\ Q_{ux} = B_{k}^{T}P_{k+1}A_{k}$|`Quu = R[k] + B^T * P_next_B;` `Qux = B^T * P_next_A;`|
|3. Affine parts|$p_{next, t} = p_{k+1} + P_{k+1}d_{k} \\ q_{k} = r_{k}+B_{k}^{T}p_{next, d}$|`p_next_d = p[k+1] + P[k+1] * d[k];` `q_u = r[k] + B^T * p_next_d;`|
|4. Levenberg-Marquardt regularisation|$Q_{uu} \leftarrow Q_{uu} + reg_{u}I$|`for (i) Quu(i, i) += reg_u;`|
|5. Positive-definite check & solve|Solve $Q_{uu}^{-1}$ via LDLT|`linalg::LDLT_decompose(Quu);` `linalg::LDLT_solve(Quu, ...);`|
|6. Policy|$K_{k} = -Q_{uu}^{-1}Q_{ux} \\ k_{ff, k} = -Q_{uu}^{-1} q_{u}$|Column-wise solve for `K[k];` solve once for `k_ff[k]`|
|7. Value-function update|$P_{k} = Q_{k} + A_{k}^{T}P_{k+1}A_{k} + K_{k}^{T} Q_{ux} \\ p_{k} = q_{k} + A_{k}^{T}p_{next,d} + Q_{ux}^{T}k_{ff, k}$|Compute `AT_PA`, `KT_Qux`, `AT_Pnd`, `QuxT_kff`|

Regularisation (`reg_x`) is added to the diagonal of `P_k` after each update.
Symmetry of `P_k` is enforced explicitly to counteract round-off errors.

##### 2.2.3. Forward Pass
```C++
dx[0] = 0;
for k = 0, ... ,H - 1:
    du[k] = K[k] * dx[k] + k_ff[k];
    dx[k+1] = A[k] * dx[k] + B[k] * du[k] + d[k];
```
- No regularisation needed because the policy has already been stabilised during the backward pass.
- The forward pass produces the actual state and input perturbations that will be applied to the nominal trajectory.

##### 2.2.4. Result
If the backward pass finishes without a math error, the method returns `SolverStatus::SUCCESS`.

### 3. How this solver is used in `SparseNMPC_IPM`
In `SparseNMPC_IPM::solve_ipm()` the Riccati solver is invoked after the Jacobian and cost parameters (`A, B, d, Q, R, q, r`) has been constructed for the current IPM iteration.
```C++
if (riccati.solve() != SolverStatus::SUCCESS) {
    return execute_fallback(...);
}
```
- The solver provides:
    - `du[k]` : The Newton step for the control trajectory
    - `dx[k]` : The corresponding state update (used to compute dual updates and the merit function).

- These vectors are used in the **IPM Loop** to:
    - Update `U_guess` (controls) and `X_pred` (states).
    - Update dual variables `duals[k]`.
    - Compute the KKT residual (`du_vec`) for convergence checking.

