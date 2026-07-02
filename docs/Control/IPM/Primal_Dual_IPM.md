# Primal-Dual IPM for Real-Time NMPC

!!! abstract "Overview"
    The Interior-Point Method (IPM) is the most powerful engine for solving Nonlinear Programming (NLP) problems with physical constraints (e.g., steering angle limits, obstacle avoidance). However, our architecture for real-time control of autonomous vehicles does not rely on the 'Classical IPM' that assumes infinite computation time. Instead, it adopts a **modern Infeasible Primal-Dual IPM** that maximizes speed and robustness.

## :material-compare: Barrier vs. Penalty

The core of IPM lies in absorbing constraints into the objective function. It is crucial to strictly distinguish between a penalty and a barrier.

<div class="grid cards" markdown>

- :material-chart-bell-curve-cumulative:
    **Quadratic Penalty** ($c(x)^2$)
    
    A 'rubber band' that imposes a penalty proportional to the square of the violation once a boundary is crossed. It temporarily allows boundary violations.

- :material-wall:
    **Logarithmic Barrier** ($-\mu \log(s)$)
    
    A 'concrete wall' that prevents crossing the line at all costs by soaring to infinity ($\infty$) as the variable approaches the boundary ($0$).

</div>

- **Strategy**: By introducing a slack variable, inequality constraints (e.g., $g(x) \le 0$) are transformed into $s > 0$. Applying a log-barrier to this $s$ enforces an 'absolute safe zone (Interior)' against obstacles and physical limits.

## :material-math-integral: Mathematical Heart: KKT Conditions & Central Path

Our solver does not blindly wait for convergence after transforming the objective function, unlike classical methods.

### A. Modified KKT Conditions

The optimal solution must satisfy the Karush-Kuhn-Tucker (KKT) conditions, where the gradient becomes zero. We modify the Complementarity condition to include the barrier parameter $\mu$.

!!! math "Modified Complementarity"
    $$
    s_{i} \cdot \lambda_{i} = \mu
    $$
    
*(The product of the slack variable $s$ and the dual variable $\lambda$ must equal $\mu$)*

### B. Direct Solution via Newton's Method (Primal-Dual Approach)

We solve this entire modified KKT system directly using a Gauss-Newton approximated **Riccati Solver ($\mathcal{O}(H)$ time complexity)**. We find the solution by simultaneously updating the state variables ($x$, Primal) and the dual variables ($\lambda$, Dual).

### C. Dynamic $\mu$ Update (Tracking the Central Path)

To find a solution within the $25\text{ms}$ limit, we mechanically shave off $\mu$ along the central path at every iteration. Based on the current average gap, the code aggressively reduces $\mu$ by a factor of $0.2$ in a single step, rapidly approaching the optimal solution ($\mu \to 0$).

## :material-door-open: Allowing Infeasible Initialization

This is the most defining difference between 1960s textbooks and modern autonomous driving controllers.

!!! warning "Classical Limitation"
    The initial guess $x^{(0)}$ must be perfectly in the 'safe interior', satisfying all constraints, before the algorithm can even start. If the vehicle has already slightly encroached on an obstacle zone or stepped on a lane line when the controller turns on, the solver diverges immediately.

!!! success "Infeasible Primal-Dual IPM"
    Even if the vehicle's physical state ($x$) or control input ($u$) temporarily violates the constraints (**Infeasible**), the solver does not stop. As long as the purely mathematical internal slack variables ($s$) and dual variables ($\lambda$) are forcefully initialized to be strictly positive ($> 0$, **Interior**), the solver effectively grabs the physical trajectory ($x$) by the collar and drags it back inside the feasible region.

## :material-map-legend: Mapping to Optimal Control Problem (OCP)

In our codebase, the mathematical elements of IPM are directly linked to physical reality as follows:

| Mathematical Concept | Dynamics System & Code Implementation | Meaning in Architecture |
| :--- | :--- | :--- |
| **Primal Variables** | State vector ($x_k$), Control input ($u_k$) | Physical control targets (steering angle, acceleration, position, velocity, etc.) |
| **Dual Variables** | Constraint multipliers ($\lambda$), Slack ($s$) | The mathematical magnitude of the repulsive force when attempting to violate a constraint |
| **Objectives** | Stage cost ($L$) based on matrices $Q, R$ | The fundamental compromise between minimizing tracking error and ensuring ride comfort (suppressing steering) |
| **Constraints** | Dynamics ($x_{k+1} = Ax + Bu$), Obstacle avoidance | Linear integrator predictions and obstacle clearance (`Obstacle_Margin`) |
| **Merit Function** | L1 residual + $\log$ barrier penalty | The "final judge" evaluating physical predictability, constraint compliance, and cost minimization into a single scalar value, serving as the brake for the Line Search. |
