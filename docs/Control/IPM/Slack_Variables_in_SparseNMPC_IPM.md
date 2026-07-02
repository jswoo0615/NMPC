# Technical Note: Slack Variables and L1 Penalty

!!! info "Background"
    This document summarizes common misconceptions encountered while implementing a primal-dual Interior Point Method for Sparse NMPC. 
    
    It is intended to explain the intuition behind the numerical behavior rather than provide a rigorous mathematical proof.

## :material-help-circle: Question 1: How can $s = 3$ when $c = -5$?

**Question:**
> Even if the vehicle is very safe, being 5m away from an obstacle ($c = -5$), if the solver's internal slack variable is calculated as $s = 3$, it results in $|-5 + 3| = |-2| = 2$, incurring a massive penalty of 2000 ($w_{L1} \times 2$). 
> If $s = 3$, since slack variables are introduced as $g(x) + s \ge 0$, does this mean the constraint boundary flexibly shifted to $g(x) \ge -3$?

### 1. Mathematical Correction: It's an Equality, Not an Inequality

It is intuitive to think that introducing a slack variable to $g(x) + s \ge 0$ shifts the boundary to $g(x) \ge -3$ when $s = 3$. However, the actual mathematical transformation in IPM works differently.

When the physical constraint (e.g., safe margin violation) is $c(x) \le 0$, the IPM solver introduces a virtual positive variable $s$ ($s > 0$) to forcefully convert it into a **perfect Equality**.

!!! math "Equality Constraint"
    $$
    c(x) + s = 0 \quad \text{subject to } s > 0
    $$

- **Consistent State**: If the vehicle is 5m away from an obstacle ($c(x) = -5$), the slack variable $s$ must be exactly $5$ for the equation to equal $0$ ($-5 + 5 = 0$).

### 2. Then why does $s = 3$ emerge when $c = -5$?

Here lies the most critical characteristic of the Primal-Dual IPM.
In the "brain" of the solver, the physical state $x$ (vehicle position, etc.) and the virtual variable $s$ (slack variable) are treated as **completely independent entities** and are calculated and updated separately.

<div class="grid cards" markdown>

- :material-calculator:
    **1. Newton Step (Direction Calculation)**
    
    The solver simultaneously calculates how much to change $x$ ($\Delta x$) and how much to change $s$ ($\Delta s$) by solving a system of equations ([RiccatiSolver](../Solver/RiccatiSolver.md)).

- :material-arrow-decision:
    **2. Line Search (Applying Step Size $\alpha$)**
    
    Both variables take a step of size $\alpha$:
    
    - New physical state: $x_{\text{new}} = x + \alpha \Delta x \implies$ results in $c(x_{\text{new}}) = -5$
    - New slack variable: $s_{\text{new}} = s + \alpha \Delta s \implies$ results in $s_{\text{new}} = 3$

</div>

**Why don't they match?** 
Because **$c(x)$ is a non-linear function**, while the $\Delta x$ and $\Delta s$ calculated by the solver are based on **linearized approximations** (Jacobian). By taking a large step ($\alpha$) assuming the world is linear, a discrepancy (Inconsistency) of $2$ occurs between the physical reality ($c = -5$) and the mathematical expectation ($s = 3$).

### 3. The L1 Penalty Intervention: "Do Not Lie"

The true meaning of $|-5 + 3| = |-2| = 2$ becomes clear.
Since the car is 5m away ($c = -5$), it is physically very safe with no collision.
However, mathematically, the absolute rule $c(x) + s = 0$ has been broken: $-5 + 3 = -2 \neq 0$. The solver's physical world and mathematical world have lost synchronization; they are "lying" to each other.

!!! warning "L1 Penalty Response"
    **[evaluate_merit](evaluate_merit.md)**: "I acknowledge the car is physically safe, but the difference between your calculated physical state ($c$) and virtual state ($s$) is off by 2. This means the numerical analysis is collapsing. Penalty: 2000! Discard this step size ($\alpha$) and recalculate!"

### Conclusion

The constraint boundary does *not* flexibly shift to $g(x) \ge -3$.
Conversely, the physical law $c(x)$ and the mathematical virtual variable $s$ must be perfectly bound by $c(x) + s = 0$. The moment they diverge even slightly (e.g., $c = -5$, $s = 3$), a strict L1 penalty is imposed to prevent trajectory derailment caused by linearization errors.

---

## :material-help-circle-outline: Question 2: Aren't Slack Variables Tied to Constraints?

**Question:**
> Aren't slack variables inherently related to constraints? Why are they separated?

### Answer: Theoretical Unity vs. Numerical Separation

Theoretically, you are 100% correct. However, the moment the math on paper is translated into a **C++ Numerical Solver**, they are divorced into completely independent variables.

=== "1. Solver Structure: Dimensional Lifting"

    If the slack variable were completely tied to the constraint, the solver could just substitute $-c(x)$ for $s$ and solve:
    $$-\mu \log(-c(x))$$
    
    However, if $c(x)$ is a non-linear function, trapping a non-linear function inside a log function causes the Jacobian and Hessian matrices to become incredibly tangled. It becomes a horrific computation impossible to solve within $25\text{ms}$.
    
    Instead, IPM designers approach it differently: **"Let's elevate the slack variable $s$ from a dependent variable of $x$ to a 'completely independent optimization variable' on par with $x$!"**
    
    - **Original Variable**: $x$ (vehicle state, steering, etc.)
    - **IPM Variable**: $Z = [x, s, \lambda]^\top$
    
    Now, in the eyes of the solver, $x$ and $s$ are just array indices `Z[0]` and `Z[1]`. They have gained the right to move completely independently.

=== "2. The Betrayal of Linearization (Linearization Error)"

    Taking these two independent variables $x$ and $s$, the solver uses Newton's Method for the next step. Newton's Method predicts the complex curved world by bluntly simplifying it into "straight lines" (tangents).
    
    The solver solves the system of equations to calculate independent steps ($\Delta x$, $\Delta s$):
    
    - $x_{\text{new}} = x + \alpha \Delta x$
    - $s_{\text{new}} = s + \alpha \Delta s$
    
    Here, tragedy strikes:
    
    - $s$ is originally a virtual variable, so it moves perfectly in a **straight line** as predicted.
    - However, the physical reality $c(x_{\text{new}})$ created by the movement of $x$ is a **non-linear curve**.
    
    Assuming a straight path, the solver takes a large step ($\alpha$). But the actual physical world $c(x)$ curves away, breaking the promise $c(x_{\text{new}}) + s_{\text{new}} = 0$ between the two independent variables.

=== "3. The True Role of L1 Penalty: A Mathematical Rubber Band"

    In summary, **while the slack variable $s$ is inherently born from the constraint $c(x)$, it is unleashed as an 'independent free variable' inside the solver to speed up optimization.**
    
    When these two freed variables ($c(x)$ and $s$) try to tear apart due to linearization errors, the **ruthless mathematical rubber band** that forcefully binds them together is the L1 Penalty term:
    
    $$w_{L1} |c(x_{\text{new}}) + s_{\text{new}}|$$
