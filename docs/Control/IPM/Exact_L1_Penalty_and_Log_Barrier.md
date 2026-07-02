# Exact L1 Penalty and Log-Barrier

!!! abstract "Overview"
    This document explains how the scalar `merit` function in the NMPC Line Search process balances tracking performance, control effort, and physical constraints (e.g., obstacle avoidance), and specifically how it **strictly enforces physical constraints**.

## :material-scale-balance: The Role of `L1_WEIGHT` (Exact Penalty)

### Common Misconception: "Does it only penalize violations?"

When dealing with a constraint $c(x) \le 0$ (e.g., safe distance violation), it is common to think that a penalty is only applied when the value is positive (i.e., a violation). However, the L1 penalty used in the Merit function operates entirely differently.

### Enforcing Strict Consistency

To solve inequality constraints, the IPM solver introduces a virtual positive slack variable $s \ge 0$, artificially converting the constraint into an equality constraint.

!!! math "Equality Constraint with Slack"
    $$
    c(x) + s = 0
    $$

The L1 penalty term included in the Merit function is:

!!! math "L1 Penalty Term"
    $$
    w_{L1} |c + s + \alpha ds|
    $$

This absolute value $(|\cdot|)$ term verifies **"Does the physical reality ($c$) perfectly align with the solver's internal virtual variable ($s$)?"**, regardless of whether the constraint is violated or not.

- Even if the vehicle is perfectly safe, being 5m away from an obstacle ($c = -5$), if the solver's internal slack variable was calculated as $s = 3$, it results in $|-5 + 3| = |-2| = 2$. This incurs a massive penalty of $2000 \ (w_{L1} \times 2)$.
- In other words, this term acts as a **strict consistency mechanism that does not tolerate even a 1mm discrepancy between the physical state and the mathematical state**.

## :material-puzzle-outline: Perfect Division of Labor (Log-Barrier vs L1 Penalty)

This system combines two extreme mechanisms to achieve absolute robustness.

<div class="grid cards" markdown>

- :material-chart-bell-curve:
    **Log-Barrier** (Long-term Direction / Soft Constraint)
    
    $$-\mu \log(s + \alpha ds)$$
    
    - **Mechanism**: Imposes an infinite ($\infty$) penalty if the internal slack variable $s$ approaches zero.
    - **Purpose**: Establishes the long-term goal of "staying within the feasible region". It gently but firmly pushes the trajectory to remain safely in the interior ($s > 0$) without touching the boundaries.

- :material-hammer:
    **L1 Penalty** (Immediate Control / Hard Constraint)
    
    $$w_{L1} |c + s + \alpha ds|$$ (where $w_{L1} = 1000.0$)
    
    - **Mechanism**: Instantly applies a massive linear penalty the moment the sum of physical reality ($c$) and the internal variable ($s$) deviates from zero.
    - **Purpose**: If the solver becomes blinded by performance (reducing cost function) and attempts to take a step ($\alpha$) that ignores constraints during the Line Search, this term blows up the `merit` value with a ruthless weight, acting as a **physical brake that instantly rejects the step**.

</div>

## :material-check-decagram: Why an "Exact" Weight?

!!! success "Inexact Penalty vs Exact Penalty"
    If a **standard quadratic penalty** ($c^2$) is used, no matter how large the weight is, the optimization solver will inevitably make a trade-off, "slightly violating" the constraint to further reduce the cost function. (Inexact Penalty)

    However, by using an **L1 penalty** ($|c|$) and setting its weight ($w_{L1}$) sufficiently larger than the maximum dual variable of the system ($\lambda_{\max}$, e.g., $1000.0$), it is mathematically proven to yield an **"Exact Optimal Solution"** where the constraint is not violated by even 1mm, regardless of how much the cost function is sacrificed.
