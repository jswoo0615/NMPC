# Mathematical Note: Static Cubic Spline Derivation

!!! info "Background"
    This document outlines the mathematical modeling, derivative expansion, and tridiagonal matrix formulation for the `StaticCubicSpline` interpolator.
    It serves as a technical reference to understand the exact numerical derivation used in the codebase.

## :material-numeric-1-box: 1. 1D Cubic Spline Basic Definition

In each interval $[x_{i}, x_{i+1}]$, the spline function is discretely defined as a cubic polynomial, taking the start point $x_{i}$ as the origin of the local coordinate system:

!!! math "Polynomial Definition"
    $$S_{i}(x) = a_{i} + b_{i}(x - x_{i}) + c_{i}(x - x_{i})^{2} + d_{i}(x - x_{i})^{3}$$

The coefficients $a_{i}, b_{i}, c_{i}, d_{i}$ are calculated to satisfy the following continuity and boundary conditions:

- **Initial Condition**: $a_{i} = y_{i}$ (The function value at the start of each interval)
- **Interval Spacing**: $h_{i} = x_{i+1} - x_{i}$
- **Natural Spline Boundary Condition**: The curvature (second derivative) at both endpoints is assumed to be zero:
  $$S_{0}^{\prime \prime}(x_{0}) = 0, \quad S_{n-1}^{\prime \prime}(x_{n}) = 0$$

### 1.1. Derivative Expansion

By differentiating the polynomial, the first and second derivatives are derived:

- **First Derivative**: $S_{i}^{\prime}(x) = b_{i} + 2c_{i}(x - x_{i}) + 3d_{i}(x - x_{i})^{2}$
- **Second Derivative**: $S_{i}^{\prime \prime}(x) = 2c_{i} + 6d_{i}(x - x_{i})$

## :material-numeric-2-box: 2. Extension to 2D Cubic Spline

In a 2D space, such as vehicle dynamics control, the $X$ and $Y$ coordinates are interpolated as independent 1D splines based on the accumulated arc length parameter $s$.

<div class="grid cards" markdown>

- :material-map-marker-path: **Position Interpolation**
    $$X(s) = S_{x}(s), \quad Y(s) = S_{y}(s)$$

- :material-compass-outline: **Heading (Yaw) Derivation**
    $$\theta (s) = \text{arctan2}\left(\frac{dY}{ds}, \frac{dX}{ds}\right)$$

- :material-steering: **Curvature Derivation**
    $$\kappa(s) = \frac{\frac{d^{2}Y}{ds^{2}}\frac{dX}{ds} - \frac{d^{2}X}{ds^{2}}\frac{dY}{ds}}{\left(\left(\frac{dX}{ds}\right)^{2} + \left(\frac{dY}{ds}\right)^{2}\right)^{3/2}}$$

</div>

## :material-numeric-3-box: 3. Core Coefficient Derivation & Tridiagonal Matrix

The core of spline interpolation is setting up a system of equations to solve for $c_{i}$ through the constraint that **all intervals must smoothly connect at the boundary points $x_{i+1}$**.

### 3.1. Discrete Interpretation of Continuity Condition

The right endpoint of interval $i$ and the left endpoint of interval $i+1$ are the same node $x_{i+1}$ in the global coordinate system:

- **Right endpoint of interval $i$**: $(x - x_{i}) = h_{i}$ in local coordinates.
- **Left endpoint of interval $i+1$**: $(x - x_{i+1}) = 0$ in its own local coordinates.

Therefore, the continuity condition boils down to: "At the same global node $x_{i+1}$, the state of the end of the left curve $S_{i}$ and the start of the right curve $S_{i+1}$ must match."

### 3.2. System of Equations via Derivative Continuity

=== "Step 1. Second Derivative Continuity"
    - $S_{i}^{\prime \prime}(x_{i+1}) = 2c_{i} + 6d_{i}h_{i}$
    - $S_{i+1}^{\prime \prime}(x_{i+1}) = 2c_{i+1} + 6d_{i+1}(0) = 2c_{i+1}$
    - Equating them: $2c_{i} + 6d_{i}h_{i} = 2c_{i+1} \implies d_{i} = \frac{c_{i+1} - c_{i}}{3h_{i}}$

=== "Step 2. Function Value Continuity"
    - $S_{i}(x_{i+1}) = a_{i} + b_{i}h_{i} + c_{i}h_{i}^{2} + d_{i}h_{i}^{3} = a_{i+1}$
    - Rearranging for $b_{i}$ (substituting $d_{i}$ from Step 1):
      $$b_{i} = \frac{a_{i+1} - a_{i}}{h_{i}} - \frac{h_{i}}{3}(c_{i+1} + 2c_{i})$$

=== "Step 3. First Derivative Continuity & Deriving $\alpha$"
    - $S_{i}^{\prime}(x_{i+1}) = b_{i} + 2c_{i}h_{i} + 3d_{i}h_{i}^{2} = b_{i+1}$
    - By substituting $b_{i}, b_{i+1}, d_{i}$ obtained in Steps 1 and 2 into this equation and sorting in descending order of $c$, the following core relation is derived:
      $$h_{i-1}c_{i-1} + 2(h_{i-1} + h_{i})c_{i} + h_{i}c_{i+1} = \frac{3}{h_{i}}(a_{i+1} - a_{i}) - \frac{3}{h_{i-1}}(a_{i} - a_{i-1})$$

### 3.3. Definition and Meaning of the Auxiliary Term $\alpha$

We define the right side of the equation as $\alpha_{i}$:

!!! math "Alpha Definition"
    $$\alpha_{i} = \frac{3}{h_{i}}(a_{i+1} - a_{i}) - \frac{3}{h_{i-1}}(a_{i} - a_{i-1})$$

**Physical Meaning:**
$\frac{a_{i+1} - a_{i}}{h_{i}}$ is the average slope of the right interval, and $\frac{a_{i} - a_{i-1}}{h_{i-1}}$ is the average slope of the left interval. That is, $\alpha_{i}$ is a scaled value of the **difference in slope change between both intervals**, which quantifies the degree of 'curvature (bending)' required by the spline at that node.

This completes the $N \times N$ Tridiagonal Matrix system with $c_{i}$ as the unknown. It allows deterministic operations with $\mathcal{O}(N)$ time complexity through the Thomas algorithm (forward elimination and backward substitution).

## :material-numeric-4-box: 4. Numerical Verification Example

To verify the concept, we directly calculate the 1D Spline coefficients for 4 data points.

- **Given Points**: $(0, 0), (1, 1), (2, 0), (3, 1)$
- **Initial Setup**: 
  - $h_{0} = 1, h_{1} = 1, h_{2} = 1$
  - $a_{0} = 0, a_{1} = 1, a_{2} = 0, a_{3} = 1$

### 4.1. Calculation of $\alpha$

- $\alpha_{1} = \frac{3}{1}(0 - 1) - \frac{3}{1}(1 - 0) = -3 - 3 = -6$
- $\alpha_{2} = \frac{3}{1}(1 - 0) - \frac{3}{1}(0 - 1) = 3 - (-3) = 6$

### 4.2. Constructing and Solving the Tridiagonal Matrix System

Due to the Natural Spline boundary conditions, $c_{0} = 0$ and $c_{3} = 0$.

- For $i = 1: 1 \cdot 0 + 2(1 + 1)c_{1} + 1 \cdot c_{2} = -6 \implies 4c_{1} + c_{2} = -6$
- For $i = 2: 1 \cdot c_{1} + 2(1 + 1)c_{2} + 1 \cdot 0 = 6 \implies c_{1} + 4c_{2} = 6$

Solving the system of equations:

1. Substitute $c_{2} = -6 - 4c_{1}$ into the second equation:
   $$c_{1} + 4(-6 - 4c_{1}) = 6 \implies c_{1} - 24 - 16c_{1} = 6 \implies -15c_{1} = 30 \implies c_{1} = -2$$
2. $c_{2} = -6 - 4(-2) = 2$
3. Result: $c_{0} = 0, c_{1} = -2, c_{2} = 2, c_{3} = 0$

### 4.3. Calculation of Remaining Coefficients ($b_{i}, d_{i}$)

Using the formulas: $b_{i} = \frac{a_{i+1} - a_{i}}{h_{i}} - \frac{h_{i}}{3}(c_{i+1} + 2c_{i}), \quad d_{i} = \frac{c_{i+1} - c_{i}}{3h_{i}}$

=== "Interval 0: $[0, 1]$"
    - $b_{0} = \frac{1-0}{1} - \frac{1}{3}(-2+0) = 1 + \frac{2}{3} \approx 1.667$
    - $d_{0} = \frac{-2-0}{3} \approx -0.667$

=== "Interval 1: $[1, 2]$"
    - $b_{1} = \frac{0-1}{1} - \frac{1}{3}(2-4) = -1 + \frac{2}{3} \approx -0.333$
    - $d_{1} = \frac{2-(-2)}{3} \approx 1.333$

=== "Interval 2: $[2, 3]$"
    - $b_{2} = \frac{1-0}{1} - \frac{1}{3}(0+4) = 1 - \frac{4}{3} \approx -0.333$
    - $d_{2} = \frac{0-2}{3} \approx -0.667$

### 4.4. Final Polynomial Model

Based on the numerical operations, the final spline polynomials are:

!!! success "Final Spline Equations"
    - $S_{0}(x) = 1.667x - 0.667x^{3} \quad (x \in [0, 1])$
    - $S_{1}(x) = 1 - 0.333(x - 1) - 2(x - 1)^{2} + 1.333(x - 1)^{3} \quad (x \in [1, 2])$
    - $S_{2}(x) = -0.333(x - 2) + 2(x - 2)^{2} - 0.667(x - 2)^{3} \quad (x \in [2, 3])$
