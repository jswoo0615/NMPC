# Vehicle Dynamics Equations

!!! info "Mathematical Foundation"
    This Technical Note details the exact mathematical formulations implemented within `RealTimeDynamicsModel.hpp`, `HighFidelityDynamicsModel.hpp`, and `VehiclePhysicsCore.hpp`. 

---

## 1. State and Input Definitions

The vehicle state $x \in \mathbb{R}^8$ and control input $u \in \mathbb{R}^2$ are defined as follows:

$$
x = \begin{bmatrix} s \\ d \\ \mu \\ v_x \\ v_y \\ r \\ \alpha_f \\ \alpha_r \end{bmatrix}, \quad
u = \begin{bmatrix} \delta \\ a_{cmd} \end{bmatrix}
$$

| Variable | Description | Unit |
| :---: | :--- | :---: |
| $s$ | Arc length along the reference path | m |
| $d$ | Lateral deviation from the reference path | m |
| $\mu$ | Heading angle error relative to the reference path | rad |
| $v_x$ | Longitudinal velocity (body frame) | m/s |
| $v_y$ | Lateral velocity (body frame) | m/s |
| $r$ | Yaw rate (body frame) | rad/s |
| $\alpha_f$ | Front tire slip angle | rad |
| $\alpha_r$ | Rear tire slip angle | rad |
| $\delta$ | Front steering angle | rad |
| $a_{cmd}$ | Longitudinal acceleration command | $m/s^2$ |

---

## 2. Load Transfer Physics (`VehiclePhysicsCore.hpp`)

Accurate modeling of lateral forces requires estimating the normal load ($F_z$) on each tire.

### 2.1 Quasi-Static Load Transfer (Real-Time Model)
To minimize computational burden in the NMPC loop, the `RealTimeDynamicsModel` uses a static approximation of lateral acceleration ($a_y \approx v_x \cdot r$) to calculate weight transfer without requiring independent roll and pitch integrators.

**Static Weight Distribution:**
$$
F_{z,f}^{stat} = m g \frac{l_r}{L}, \quad F_{z,r}^{stat} = m g \frac{l_f}{L}
$$

**Longitudinal Load Transfer:**
$$
\Delta F_z^{long} = m a_{cmd} \frac{h_{cg}}{L}
$$

**Lateral Load Transfer (Front & Rear):**
$$
\Delta F_{z,lat}^{front} = m (v_x r) \frac{h_{cg}}{t_f} K_{df}
$$
$$
\Delta F_{z,lat}^{rear} = m (v_x r) \frac{h_{cg}}{t_r} (1 - K_{df})
$$

Where $K_{df}$ is the front roll stiffness distribution ratio, and $t_f, t_r$ are the track widths.

### 2.2 Suspension Geometric Load Transfer (High-Fidelity Model)
The `HighFidelityDynamicsModel` actively calculates chassis roll ($\phi$) and pitch ($\theta$) by modeling the roll center heights and suspension spring stiffnesses.

$$
\theta = \frac{m a_{cmd} h_{cg}}{K_{pitch}}
$$
$$
\phi = \frac{m (v_x r) h_{roll\_arm}}{K_{roll} - m g h_{roll\_arm}}
$$

These angles are then used to compute the exact dynamic wheel loads and corresponding camber thrust effects.

---

## 3. Tire Physics & Magic Formula

### 3.1 Steady-State Slip Angle calculation
The steady-state kinematic slip angles are computed using the velocities at the front and rear axles:

$$
\alpha_{ss,f} = \delta - \arctan\left(\frac{v_y + l_f r}{v_x}\right)
$$
$$
\alpha_{ss,r} = - \arctan\left(\frac{v_y - l_r r}{v_x}\right)
$$

### 3.2 Relaxation Length (First-Order Delay)
Tire forces do not develop instantly. To prevent numerical instability and model real-world high-speed dynamics, a relaxation length ($\sigma$) is introduced:

$$
\dot{\alpha}_f = \frac{v_x}{\sigma_f} (\alpha_{ss,f} - \alpha_f)
$$
$$
\dot{\alpha}_r = \frac{v_x}{\sigma_r} (\alpha_{ss,r} - \alpha_r)
$$

### 3.3 Pacejka Magic Formula
The non-linear lateral force $F_y$ is computed using a load-dependent Pacejka Magic Formula.

**Friction Coefficient Decay:**
$$
\mu = \mu_0 - d\mu (F_z - F_{z,nom})
$$

**Cornering Stiffness:**
$$
C_\alpha = c_1 F_z - c_2 F_z^2
$$

**Lateral Force Calculation:**
$$
D = \mu F_z, \quad B = \frac{C_\alpha}{C_{shape} D}
$$
$$
F_y = D \sin(C_{shape} \arctan(B \alpha))
$$

*(In the C++ implementation, standard `sin` and `atan` are substituted with high-speed LUT operations via `FastMath.hpp` when evaluating double precision, accelerating real-time integration.)*

---

## 4. Frenet Kinematics and Equations of Motion

Finally, combining the forces yields the state derivatives. 

Let $\kappa$ be the road curvature at the vehicle's position.

### 4.1 Kinematics (Path Tracking)
$$
\dot{s} = \frac{v_x \cos(\mu) - v_y \sin(\mu)}{1 - d \cdot \kappa}
$$
$$
\dot{d} = v_x \sin(\mu) + v_y \cos(\mu)
$$
$$
\dot{\mu} = r - \kappa \cdot \dot{s}
$$

*(To avoid singularities when driving strictly backward or perfectly on the path, the denominator $(1 - d \cdot \kappa)$ is safely bounded.)*

### 4.2 Dynamics (Body Accelerations)
$$
\dot{v}_x = a_{cmd} - \frac{F_{drag}}{m} + r \cdot v_y
$$
$$
\dot{v}_y = \frac{F_{y,f} \cos(\delta) + F_{y,r}}{m} - r \cdot v_x
$$
$$
\dot{r} = \frac{l_f F_{y,f} \cos(\delta) - l_r F_{y,r}}{I_z}
$$

These equations natively form the $\dot{x} = f(x, u)$ vector, which is fed into the Runge-Kutta 4 (RK4) integrator inside the `SparseNMPC_IPM` solver.
