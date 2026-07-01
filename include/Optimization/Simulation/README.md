# Simulation Module

This directory contains the numerical integration engines required to discretize continuous-time vehicle dynamics models (ODEs) over finite time steps.

## Core Components

### 1. `Integrator.hpp`
Implements a high-precision, fixed-step **Runge-Kutta 4th Order (RK4)** numerical integrator. It serves as the primary physics engine driving both the NMPC predictions and the Estimator's state propagations.

**Key Features & Engineering Highlights:**
- **Strict Zero-Allocation**: Built entirely using `matrix::StaticVector`. The internal evaluations of the four Runge-Kutta slopes ($k_1, k_2, k_3, k_4$) are performed on the stack, guaranteeing that the integration process never triggers dynamic memory allocation or garbage collection pauses.
- **Template-Driven Polymorphism**: The `step_rk4` algorithm and the `IntegratorEngine` wrapper are fully templated on the underlying numeric type `T`. This is a critical architectural decision that enables a powerful dual-use mechanism:
  - **Forward Simulation**: When instantiated with `T = double`, it acts as an extremely fast forward-simulator for the Extended Kalman Filter (`EKF.hpp`), Moving Horizon Estimator (`SparseMHE.hpp`), and the Plant model.
  - **Exact Automatic Differentiation (AD)**: When instantiated with `T = Dual` or `T = DualVec` (from the `AD` module), it seamlessly propagates the dual number chain-rule through the entire RK4 integration mathematics. This uniquely empowers the NMPC solver (`SparseNMPC_IPM.hpp`) to automatically extract exact, continuous-time Jacobian matrices ($A$ and $B$) without requiring engineers to manually derive or code complex partial derivatives.
