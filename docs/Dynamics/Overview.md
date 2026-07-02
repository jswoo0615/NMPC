# Vehicle Dynamics Module

!!! abstract "Overview"
    This module contains the physical vehicle models and mathematical utilities utilized by both the real-time NMPC solver and the high-fidelity simulator.

## :material-shield-check: Design Philosophy

The dynamics module strictly separates the physical equations of motion from the numerical solver logic. By heavily utilizing C++ templates, a single unified set of physics equations can be compiled for both standard scalar arithmetic (for forward simulation) and Automatic Differentiation (for the NMPC's exact Jacobian and Hessian generation). All functions are marked `CUDA_CALLABLE` to support future heterogeneous (GPU/CPU) compute architectures.

## :material-layers: Core Components

### 1. `VehicleParams.hpp`
Defines the core, strictly-typed data structures for vehicle parameters.

- **`VehicleDynamicsParams`**: Fundamental vehicle properties (mass, yaw inertia, dimensions, CG height).
- **`TireLoadParams`**: Pacejka Magic Formula tire coefficients, including load-dependent friction decay.
- **`SuspensionParams`**: Suspension geometry parameters (roll center heights, camber gain, stiffness) for high-fidelity simulation.

### 2. `FastMath.hpp`
A high-performance mathematical utility module tailored for real-time execution.

- Implements a highly optimized, Cache-friendly Look-Up Table (LUT) based `FastTrig` class for `sin` and `atan` approximations.
- Uses C++ template specialization to smartly route math operations: 
  - Standard `double` evaluations utilize the fast LUT approximations to significantly speed up real-time forward integration.
  - `Dual` and `DualVec` Automatic Differentiation types safely fall back to exact standard library or AD-specific math implementations, guaranteeing that the solver's derivative chains remain 100% exact and uncorrupted by LUT quantization errors.

### 3. `VehiclePhysicsCore.hpp`
Contains the foundational physical calculations decoupled from the state-space ordinary differential equations (ODEs).

- **Load Transfer**: Implements both computationally lightweight quasi-static load transfer (for the real-time NMPC) and comprehensive suspension geometric load transfer (considering pitch, roll, and roll center heights).
- **Tire Physics**: Computes lateral forces using a fully non-linear, load-dependent Pacejka Magic Formula. It includes dynamic camber thrust calculations driven by chassis roll.

### 4. `RealTimeDynamicsModel.hpp`
An 8-state Frenet frame vehicle model meticulously optimized for the real-time NMPC solver.

- Uses quasi-static load transfer to minimize computational overhead within the SQP loop.
- Models the delay in tire force generation using a first-order relaxation length ODE, crucial for capturing transient stability limits at high speeds.
- Fully compatible with the `AD` module, allowing the NMPC solver to automatically extract exact, closed-form Jacobians via the RK4 integrator.

### 5. `HighFidelityDynamicsModel.hpp`
A comprehensive 8-state Frenet frame model designed for robust simulation and algorithm validation (Plant Model).

- Incorporates full 4-wheel independent load transfer, suspension roll/pitch geometry, aerodynamic drag, and dynamic camber thrust.
- Includes a dedicated `extractJacobians` function. By manually instantiating the `Dual` type, it allows engineers to directly extract continuous-time $A$ and $B$ state-space matrices. This is highly useful for classical linear stability analysis, eigenvalue inspection, or LQR bench-marking.
