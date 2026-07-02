# Dynamics API Reference

!!! info "API Documentation"
    This document provides a deep dive into the specific API signatures and data structures used in the Vehicle Dynamics module.

## :material-cube-outline: `RealTimeDynamicsModel`

```cpp
class RealTimeDynamicsModel;
```
Optimized 8-DoF Frenet Bicycle Model tailored for the Interior-Point NMPC solver. It intentionally drops suspension kinematics to achieve extremely fast computation times while preserving vital non-linear tire characteristics.

### Method: `operator()`
```cpp
template <typename T>
matrix::StaticVector<T, 8> operator()(const matrix::StaticVector<T, 8>& x, const matrix::StaticVector<T, 2>& u) const;
```
Calculates the state derivative $\dot{x} = f(x, u)$.
Compatible with Automatic Differentiation when `T` is instantiated as `Dual<double>`.

---

## :material-cube-outline: `HighFidelityDynamicsModel`

```cpp
class HighFidelityDynamicsModel;
```
A highly accurate vehicle model designed to act as the simulation plant.

### Method: `extractJacobians`
```cpp
void extractJacobians(const matrix::StaticVector<double, 8>& x0, 
                      const matrix::StaticVector<double, 2>& u0,
                      std::array<std::array<double, 8>, 8>& A,
                      std::array<std::array<double, 2>, 8>& B) const;
```
Uses Auto-Diff (Dual numbers) to continuously extract exact analytic Jacobians ($A$ and $B$ matrices) at the current operating point `(x0, u0)`. Extremely useful for checking local linear stability.

---

## :material-math-integral: `VehiclePhysicsCore.hpp` Functions

### `computeQuasiStaticLoadTransfer()`
```cpp
template <typename T>
FourWheelLoads<T> computeQuasiStaticLoadTransfer(
    const VehicleDynamicsParams<T>& params, const T v_x, const T yaw_rate, const T a_cmd);
```
Calculates normal tire loads using rigid-body static approximations.

### `computeSuspensionLoadTransfer()`
```cpp
template <typename T>
FourWheelLoads<T> computeSuspensionLoadTransfer(
    const VehicleDynamicsParams<T>& veh_params, const SuspensionParams<T>& susp_params,
    const T v_x, const T yaw_rate, const T a_cmd, ChassisAttitude<T>& out_attitude);
```
Calculates 4-wheel loads by simulating suspension springs, pitch angles, roll angles, and roll center kinematics.

### `computeBicycleLateralForces()`
```cpp
template <typename T>
BicycleLateralForces<T> computeBicycleLateralForces(
    const TireLoadParams<T>& tire_params, const FourWheelLoads<T>& loads, 
    const T alpha_f, const T alpha_r);
```
Computes equivalent front and rear axle lateral forces by individually evaluating the Pacejka Magic Formula for all 4 tires and summing them.

---

## :material-flash: `FastMath.hpp`

### `FastTrig` Class
A singleton class `FastTrig::getInstance()` that computes `sin` and `atan` using pre-computed 1024-element lookup tables (LUT).

### Template Routing
```cpp
template <typename T>
inline T math_sin(const T& x);
```
- If `T` is `double`, it redirects to `FastTrig` for immense speedup.
- If `T` is `Dual<double>`, it redirects to `std::sin()` or `ad::sin()` to guarantee that the Automatic Differentiation chain evaluates the exact analytic derivative without LUT discontinuities.
