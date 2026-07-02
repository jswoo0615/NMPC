# SafeTrajectoryBuffer API Reference

!!! abstract "Overview"
    This document outlines the `SafeTrajectoryBuffer` API, which provides **fail-safe mechanisms** for the NMPC controller. It acts as a continuous safety net, storing the most recently feasible optimization results. If the IPM solver fails to converge or diverges during a control cycle, this buffer is invoked to provide either a warm-start trajectory or a deterministic emergency braking command.

## :material-shield-check: Architecture & Design Philosophy

The `SafeTrajectoryBuffer` handles the critical real-time requirement of autonomous driving: **Never output NaN or unpredictable control commands.**

1. **Continuity**: It guarantees that the controller always has a valid $(X, U)$ trajectory to fall back on.
2. **Emergency Fallback**: If no previous valid trajectory exists, it can inject a safe hard-coded fallback command (e.g., straight steering and max braking).
3. **Deterministic Memory**: It uses `std::array` with `matrix::StaticVector` to avoid dynamic allocation.

## :material-cube-outline: Core Class: `SafeTrajectoryBuffer`

```cpp
template <std::size_t H, std::size_t Nx, std::size_t Nu>
class SafeTrajectoryBuffer;
```

### 1. Template Parameters
- **`H`**: Prediction horizon length.
- **`Nx`**: State dimension.
- **`Nu`**: Control input dimension.

### 2. Key Public Members
=== "Stored Trajectories"
    | Variable | Description |
    | :--- | :--- |
    | `std::array<matrix::StaticVector<double, Nx>, H + 1> X_safe` | The latest feasible state trajectory. |
    | `std::array<matrix::StaticVector<double, Nu>, H> U_safe` | The latest feasible control trajectory. |
    | `bool has_valid_trajectory` | `true` if a feasible trajectory was committed at least once; otherwise `false`. |

### 3. Key Public Methods

#### `commit()`
```cpp
void commit(
    const std::array<matrix::StaticVector<double, Nx>, H + 1>& X,
    const std::array<matrix::StaticVector<double, Nu>, H>& U
);
```
- **Purpose**: Saves a feasible trajectory.
- **Usage**: Called at the end of a control cycle *only if* the IPM/SQP solver converges successfully and satisfies the KKT tolerance.

#### `generate_fallback_control()`
```cpp
matrix::StaticVector<double, Nu> generate_fallback_control(double braking_acceleration) const;
```
- **Purpose**: Generates a deterministic emergency control command.
- **Behavior**:
  - Uses an internal `ControlIndex` enum to guarantee correct vector mappings.
  - Sets **Steering = 0.0** (straight ahead).
  - Sets **Acceleration = `braking_acceleration`** (max braking).
- **Constraints**: Requires `Nu >= 2` and `braking_acceleration <= 0.0`.

## :material-cog-sync: Integration with `SparseNMPC_IPM`

In the NMPC solver, `SafeTrajectoryBuffer` is heavily utilized during the `execute_fallback()` routine.

If the main KKT solver diverges or receives `NaN` inputs, the solver skips its standard update, signals `fallback_triggered = true`, and directly loads the output of `SafeBuffer`. If a valid trajectory already exists in the buffer, it utilizes that trajectory's next steps to smoothly degrade. If the buffer is empty (e.g., failure on the very first frame), it immediately injects `generate_fallback_control()` to forcefully stop the vehicle safely.
