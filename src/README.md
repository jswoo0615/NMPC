# Source Module (`src`)

This directory contains the integration layer that wraps the core C++ NMPC mathematical engine into a high-level language interface. This enables seamless deployment of the solver into modern autonomous driving simulators and middleware frameworks.

## Core Components

### 1. `nmpc_wrapper.cpp`
This file utilizes **PyBind11** to expose the `SparseNMPC_IPM` solver and its related algorithmic utilities (like `StaticCubicSpline2D`) as a native, highly optimized Python module. It is specifically designed to act as the primary bridge between the hard real-time C++ solver and the Python-based CARLA simulator ecosystem.

**Key Features & Engineering Highlights:**
- **Zero-Copy Memory Binding**: By leveraging `pybind11/numpy.h`, the wrapper directly maps Python Numpy arrays (containing ego state, waypoints, and dynamic obstacles) to raw C++ pointers (`double* ego_state_ptr`). This completely bypasses the massive Python-to-C++ data serialization and copying overhead, preserving the solver's microsecond-level execution speeds.
- **Robust Cartesian to Frenet Projection**: Implements a highly resilient `project_vehicle_state` function. It uses a combination of coarse grid search and fine Newton-Raphson optimization to accurately project the vehicle's global $(x, y, \text{yaw})$ coordinates onto the reference spline, extracting the required Frenet $(s, d, \mu)$ states. It incorporates a smart "Cold Start" detection mechanism to intelligently toggle between fast local searches and exhaustive global searches depending on tracking divergence.
- **KKT Proof & Telemetry Extraction**: Unlike generic controllers that only return control commands, this wrapper extracts deep mathematical telemetry from the solver. As noted in the code ("Phase 5 Architect's Fix"), it pulls out the minimum slack margins (`min_slack`), maximum dual multipliers (`max_lam`), and the overall KKT error. This allows the Python supervisory node to actively monitor the physical limits of the vehicle (e.g., tire friction saturation) and the numerical health of the optimization in real-time.
