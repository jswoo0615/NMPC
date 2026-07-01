# Evaluator Module

This directory contains algorithmic utilities used to evaluate and interpolate mathematical geometries required by the NMPC controller. It strictly adheres to the project's global policy of zero dynamic memory allocation.

## Core Components

### 1. `StaticCubicSpline.hpp`
Implements high-speed, strictly allocation-free 1D and 2D Cubic Spline interpolation algorithms. This module is essential for processing discrete, raw global waypoints (e.g., from a high-level router) into perfectly smooth, continuous, and continuously differentiable ($C^2$) reference paths for the NMPC.

**Key Features & Engineering Highlights:**
- **Zero-Allocation (`StaticCubicSpline1D` / `StaticCubicSpline2D`)**: Instead of relying on standard `std::vector` structures like most open-source spline libraries, these classes are built entirely on compile-time `std::array<double, MaxPoints>`. This guarantees that constructing and evaluating long reference paths will **never** trigger a dynamic memory allocation (`malloc`), completely preventing memory fragmentation and latency spikes in the real-time control thread.
- **$O(\log N)$ Evaluation**: Uses binary search (`search_index`) to rapidly locate the active cubic polynomial segment for a given query parameter (e.g., arc length $s$). This ensures highly deterministic query speeds during the tight NMPC Integration loops.
- **Analytical Derivatives**:
  - `calc_d1(t)` & `calc_d2(t)`: Provides exact analytical 1st and 2nd derivatives of the spline polynomials.
  - `calc_curvature(t)`: Analytically calculates the geometric path curvature ($\kappa$) using the evaluated 1st and 2nd derivatives. This smooth curvature profile is passed directly into the vehicle dynamics model (`RealTimeDynamicsModel`) as a critical feed-forward tracking reference, preventing steering chattering.
