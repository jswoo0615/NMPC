#ifndef OPTIMIZATION_DYNAMICS_FAST_MATH_HPP_
#define OPTIMIZATION_DYNAMICS_FAST_MATH_HPP_

#include <algorithm>
#include <cmath>
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Utils/CUDAMacros.hpp"

namespace Optimization {
namespace FastMath {

constexpr size_t LUT_SIZE = 1024;
constexpr double PI = 3.14159265358979323846;

class FastTrig {
private:
    double sin_lut[LUT_SIZE];
    double atan_lut[LUT_SIZE];
    const double sin_min = -PI;
    const double sin_max = PI;
    const double atan_min = -10.0;
    const double atan_max = 10.0;

    FastTrig() {
        for (size_t i = 0; i < LUT_SIZE; ++i) {
            double x_sin = sin_min + (sin_max - sin_min) * i / (LUT_SIZE - 1.0);
            sin_lut[i] = std::sin(x_sin);

            double x_atan = atan_min + (atan_max - atan_min) * i / (LUT_SIZE - 1.0);
            atan_lut[i] = std::atan(x_atan);
        }
    }
public:
    static FastTrig& getInstance() {
        static FastTrig instance;
        return instance;
    }

    CUDA_CALLABLE inline double fast_sin(double x) const {
        while (x > PI) x -= 2.0 * PI;
        while (x < -PI) x += 2.0 * PI;
        
        double norm = (x - sin_min) / (sin_max - sin_min);
        double idx_exact = norm * (LUT_SIZE - 1.0);
        int idx = static_cast<int>(idx_exact);
        if (idx < 0) idx = 0;
        if (idx >= LUT_SIZE - 1) return sin_lut[LUT_SIZE - 1];
        
        double frac = idx_exact - idx;
        return sin_lut[idx] + frac * (sin_lut[idx+1] - sin_lut[idx]);
    }

    CUDA_CALLABLE inline double fast_atan(double x) const {
        if (x > atan_max) return PI / 2.0;
        if (x < atan_min) return -PI / 2.0;

        double norm = (x - atan_min) / (atan_max - atan_min);
        double idx_exact = norm * (LUT_SIZE - 1.0);
        int idx = static_cast<int>(idx_exact);
        if (idx < 0) idx = 0;
        if (idx >= LUT_SIZE - 1) return atan_lut[LUT_SIZE - 1];

        double frac = idx_exact - idx;
        return atan_lut[idx] + frac * (atan_lut[idx+1] - atan_lut[idx]);
    }
};

template <typename T>
CUDA_CALLABLE inline T fast_sin(const T& x) { 
    using std::sin;
    using Optimization::matrix::ad::sin;
    return sin(x); 
} 

template <>
CUDA_CALLABLE inline double fast_sin(const double& x) { return FastTrig::getInstance().fast_sin(x); }

template <typename T>
CUDA_CALLABLE inline T math_sin(const T& x) {
    return fast_sin(x);
}

template <typename T>
CUDA_CALLABLE inline T fast_atan(const T& x) { 
    using std::atan;
    using Optimization::matrix::ad::atan;
    return atan(x); 
} 

template <>
CUDA_CALLABLE inline double fast_atan(const double& x) { return FastTrig::getInstance().fast_atan(x); }

template <typename T>
CUDA_CALLABLE inline T math_cos(const T& x) {
    using std::cos;
    using Optimization::matrix::ad::cos;
    return cos(x);
}

template <typename T>
CUDA_CALLABLE inline T math_atan2(const T& y, const T& x) {
    using std::atan2;
    using Optimization::matrix::ad::atan2;
    return atan2(y, x);
}

} // namespace FastMath
} // namespace Optimization

#endif // OPTIMIZATION_DYNAMICS_FAST_MATH_HPP_
