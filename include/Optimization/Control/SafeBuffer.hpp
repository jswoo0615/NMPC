#ifndef OPTIMIZATION_CONTROL_SAFE_BUFFER_HPP_
#define OPTIMIZATION_CONTROL_SAFE_BUFFER_HPP_

#include <array>
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

namespace Optimization {
namespace controller {

template <size_t H, size_t Nx, size_t Nu>
class SafeBuffer {
public:
    std::array<matrix::StaticVector<double, Nx>, H + 1> X_safe;
    std::array<matrix::StaticVector<double, Nu>, H> U_safe;
    bool has_valid_trajectory = false;

    SafeBuffer() {
        for (size_t k = 0; k <= H; ++k) X_safe[k].set_zero();
        for (size_t k = 0; k < H; ++k) U_safe[k].set_zero();
    }

    void commit(const std::array<matrix::StaticVector<double, Nx>, H + 1>& X, 
                const std::array<matrix::StaticVector<double, Nu>, H>& U) {
        X_safe = X;
        U_safe = U;
        has_valid_trajectory = true;
    }

    matrix::StaticVector<double, Nu> extract_fallback_control(double max_decel) {
        matrix::StaticVector<double, Nu> safe_u;
        safe_u.set_zero();
        safe_u(0) = 0.0; // Steering straight
        safe_u(1) = max_decel; // Max deceleration
        return safe_u;
    }
};

} // namespace controller
} // namespace Optimization

#endif // OPTIMIZATION_CONTROL_SAFE_BUFFER_HPP_
