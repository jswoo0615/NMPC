#ifndef OPTIMIZATION_MATRIX_MATH_TRAITS_HPP_
#define OPTIMIZATION_MATRIX_MATH_TRAITS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#if !defined(__CUDACC__)
    #include "Optimization/Matrix/AD/Dual.hpp"
#endif
#include "Optimization/Utils/CUDAMacros.hpp"

namespace Optimization {
    enum class MathStatus {
        SUCCESS = 1,
        SINGULAR = -1,
        ILL_CONDITIONED = -2,
        NUMERICAL_ERROR = -5
    };

    template <typename T>
    struct MathTraits {
        CUDA_CALLABLE static inline T abs(const T& x) noexcept {
            return std::abs(x);
        }
        CUDA_CALLABLE static inline T sqrt(const T& x) noexcept {
            return std::sqrt(x);
        }
        CUDA_CALLABLE static inline T max(const T& x, const T& y) noexcept {
            return std::max(x, y);
        }
        CUDA_CALLABLE static inline T min(const T& x, const T& y) noexcept {
            return std::min(x, y);
        }
        CUDA_CALLABLE static inline bool near_zero(const T& x, T tol = std::numeric_limits<T>::epsilon()) noexcept {
            return std::abs(x) <= tol;
        }
        CUDA_CALLABLE static inline T get_value(const T& x) noexcept {
            return x;
        }
    };

    #if !defined(__CUDACC__)
    template <typename T>
    struct MathTraits<Optimization::Dual<T>> {
        CUDA_CALLABLE static inline Optimization::Dual<T> abs(const Optimization::Dual<T>& x) noexcept {
            return Optimization::ad::abs(x);
        }
        CUDA_CALLABLE static inline Optimization::Dual<T> sqrt(const Optimization::Dual<T>& x) noexcept {
            return Optimization::ad::sqrt(x);
        }
        CUDA_CALLABLE static inline Optimization::Dual<T> max(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) noexcept {
            return (x.v > y.v) ? x : y;
        }
        CUDA_CALLABLE static inline Optimization::Dual<T> min(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) noexcept {
            return (x.v < y.v) ? y : x;
        }
        CUDA_CALLABLE static inline bool near_zero(const Optimization::Dual<T>& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) noexcept {
            return std::abs(x.v) <= tol;
        }
        CUDA_CALLABLE static inline T get_value(const Optimization::Dual<T>& x) noexcept {
            return x.v;
        }
    };

    template <typename T, std::size_t N>
    struct MathTraits<Optimization::DualVec<T, N>> {
        CUDA_CALLABLE static inline Optimization::DualVec<T, N> abs(const Optimization::DualVec<T, N>& x) noexcept {
            return Optimization::ad::abs(x);
        }
        CUDA_CALLABLE static inline Optimization::DualVec<T, N> sqrt(const Optimization::DualVec<T, N>& x) noexcept {
            return Optimization::ad::sqrt(x);
        }
        CUDA_CALLABLE static inline Optimization::DualVec<T, N> max(const Optimization::DualVec<T, N>& x, const Optimization::DualVec<T, N>& y) noexcept {
            return (x.v > y.v) ? x : y;
        }
        CUDA_CALLABLE static inline Optimization::DualVec<T, N> min(const Optimization::DualVec<T, N>& x, const Optimization::DualVec<T, N>& y) noexcept {
            return (x.v < y.v) ? y : x;
        }
        CUDA_CALLABLE static inline bool near_zero(const Optimization::DualVec<T, N>& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) noexcept {
            return std::abs(x.v) <= tol;
        }
        CUDA_CALLABLE static inline T get_value(const Optimization::DualVec<T, N>& x) noexcept {
            return x.v;
        }
    };

    #endif
} // namespace Optimization

#endif // OPTIMIZATION_MATRIX_MATH_TRAITS_HPP_