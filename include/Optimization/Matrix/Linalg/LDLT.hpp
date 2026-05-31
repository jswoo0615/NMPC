#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_

#include <cmath>
#include <limits>
#include <type_traits>

#include "Optimization/Matrix/Linalg/Core.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Optimization {
namespace linalg {

template <class MatType>
inline MathStatus LDLT_decompose(MatType& mat) noexcept {
    using T = typename std::remove_const<typename MatType::value_type>::type;
    constexpr std::size_t N = MatType::NumRows;
    static_assert(MatType::NumRows == MatType::NumCols, "LDLT requires square matrix");

    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < j; ++k) {
            T temp = mat(j, k) * mat(k, k);
            std::size_t i = j;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t v_temp = vdupq_n_f32(temp);
                for (; i + 3 < N; i += 4) {
                    float32x4_t v_mik = vld1q_f32(&mat(i, k));
                    float32x4_t v_mij = vld1q_f32(&mat(i, j));
                    vst1q_f32(&mat(i, j), vmlsq_f32(v_mij, v_mik, v_temp));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                float64x2_t v_temp = vdupq_n_f64(temp);
                for (; i + 1 < N; i += 2) {
                    float64x2_t v_mik = vld1q_f64(&mat(i, k));
                    float64x2_t v_mij = vld1q_f64(&mat(i, j));
                    vst1q_f64(&mat(i, j), vmlsq_f64(v_mij, v_mik, v_temp));
                }
            }
#elif !defined(__CUDACC__) && defined(__AVX2__)
            if constexpr (std::is_same_v<T, float>) {
                __m256 v_temp = _mm256_set1_ps(temp);
                for (; i + 7 < N; i += 8) {
                    __m256 v_mik = _mm256_loadu_ps(&mat(i, k));
                    __m256 v_mij = _mm256_loadu_ps(&mat(i, j));
                    _mm256_storeu_ps(&mat(i, j), _mm256_fnmadd_ps(v_mik, v_temp, v_mij));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d v_temp = _mm256_set1_pd(temp);
                for (; i + 3 < N; i += 4) {
                    __m256d v_mik = _mm256_loadu_pd(&mat(i, k));
                    __m256d v_mij = _mm256_loadu_pd(&mat(i, j));
                    _mm256_storeu_pd(&mat(i, j), _mm256_fnmadd_pd(v_mik, v_temp, v_mij));
                }
            }
#endif
            for (; i < N; ++i) mat(i, j) -= mat(i, k) * temp;
        }

        T d_jj = mat(j, j);
        // [Architect's Fix] nan 침투 및 극소 피벗(Pivot)에 의한 행렬 폭주(Explosion) 방지
        if (std::isnan(d_jj) || std::abs(d_jj) <= static_cast<T>(1e-9)) {
            return MathStatus::SINGULAR;
        }

        T inv_D = static_cast<T>(1.0) / d_jj;
        for (std::size_t i = j + 1; i < N; ++i) mat(i, j) *= inv_D;
    }
    return MathStatus::SUCCESS;
}

/**
 * @brief [Architect's Update] 스칼라 강등 해결 (SIMD-Accelerated Solver)
 */
template <class MatType, class VecB, class VecX>
inline void LDLT_solve(const MatType& mat, const VecB& b, VecX& x) noexcept {
    using T = typename std::remove_const<typename MatType::value_type>::type;
    constexpr std::size_t N = MatType::NumRows;

    for (std::size_t i = 0; i < N; ++i) x(i) = b(i);

    // 1. Forward substitution (L * y = b)
    for (std::size_t k = 0; k < N; ++k) {
        T x_k = x(k);
        std::size_t i = k + 1;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_xk = vdupq_n_f32(x_k);
            for (; i + 3 < N; i += 4) {
                float32x4_t v_mat = vld1q_f32(&mat(i, k));
                float32x4_t v_x = vld1q_f32(&x(i));
                vst1q_f32(&x(i), vmlsq_f32(v_x, v_mat, v_xk));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            float64x2_t v_xk = vdupq_n_f64(x_k);
            for (; i + 1 < N; i += 2) {
                float64x2_t v_mat = vld1q_f64(&mat(i, k));
                float64x2_t v_x = vld1q_f64(&x(i));
                vst1q_f64(&x(i), vmlsq_f64(v_x, v_mat, v_xk));
            }
        }
#elif !defined(__CUDACC__) && defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_xk = _mm256_set1_ps(x_k);
            for (; i + 7 < N; i += 8) {
                __m256 v_mat = _mm256_loadu_ps(&mat(i, k));
                __m256 v_x = _mm256_loadu_ps(&x(i));
                _mm256_storeu_ps(&x(i), _mm256_fnmadd_ps(v_mat, v_xk, v_x));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d v_xk = _mm256_set1_pd(x_k);
            for (; i + 3 < N; i += 4) {
                __m256d v_mat = _mm256_loadu_pd(&mat(i, k));
                __m256d v_x = _mm256_loadu_pd(&x(i));
                _mm256_storeu_pd(&x(i), _mm256_fnmadd_pd(v_mat, v_xk, v_x));
            }
        }
#endif
        for (; i < N; ++i) x(i) -= mat(i, k) * x_k;
    }

    // 2. Diagonal scaling (D * z = y)
    for (std::size_t i = 0; i < N; ++i) x(i) /= mat(i, i);

    // 3. Backward substitution (L^T * x = z)
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0.0);
        std::size_t k = i + 1;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_sum = vdupq_n_f32(0.0f);
            for (; k + 3 < N; k += 4) {
                float32x4_t v_mat = vld1q_f32(&mat(k, i));
                float32x4_t v_x = vld1q_f32(&x(k));
                v_sum = vfmaq_f32(v_sum, v_mat, v_x);
            }
#if defined(__aarch64__)
            sum += vaddvq_f32(v_sum);
#else
            sum += vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + 
                   vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
#endif
        } else if constexpr (std::is_same_v<T, double>) {
            float64x2_t v_sum = vdupq_n_f64(0.0);
            for (; k + 1 < N; k += 2) {
                float64x2_t v_mat = vld1q_f64(&mat(k, i));
                float64x2_t v_x = vld1q_f64(&x(k));
                v_sum = vfmaq_f64(v_sum, v_mat, v_x);
            }
#if defined(__aarch64__)
            sum += vaddvq_f64(v_sum);
#else
            sum += vgetq_lane_f64(v_sum, 0) + vgetq_lane_f64(v_sum, 1);
#endif
        }
#elif !defined(__CUDACC__) && defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_sum = _mm256_setzero_ps();
            for (; k + 7 < N; k += 8) {
                __m256 v_mat = _mm256_loadu_ps(&mat(k, i));
                __m256 v_x = _mm256_loadu_ps(&x(k));
                v_sum = _mm256_fmadd_ps(v_mat, v_x, v_sum);
            }
            __m128 v_low = _mm256_castps256_ps128(v_sum);
            __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
            v_low = _mm_add_ps(v_low, v_high);
            v_low = _mm_hadd_ps(v_low, v_low);
            v_low = _mm_hadd_ps(v_low, v_low);
            sum += _mm_cvtss_f32(v_low);
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d v_sum = _mm256_setzero_pd();
            for (; k + 3 < N; k += 4) {
                __m256d v_mat = _mm256_loadu_pd(&mat(k, i));
                __m256d v_x = _mm256_loadu_pd(&x(k));
                v_sum = _mm256_fmadd_pd(v_mat, v_x, v_sum);
            }
            __m128d v_low = _mm256_castpd256_pd128(v_sum);
            __m128d v_high = _mm256_extractf128_pd(v_sum, 1);
            v_low = _mm_add_pd(v_low, v_high);
            v_low = _mm_hadd_pd(v_low, v_low);
            sum += _mm_cvtsd_f64(v_low);
        }
#endif
        for (; k < N; ++k) sum += mat(k, i) * x(k);
        x(i) -= sum;
    }
}

}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_