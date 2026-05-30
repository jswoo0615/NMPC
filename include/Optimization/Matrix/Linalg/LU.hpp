#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    #include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace Optimization {
    namespace linalg {
        template <typename MatA, typename MatLU>
        inline MathStatus lu_decompose(const MatA& A, MatLU& LU, std::array<int, MatA::NumRows>& p) noexcept {
            using T = typename std::remove_const<typename MatA::value_type>::type;
            constexpr std::size_t N = MatA::NumRows;
            static_assert(MatA::NumRows == MatA::NumCols, "LU requires square matrix");

            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t i = 0; i < N; ++i) {
                    LU(i, j) = A(i, j);
                }
            }

            for (std::size_t i = 0; i < N; ++i) {
                p[i] = static_cast<int>(i);
            }

            for (std::size_t k = 0; k < N; ++k) {
                // Partial Pivoting
                T max_val = 0.0;
                std::size_t pivot_idx = k;
                for (std::size_t i = k; i < N; ++i) {
                    T abs_val = std::abs(LU(i, k));
                    if (abs_val > max_val) {
                        max_val = abs_val;
                        pivot_idx = i;
                    }
                }

                if (max_val <= std::numeric_limits<T>::epsilon()) {
                    return MathStatus::SINGULAR;
                }

                if (pivot_idx != k) {
                    std::swap(p[k], p[pivot_idx]);
                    for (std::size_t j = 0; j < N; ++j) {
                        T temp = LU(k, j);
                        LU(k, j) = LU(pivot_idx, j);
                        LU(pivot_idx, j) = temp;
                    }
                }

                T inv_pivot = static_cast<T>(1.0) / LU(k, k);
                for (std::size_t i = k + 1; i < N; ++i) {
                    LU(i, k) *= inv_pivot;
                }
                for (std::size_t j = k + 1; j < N; ++j) {
                    T u_kj = LU(k, j);
                    std::size_t i = k +1;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_ukj = vdupq_n_f32(u_kj);
                            for (; i + 3 < N; i += 4) {
                                float32x4_t v_lik = vld1q_f32(&LU(i, k));
                                float32x4_t v_aij = vld1q_f32(&LU(i, j));
                                vst1q_f32(&LU(i, j), vmlsq_f32(v_aij, v_lik, v_ukj));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_ukj = vdupq_n_f64(u_kj);
                            for (; i + 1 < N; i += 2) {
                                float64x2_t v_lik = vld1q_f64(&LU(i, k));
                                float64x2_t v_aij = vld1q_f64(&LU(i, j));
                                vst1q_f64(&LU(i, j), vmlsq_f64(v_aij, v_lik, v_ukj));                           
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_ukj = _mm256_set1_ps(u_kj);
                            for (; i + 7 < N; i += 8) {
                                __m256 v_lik = _mm256_loadu_ps(&LU(i, k));
                                __m256 v_aij = _mm256_loadu_ps(&LU(i, j));
                                _mm256_storeu_ps(&LU(i, j), _mm256_fnmadd_ps(v_lik, v_ukj, v_aij));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_ukj = _mm256_set1_pd(u_kj);
                            for (; i + 3 < N; i += 4) {
                                __m256d v_lik = _mm256_loadu_pd(&LU(i, k));
                                __m256d v_aij = _mm256_loadu_pd(&LU(i, j));
                                _mm256_storeu_pd(&LU(i, j), _mm256_fnmadd_pd(v_lik, v_ukj, v_aij));
                            }
                        }
                    #endif

                    for (; i < N; ++i) {
                        LU(i, j) -= LU(i, k) * u_kj;
                    }
                }
            }
            return MathStatus::SUCCESS;
        }

        /**
         * @brief 스택 요동 방지 및 SIMD 가속 In-place Solver
         */
        template <typename MatLU, typename VecB, typename VecX>
        inline void lu_solve(const MatLU& LU, const std::array<int, MatLU::NumRows>& p, const VecB& b, VecX& x) noexcept {
            using T = typename std::remove_const<typename MatLU::value_type>::type;
            constexpr std::size_t N = MatLU::NumRows;

            // 임시 벡터 y를 제거하고 결과 벡터 x에 피벗 적용 결과를 바로 복사 (Zero-Allocation)
            for (std::size_t i = 0; i < N; ++i) {
                x(i) = b(p[i]);
            }

            // 1. Forward substitution (L * y = b_perm) -> Memory Contiguous SAXPY
            // L 행렬의 대각 성분은 1.0으로 간주
            for (std::size_t k = 0; k < N; ++k) {
                T x_k = x(k);
                std::size_t i = k + 1;
                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_xk = vdupq_n_f32(x_k);
                        for (; i + 3 < N; i += 4) {
                            float32x4_t v_L = vld1q_f32(&LU(i, k));
                            float32x4_t v_x = vld1q_f32(&x(i));
                            vst1q_f32(&x(i), vmlsq_f32(v_x, v_L, v_xk));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_xk = vdupq_n_f64(x_k);
                        for (; i + 1 < N; i += 2) {
                            float64x2_t v_L = vld1q_f64(&LU(i, k));
                            float64x2_t v_x = vld1q_f64(&x(i));
                            vst1q_f64(&x(i), vmlsq_f64(v_x, v_L, v_xk));
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_xk = _mm256_set1_ps(x_k);
                        for (; i + 7 < N; i += 8) {
                            __m256 v_L = _mm256_loadu_ps(&LU(i, k));
                            __m256 v_x = _mm256_loadu_ps(&x(i));
                            _mm256_storeu_ps(&x(i), _mm256_fnmadd_ps(v_L, v_xk, v_x));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_xk = _mm256_set1_pd(x_k);
                        for (; i + 3 < N; i += 4) {
                            __m256d v_L = _mm256_loadu_pd(&LU(i, k));
                            __m256d v_x = _mm256_loadu_pd(&x(i));
                            _mm256_storeu_pd(&x(i), _mm256_fnmadd_pd(v_L, v_xk, v_x));
                        }
                    }
                #endif
                for (; i < N; ++i) {
                    x(i) -= LU(i, k) * x_k;
                }
            }

            // 2. Backward substitution (U * x = y) -> Memory Contiguous SAXPY
            for (int k = static_cast<int>(N) - 1; k >= 0; --k) {
                x(k) /= LU(k, k);
                T x_k = x(k);
                std::size_t i = 0;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_xk = vdupq_n_f32(x_k);
                        for (; i + 3 < k; i += 4) {
                            float32x4_t v_U = vld1q_f32(&LU(i, k));
                            float32x4_t v_x = vld1q_f32(&x(i));
                            vst1q_f32(&x(i), vmlsq_f32(v_x, v_U, v_xk));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_xk = vdupq_n_f64(x_k);
                        for (; i + 1 < k; i += 2) {
                            float64x2_t v_U = vld1q_f64(&LU(i, k));
                            float64x2_t v_x = vld1q_f64(&x(i));
                            vst1q_f64(&x(i), vmlsq_f64(v_x, v_U, v_xk));
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_xk = _mm256_set1_ps(x_k);
                        for (; i + 7 < k; i += 8) {
                            __m256 v_U = _mm256_loadu_ps(&LU(i, k));
                            __m256 v_x = _mm256_loadu_ps(&x(i));
                            _mm256_storeu_ps(&x(i), _mm256_fnmadd_ps(v_U, v_xk, v_x));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_xk = _mm256_set1_pd(x_k);
                        for (; i + 3 < k; i += 4) {
                            __m256d v_U = _mm256_loadu_pd(&LU(i, k));
                            __m256d v_x = _mm256_loadu_pd(&x(i));
                            _mm256_storeu_pd(&x(i), _mm256_fnmadd_pd(v_U, v_xk, v_x));
                        }
                    }
                #endif

                for (; i < k; ++i) {
                    x(i) -= LU(i, k) * x_k;
                }
            }
        }
    } // namespace linalg
} // namespace Optimization
 
#endif // OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_