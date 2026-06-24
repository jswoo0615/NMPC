#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_

#include <cmath>
#include <limits>
#include <type_traits>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Core/StaticMatrixView.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
    #include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace Optimization {
    namespace linalg {
        template <typename MatA, typename MatL>
        inline MathStatus cholesky_decompose(const MatA& A, MatL& L) noexcept {
            using T = std::remove_const_t<typename MatA::value_type>;
            constexpr std::size_t N = MatA::NumRows;
            static_assert(MatA::NumRows == MatA::NumCols, "Cholesky requires square matrix");

            L.set_zero();

            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t i = j; i < N; ++i) {
                    L(i, j) = A(i, j);
                }
            }

            for (std::size_t k = 0; k < N; ++k) {
                T l_kk = L(k, k);

                if (l_kk <= std::numeric_limits<T>::epsilon()) {
                    return MathStatus::SINGULAR;
                }
                l_kk = std::sqrt(l_kk);

                L(k, k) = l_kk;
                T inv_l_kk = static_cast<T>(1.0) / l_kk;

                for (std::size_t i = k + 1; i < N; ++i) {
                    L(i, k) *= inv_l_kk;
                }
                for (std::size_t j = k + 1; j < N; ++j) {
                    T l_jk = L(j, k);
                    std::size_t i = j;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_ljk = vdupq_n_f32(l_jk);
                            for (; i + 3 < N; i += 4) {
                                float32x4_t v_lik = vld1q_f32(&L(i, k));
                                float32x4_t v_lij = vld1q_f32(&L(i, j));
                                vst1q_f32(&L(i, j), vmlsq_f32(v_lij, v_lik, v_ljk));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_ljk = vdupq_n_f64(l_jk);
                            for (; i + 1 < N; i += 2) {
                                float64x2_t v_lik = vld1q_f64(&L(i, k));
                                float64x2_t v_lij = vld1q_f64(&L(i, j));
                                vst1q_f64(&L(i, j), vmlsq_f64(v_lij, v_lik, v_ljk));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_ljk = _mm256_set1_ps(l_jk);
                            for (; i + 7 < N; i += 8) {
                                __m256 v_lik = _mm256_loadu_ps(&L(i, k));
                                __m256 v_lij = _mm256_loadu_ps(&L(i, j));
                                _mm256_storeu_ps(&L(i, j), _mm256_fnmadd_ps(v_lik, v_ljk, v_lij));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_ljk = _mm256_set1_pd(l_jk);
                            for (; i + 3 < N; i += 4) {
                                __m256d v_lik = _mm256_loadu_pd(&L(i, k));
                                __m256d v_lij = _mm256_loadu_pd(&L(i, j));
                                _mm256_storeu_pd(&L(i, j), _mm256_fnmadd_pd(v_lik, v_ljk, v_lij));
                            }
                        }
                    #endif

                    for (; i < N; ++i) {
                        L(i, j) -= L(i, k) * l_jk;
                    }
                }
            }
            return MathStatus::SUCCESS;
        }

        /**
         * @brief 스택 요동 방지 및 SIMD 가속 In-place Solver
         */
        template <typename MatL, typename VecB, typename VecX>
        inline void cholesky_solver(const MatL& L, const VecB& b, VecX& x) noexcept {
            using T = std::remove_const_t<typename MatL::value_type>;
            constexpr std::size_t N = MatL::NumRows;

            // 임시 벡터 y의 할당을 피하고 결과 벡터 x 공간을 재활용하여 In-place 연산
            for (std::size_t i = 0; i < N; ++i) {
                x(i) = b(i);
            }

            // 1. Forward substitution (L * x = b) -> Memory Contiguous SAXPY
            for (std::size_t k = 0; k < N; ++k) {
                x(k) /= L(k, k);
                T x_k = x(k);
                std::size_t i = k + 1;
                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_xk = vdupq_n_f32(x_k);
                        for (; i + 3 < N; i += 4) {
                            float32x4_t v_l = vld1q_f32(&L(i, k));
                            float32x4_t v_x = vld1q_f32(&x(i));
                            vst1q_f32(&x(i), vmlsq_f32(v_x, v_l, v_xk));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_xk = vdupq_n_f64(x_k);
                        for (; i + 1 < N; i += 2) {
                            float64x2_t v_l = vld1q_f64(&L(i, k));
                            float64x2_t v_x = vld1q_f64(&x(i));
                            vst1q_f64(&x(i), vmlsq_f64(v_x, v_l, v_xk));
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_xk = _mm256_set1_ps(x_k);
                        for (; i + 7 < N; i += 8) {
                            __m256 v_l = _mm256_loadu_ps(&L(i, k));
                            __m256 v_x = _mm256_loadu_ps(&x(i));
                            _mm256_storeu_ps(&x(i), _mm256_fnmadd_ps(v_l, v_xk, v_x));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_xk = _mm256_set1_pd(x_k);
                        for (; i + 3 < N; i += 4) {
                            __m256d v_l = _mm256_loadu_pd(&L(i, k));
                            __m256d v_x = _mm256_loadu_pd(&x(i));
                            _mm256_storeu_pd(&x(i), _mm256_fnmadd_pd(v_l, v_xk, v_x));
                        }
                    }
                #endif
                for (; i < N; ++i) {
                    x(i) -= L(i, k) * x_k;
                }
            }

            // 2. Backward substitution (L^T * x = y) -> Memory Contiguous Dot Product
            for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0.0);
                std::size_t k = i + 1;
                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_sum = vdupq_n_f32(0.0f);
                        for (; k + 3 < N; k += 4) {
                            float32x4_t v_l = vld1q_f32(&L(k, i));
                            float32x4_t v_x = vld1q_f32(&x(k));
                            v_sum = vfmaq_f32(v_sum, v_l, v_x);
                        }
                        // 배열 덤프 제거, ARM64 네이티브 수평적 덧셈 적용
                        #if defined(__aarch64__)
                            sum += vaddvq_f32(v_sum);
                        #else
                            // 구형 32bit ARM Fallback 
                            float sum_arr[4];
                            vst1q_f32(sum_arr, v_sum);
                            sum += sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                        #endif
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_sum = vdupq_n_f64(0.0);
                        for (; k + 1 < N; k += 2) {
                            float64x2_t v_l = vld1q_f64(&L(k, i));
                            float64x2_t v_x = vld1q_f64(&x(k));
                            v_sum = vfmaq_f64(v_sum, v_l, v_x);
                        }
                        // 배열 덤프 제거, ARM64 네이티브 수평적 덧셈 적용
                        #if defined(__aarch64__)
                            sum += vaddvq_f64(v_sum);
                        #else
                            double sum_arr[2];
                            vst1q_f64(sum_arr, v_sum);
                            sum += sum_arr[0] + sum_arr[1];
                        #endif
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_sum = _mm256_setzero_ps();
                        for (; k + 7 < N; k += 8) {
                            __m256 v_l = _mm256_loadu_ps(&L(k, i));
                            __m256 v_x = _mm256_loadu_ps(&x(k));
                            v_sum = _mm256_fmadd_ps(v_l, v_x, v_sum);
                        }
                        // 레지스터 내 수평적 덧셈 (Horizontal Add) 적용
                        __m128 v_low = _mm256_castps256_ps128(v_sum);
                        __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
                        v_low = _mm_add_ps(v_low, v_high);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        sum += _mm_cvtss_f32(v_low);
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_sum = _mm256_setzero_pd();
                        for (; k + 3 < N; k += 4) {
                            __m256d v_l = _mm256_loadu_pd(&L(k, i));
                            __m256d v_x = _mm256_loadu_pd(&x(k));
                            v_sum = _mm256_fmadd_pd(v_l, v_x, v_sum);
                        }
                        // 레지스터 내 수평적 덧셈 (Horizontal Add 적용)
                        __m128d v_low = _mm256_castpd256_pd128(v_sum);
                        __m128d v_high = _mm256_extractf128_pd(v_sum, 1);
                        v_low = _mm_add_pd(v_low, v_high);
                        v_low = _mm_hadd_pd(v_low, v_low);
                        sum += _mm_cvtsd_f64(v_low);
                    }
                #endif

                for (; k < N; ++k) {
                    sum += L(k, i) * x(k);
                }
                x(i) = (x(i) - sum) / L(i, i);
            }
        }

    } // namespace linalg
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_