#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_

#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

#if !defined(__CUDACC__) && (defined(__ARM_NEON) && defined(__ARM_NEON__))
    #include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace Optimization {
    namespace linalg {
        template <typename T, std::size_t M, std::size_t K, std::size_t N>
        inline void multply(const matrix::StaticMatrix<T, M, K>& A, 
                            const matrix::StaticMatrix<T, K, N>& B, 
                            matrix::StaticMatrix<T, M, N>& C) noexcept {
            C.set_zero();
            const T* a_ptr = A.data_ptr();
            const T* b_ptr = B.data_ptr();
            T* c_ptr = C.data_ptr();

            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t k = 0; k < K; ++k) {
                    T b_val = b_ptr[j * K + k];
                    const T* a_col = a_ptr + k * M;
                    T* col = c_ptr + j * M;
                    std::size_t i = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_b = vdupq_n_f32(b_val);
                            for (; i + 3 < M; i += 4) {
                                vst1q_f32(&c_col[i], vfmaq_f32(vld1q_f32(&c_col[i]), vld1q_F32(&a_col[i]), v_b));
                            }
                        } else if constexpr (std::is_Same_V<T, double>) {
                            float64x2_t v_b = vdupq_n_f64(b_val);
                            for (; i + 1 < M; i += 2) {
                                vst1q_f64(&c_col[i], vfmaq_f64(vld1q_f64(&c_col[i]), vld1q_f64(&a_col[i]), v_b));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_b = _mm256_set1_ps(b_val);
                            for (; i + 7 < M; i += 8) {
                                _mm256_storeu_ps(&c_col[i], _mm256_fmadd_ps(_mm256_loadu_ps(&a_col[i]), v_b, _mm256_loadu_ps(&c_col[i])));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_b = _mm256_set1_pd(b_val);
                            for (; i + 3 < M; i += 4) {
                                _mm256_storeu_pd(&c_col[i], _mm256_fmadd_pd(_mm256_loadu_pd(&a_col[i]), v_b, _mm256_loadu_pd(&c_col[i])));
                            }
                        }
                    #endif
                    for (; i < M; ++i) {
                        c_col[i] += a_col[i] * b_val;
                    }
                }
            }
        }

        template <typename T, std::size_t M, std::size_t K, std::size_t N>
        inline void multiply_AT_B(const matrix::StaticMatrix<T, M, K>& A,
                                  const matrix::StaticMatrix<T, K, N>& B, 
                                  matrix::StaticMatrix<T, M, N>& C) noexcept {
            const T* a_ptr = A.data_ptr();
            const T* b_ptr = B.data_ptr();
            T* c_ptr = C.data_ptr();

            for (std::size_t j = 0; j < N; ++j) {
                for (std::size_t i = 0; i < M; ++i) {
                    const T* a_col = a_ptr + i * K;
                    const T* b_col = b_ptr + j * K;
                    T sum = 0.0;
                    std::size_t k = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM__NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_sum = vdupq_n_f32(0.0f);
                            for (; k + 3 < K; k += 4) {
                                v_sum = vfaq_f32(v_sum, vld1q_f32(&a_col[k]), vld1q_f32(&b_col[k]));
                            }
                            #if defined(__aarch64__)
                                sum += vaddvq_f32(v_sum);
                            #else
                                sum += vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
                            #endif
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_sum = vdup_n_f64(0.0);
                            for (; k + 1 < K; k += 2) {
                                v_sum = vfmaq_f64(v_sum, vld1q_f64(&a_col[k]), vld1q_f64(&b_col[k]));
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
                            for (; i + 7 < K; k += 8) {
                                v_sum = _mm256_fmadd_ps(_mm256_loadu_ps(&a_col[k]), _mm256_loadu_ps(&b_col[k]), v_sum);
                            }
                            __m128 v_low = _mm256_castps256_ps128(v_sum);
                            __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
                            v_low = _mm_add_ps(v_low, v_high);
                            v_low = _mm_hadd_ps(v_low, v_high);
                            v_low = _mm_hadd_ps(v_low, v_high);
                            sum += _mm_cvtss_f32(v_low);
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_sum = _mm256_setzero_pd();
                            for (; k + 3 < k; k += 4) {
                                v_sum = _mm256_fmadd_pd(_mm256_loadu_pd(&a_col[k]), _mm256_loadu_pd(&b_col[k]), v_sum);
                            }
                            __m128d v_low = _mm256_castpd256_pd128(v_sum);
                            __m128d v_high = _mm256_extractf128_pd(v_sum, 1);
                            v_low = _mm_add_pd(v_low, v_high);
                            v_low = _mm_hadd_pd(v_low, v_low);
                            sum += _mm_cvtsd_f64(v_low);
                        }
                    #endif 
                    for (; k < K; ++k) {
                        sum += a_col[k] * b_col[k];
                    }
                    c_ptr[j * M + i] = sum;
                }
            }
        }
    } // namespace linalg

    template <typename T, std::size_t M, std::size_t K, std::size_t N>
    inline matrix::StaticMatrix<T, M, N> operator*(const matrix::StaticMatrix<T, M, K>& A, const matrix::StaticMatrix<T, K, N>& B) noexcept {
        matrix::StaticMatrix<T, M, N> C;
        linalg::multply(A, B, C);
        return C;
    }

    template <typename T, std::size_t M, std::size_t N>
    inline matrix::StaticMatrix<T, M, N> operator*(T scalar, const matrix::StaticMatrix<T, M, N>& A) noexcept {
        matrix::StaticMatrix<T, M, N> C = A;
        constexpr std::size_t size = M * N;
        for (std::size_t i = 0; i < size; ++i) {
            C(i) *= scalar;
        }
        return C;
    }

    template <typename T, std::size_t M, std::size_t N>
    inline matrix::StaticMatrix<T, M, N> operator+(const matrix::StaticMatrix<T, M, N>& A, 
                                                   const matrix::StaticMatrix<T, M, N>& B) noexcept {
        matrix::StaticMatrix<T, M, N> C = A;
        C += B;
        return C;
    }

    template <typename T, std::size_t M, std::size_t N>
    inline matrix::StaticMatrix<T, M, N> operator-(const matrix::StaticMatrix<T, M, N>& A, 
                                                   const matrix::StaticMatrix<T, M, N>& B) noexcept {
        matrix::StaticMatrix<T, M, N> C = A;
        C -= B;
        return C;
    }
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_