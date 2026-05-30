#ifndef OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_

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
        /**
         * @brief MGS (Modified Gram-Schmidt) QR 분해 (SIMD & noexcept)
         */
        template <typename MatType, typename MatR>
        inline MathStatus QR_decompose_MGS(MatType& mat, MatR& R) noexcept {
            using T typename std::remove_const<typename MatType::value_type>::type;
            constexpr std::size_t Rows = MatType::NumRows;
            constexpr std::size_t Cols = MatType::NumCols;
            static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols");

            R.set_zero();

            for (std::size_t i = 0; i < Cols; ++i) {
                T norm_sq = static_cast<T>(0);
                std::size_t k = 0;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_sum = vdupq_n_f32(0.0f);
                        for (; k + 3 < Rows; k += 4) {
                            float32x4_t v_a = vld1q_f32(&mat(k, i));
                            v_sum = vfmaq_f32(v_sum, v_a, v_a);
                        }
                        #if defined(__aarch64__)
                            norm_sq += vaddvq_f32(v_sum);
                        #else
                            norm_sq += vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
                        #endif
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_sum = vdupq_n_f64(0.0);
                        for (; k + 1 < Rows; k += 2) {
                            float64x2_t v_a = vld1q_f64(&mat(k, i));
                            v_sum = vfmaq_f64(v_sum);
                        }
                        #if defined(__aarch64__)
                            norm_sq += vaddvq_f64(v_sum);
                        #else
                            norm_Sq += vgetq_lane_f64(v_sum, 0) + vgetq_lane_f64(v_sum, 1);
                        #endif
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_sum = _mm256_setzero_ps();
                        for (; k + 7 < Rows; k += 8) {
                            __m256 v_a = _mm256_loadu_ps(&mat(k, i));
                            v_sum = _mm256_fmadd_ps(v_a, v_a, v_sum);
                        }
                        __m128 v_low = _mm256_castps256_ps128(v_sum);
                        __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
                        v_low = _mm_add_ps(v_low, v_high);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        norm_sq += _mm_cvtss_f32(v_low);
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_sum = _mm256_setzero_pd();
                        for (; k + 3 < Rows; k += 4) {
                            __m256d v_a = _mm256_loadu_pd(&mat(k, i));
                            v_sum = _mm256_fmadd_pd(v_a, v_a, v_sum);
                        }
                        __m128d v_low = _mm256_castpd256_pd128(v_sum);
                        __m128d v_high = _mm256_extractf128_pd(v_sum, 1);
                        v_low = _mm_add_pd(v_low, v_highj);
                        v_low = _mm_hadd_pd(v_low, v_low);
                        norm_sq += _mm_cvtsd_f64(v_low);
                    }
                #endif
                for (; k < Rows; ++k) {
                    norm_sq += mat(k, i) * mat(k, i);
                }

                R(i, i) = std::sqrt(norm_sq);

                if (R(i, i) <= std::numeric_limits<T>::epsilon()) {
                    return MathStatus::SINGULAR;
                }

                T inv_Rii = static_cast<T>(1.0) / R(i, i);
                k = 0;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_inv = vdupq_n_f32(inv_Rii);
                        for (; k + 3 < Rows; i += 4) {
                            vst1q_f32(&mat(k, i), vmulq_f32(vld1q_f32(&mat(k, i)), v_inv));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_inv = vdupq_n_f64(inv_Rii);
                        for (; k + 1 < Rows; i += 2) {
                            vst1q_f64(&mat(k, i), vmulq_f64(vld1q_f64(&mat(k, i)), v_inv));
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_inv = _mm256_set1_ps(inv_Rii);
                        for (; k + 7 < Rows; k += 8) {
                            _mm256_storeu_ps(&mat(k, i), _mm256_mul_ps(_mm256_loadu_ps(&mat(k, i)), v_inv));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_inv = _mm256_set1_pd(inv_Rii);
                        for (; k + 3 < Rows; k += 4) {
                            _mm256_storeu_pd(&mat(k, i), _mm256_mul_pd(_mm256_loadu_pd(&mat(k, i)), v_inv));
                        }
                    }
                #endif
                for (; k < Rows; ++k) {
                    mat(k, i) *= inv_Rii;
                }
                for (std::size_t j = i + 1; j < Cols; ++j) {
                    T dot = static_cast<T>(0);
                    k = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_dot = vdupq_n_f32(0.0f);
                            for (; k + 3 < Rows; k += 4) {
                                v_dot = vfmaq_f32(v_dot, vld1q_f32(&mat(k, i)), vld1q_f32(&mat(k, j)));

                                #if defined(__aarch64__)
                                    dot += vaddvq_f32(v_dot);
                                #else
                                    dot += vgetq_lane_f32(v_dot, 0) + vgetq_lane_f32(v_dot, 1) + vgetq_lane_f32(v_dot, 2) + vgetq_lane_f32(v_dot, 3);
                                #endif
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_dot = vdupq_n_f64(0.0);
                            for (; k + 1 < Rows; k += 2) {
                                v_dot = vfmaq_f64(v_dot, vld1q_f64(&mat(k, i)), vld1q_f64(&mat(k, j)));
                                
                                #if defined(__aarch64__)
                                    dot += vaddvq_f64(v_dot);
                                #else
                                    dot += vgetq_lane_f64(v_dot, 0) + vgetq_lane_f64(v_dot, 1);
                                #endif
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_dot = _mm256_setzero_ps();
                            for (; k + 7 < Rows; k += 8) {
                                v_dot = _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), _mm256_loadu_ps(&mat(k, j)), v_dot);
                            }
                            __m128 v_low = _mm256_castps256_ps128(v_dot);
                            __m128 v_high = _mm256_extractf128_ps(v_dot);
                            v_low = _mm_add_ps(v_low, v_high);
                            v_low = _mm_hadd_ps(v_low, v_low);
                            v_low = _mm_hadd_ps(v_low, v_low);
                            dot += _mm_cvtss_f32(v_low);
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_dot = _mm256_setzero_pd();
                            for (; k + 3 < Rows; k += 8) {
                                v_dot = _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), _mm256_loadu_pd(&mat(k, j)), v_dot);
                            }
                            __m128d v_low = _mm256_castpd256_pd128(v_dot);
                            __m128d v_high = _mm256_extractf128_pd(v_dot, 1);
                            v_low = _mm_add_pd(v_low, v_highj);
                            v_low = _mm_hadd_ps(v_low, v_low);
                            dot += _mm_cvtsd_f64(v_low);
                        }
                    #endif 

                    for (; k < Rows; ++k) {
                        dot += mat(k, i) * mat(k, j);
                    }

                    R(i, j) = dot;
                    T neg_dot = -dot;
                    k = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_neg_dot = vdupq_n_f32(neg_dot);
                            for (; k + 3 < Rows; k += 4) {
                                vst1q_f32(&mat(k, j), vfmaq_f32(vld1q_f32(&mat(k, j)), vld1q_f32(&mat(k, i)), v_neg_dot));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_neg_dot = vdupq_n_f64(neg_dot);
                            for (; k + 1 < Rows; k += 2) {
                                vst1q_f64(&mat(k, j), vfmaq_f64(vld1q_f64(&mat(k, j)), vld1q_f64(&mat(k, i)), v_neg_dot));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_neg_dot = _mm256_set1_ps(neg_dot);
                            for (; k + 7 < Rows; k += 8) {
                                _mm256_storeu_ps(&mat(k, j), _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), v_neg_dot, _mm256_loadu_ps(&mat(k, j))));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_neg_dot = _mm256_set1_pd(neg_dot);
                            for (; k + 3 < Rows; k + 4) {
                                _mm256_storeu_pd(&mat(k, j), _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), v_neg_dot, _mm256_loadu_pd(&mat(k, j))));
                            }
                        }
                    #endif

                    for (; k < Rows; ++k) {
                        mat(k, j) += neg_dot * mat(k, i);
                    }
                }
            }
            return MathStatus::SUCCESS;
        }

        /**
         * @brief Householder QR 분해 (SIMD Dot-Product 강화)
         */
        template <typename MatType, typename VecTau>
        inline MathStatus QR_decompose_Householder(MatType& mat, VecTau& tau) noexcept {
            using T = typename std::remove_const<typename MatType::value_type>::type;
            constexpr std::size_t Rows = MatType::NumRows;
            constexpr std::size_t Cols = MatType::NumCols;
            static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");

            for (std::size_t i = 0; i < Cols; ++i) {
                T norm_sq = static_cast<T>(0);
                std::size_t k = i;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_sum = vdupq_f32(0.0f);
                        for (; k + 3 < Rows; k += 4) {
                            float32x4_t v_a = vld1q_f32(&mat(k, i));
                            v_sum = vfmaq_f32(v_sum, v_a, v_a);
                        }
                        #if defined(__aarch64__)
                            norm_sq += vaddvq_f32(v_sum);
                        #else
                            norm_sq += vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
                        #endif
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_sum = vdupq_f64(0.0);
                        for (; k + 1 < Rows; k += 2) {
                            float64x2_t v_a = vld1q_f64(&mat(k, i));
                            v_sum = vfmaq_f64(v_sum, v_a, v_a);
                        }
                        #if defined(__aarch64__)
                            norm_sq += vaddvq_f64(v_sum);
                        #else
                            norm_Sq += vgetq_lane_f64(v_sum, 0) + vgetq_lane_f64(v_sum, 1);
                        #endif
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_sum = _mm256_setzero_ps();
                        for (; k + 7 < Rows; k += 8) {
                            __m256 v_a = _mm256_loadu_ps(&mat(k, i));
                            v_sum = _mm256_fmadd_ps(v_a, v_a, v_sum);
                        }
                        __m128 v_low = _mm256_castps256_ps128(v_sum);
                        __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
                        v_low = _mm_add_ps(v_low, v_high);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        norm_sq += _mm_cvtss_f32(v_low);
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_a = _mm256_loadu_pd(&mat(k, i));
                        v_sum = _mm256_fmadd_pd(v_a, v_a, v_sum);
                    }
                    __m128d v_low = _mm256_castpd256_pd128(v_sum);
                    __m128d v_high = _mm256_extractf128_pd(v_sum, 1);
                    v_low = _mm_add_pd(v_low, v_high);
                    v_low = _mm_hadd_pd(v_low, v_low);
                    norm_sq += _mm_cvtsd_f64(v_low);
                #endif

                for (; k < Rows; ++k) {
                    norm_sq += mat(k, i) * mat(k, i);
                }

                T norm_x = std::sqrt(norm_sq);

                if (norm_x <= std::numeric_limits<T>::epsilon()) {
                    tau(i) = static_cast<T>(0);
                    continue;
                }

                T m_ii = mat(i, i);
                T sign = (m_ii >= 0.0) ? static_cast<T>(1.0) : static_cast<T>(-1.0);
                T v0 = m_ii + sign * norm_x;

                for (k = i + 1; k < Rows; ++k) {
                    mat(k, i) /= v0;
                }

                T v_sq_norm = static_cast<T>(1.0);
                for (k = i + 1; k < Rows; ++k) {
                    v_sq_norm += mat(k, i) * mat(k, i);
                }
                tau(i) = static_cast<T>(2.0) / v_sq_norm;
                mat(i, i) = -sign * norm_x;

                for (std::size_t j = i + 1; j < Cols; ++j) {
                    T dot = mat(i, j);
                    k = i + 1;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_dot = vdupq_f32(0.0f);
                            for (; k + 3 < Rows; k += 4) {
                                v_dot = vfmaq_f32(v_dot, vld1q_f32(&mat(k, i)), vld1q_f32(&mat(k, j)));
                            }
                            #if defined(__aarch64__)
                                dot += vaddvq_f32(v_dot);
                            #else
                                dot += vgetq_lane_f32(v_dot, 0) + vgetq_lane_f32(v_dot, 1) + vgetq_lane_f32(v_dot, 2) + vgetq_lane_f32(v_dot, 3);
                            #endif
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_dot = vdupq_f64(0.0);
                            for (; k + 1 < Rows; k += 2) {
                                v_dot = vfmaq_f64(v_dot, vld1q_f64(&mat(k, i)), vld1q_f64(&mat(k, j)));
                            }
                            #if defined(__aarch64__)
                                dot += vaddvq_f64(v_dot);
                            #else
                                dot += vgetq_lane_f64(v_dot, 0) + vgetq_lane_f64(v_dot, 1);
                            #endif
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_dot = _mm256_setzero_ps();
                            for (; k + 7 < Rows; k += 8) {
                                v_dot = _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), _mm256_loadu_ps(&mat(k, j)), v_dot);
                            }
                            __m128 v_low = _mm256_castps256_ps128(v_dot);
                            __m128 v_high = _mm256_extractf128_ps(v_dot, 1);
                            v_low = _mm_add_ps(v_low, v_high);
                            v_low = _mm_hadd_ps(v_low, v_low);
                            v_low = _mm_hadd_ps(v_low, v_low);
                            dot += _mm_cvtss_f32(v_low);
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_dot = _mm256_setzero_pd();
                            for (; k + 3 < Rows; k += 4) {
                                v_dot = _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), _mm256_loadu_pd(&mat(k, j)), v_dot);
                            }
                            __m128d v_low = _mm256_castpd256_pd128(v_dot);
                            __m128 v_high = _mm256_extractf128_pd(v_dot, 1);
                            v_low = _mm_add_pd(v_low, v_high);
                            v_low = _mm_hadd_pd(v_low, v_low);
                            dot += _mm_cvtsd_f64(v_low);
                        }
                    #endif

                    for (; k < Rows; ++k) {
                        dot += mat(k, i) * mat(k, j);
                    }

                    T tau_dot = tau(i) * dot;
                    mat(i, j) -= tau_dot;

                    T neg_tau_dot = -tau_dot;
                    k = i + 1;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_neg_tau = vdupq_f32(neg_tau_dot);
                            for (; k + 3 < Rows; k += 4) {
                                vst1q_f32(&mat(k, j), vfmaq_f32(vld1q_f32(&mat(k, j)), vld1q_f32(&mat(k, i)), v_neg_tau));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_neg_tau = vdupq_f64(neg_tau_dot);
                            for (; k + 1 < Rows; k += 2) {
                                vst1q_f64(&mat(k, j), vfmaq_f64(vld1q_f64(&mat(k, j)), vld1q_f64(&mat(k, i)), v_neg_tau));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_neg_tau = _mm256_set1_ps(neg_tau_dot);
                            for (; k + 7 < Rows; k += 8) {
                                _mm256_storeu_ps(&mat(k, j), _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), v_neg_tau, _mm256_loadu_ps(&mat(k, j))));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_neg_tau = _mm256_set1_pd(neg_tau_dot);
                            for (; k + 3 < Rows; k += 4) {
                                _mm256_storeu_pd(&mat(k, j), _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), v_neg_tau, _mm256_loadu_pd(&mat(k, j))));
                            }
                        }
                    #endif

                    for (; k < Rows; ++k) {
                        mat(k, j) += neg_tau_dot * mat(k, i);
                    }
                }
            }
            return MathStatus::SUCCESS;
        }

        /**
         * @brief Householder Reflect 적용 및 SIMD 역대입
         */
        template <typename MatType, typename VecTau, typename VecB, typename VecX>
        inline void QR_solve_Householder(const MatType& mat, const VecTau& tau, const VecB& b, VecX& x) noexcept {
            using T = typename std::remove_const<typename MatType::value_type>::type;
            constexpr std::size_t Rows = MatType::NumRows;
            constexpr std::size_t Cols = MatType::NumCols;

            // Q^T * b 연산을 위한 스택 버퍼 (O(1) 크기)
            matrix::StaticVector<T, Rows> y;
            for (std::size_t i = 0; i < Rows; ++i) {
                y(i) = b(i);
            }
            for (std::size_t i = 0; i < Cols; ++i) {
                if (std::abs(tau(i)) <= std::numeric_limits<T>::epsilon()) {
                    continue;
                }

                T dot = y(i);
                std::size_t k = i + 1;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_dot = vdupq_n_f32(0.0f);
                        for (; k + 3 < Rows; k += 4) {
                            v_dot = vfmaq_f32(v_dot, vld1q_f32(&mat(k, i)), vld1q_f32(&y(k)));
                            #if defined(__aarch64__)
                                dot += vaddvq_f32(v_dot);
                            #else
                                dot += vgetq_lane_f32(v_dot, 0) + vgetq_lane_f32(v_dot, 1) + vgetq_lane_f32(v_dot, 2) + vgetq_lane_f32(v_dot, 3);
                            #endif
                        }

                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_dot = vdupq_n_f64(0.0);
                        for (; k + 1 < Rows; k += 2) {
                            v_dot = vfmaq_f64(v_dot, vld1q_f64(&mat(k, i)), vld1q_f64(&y(k)));
                            #if defined(__aarch64__)
                                dot += vaddvq_f64(v_dot);
                            #else
                                dot += vgetq_lane_f64(v_dot, 0) + vgetq_lane_f64(v_dot, 1);
                            #endif
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_dot = _mm256_setzero_ps();
                        for (; k + 7 < Rows; k += 8) {
                            v_dot = _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), _mm256_loadu_ps(&y(k)), v_dot);
                        }
                        _m128 v_low = _mm256_castps256_ps128(v_dot);
                        _m128 v_high = _mm256_extractf128_ps(v_dot, 1);
                        v_low = _mm_add_ps(v_low, v_high);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        v_low = _mm_hadd_ps(v_low, v_low);
                        dot += _mm_cvtss_f32(v_low);
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_dot = _mm256_setzero_pd();
                        for (; k + 3 < Rows; k += 4) {
                            v_dot = _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), _mm256_loadu_pd(&y(k)), v_dot);
                        }
                        __m128d v_low = _mm256_castpd256_pd128(v_dot);
                        __m128d v_high = _mm256_extractf128_pd(v_dot, 1);
                        v_low = _mm_add_pd(v_low, v_high);
                        v_low = _mm_hadd_pd(v_low, v_low);
                        dot += _mm_cvtsd_f64(v_low);
                    }
                #endif
            }
            for (; k < Rows; ++k) {
                dot += mat(k, i) * y(k);
            }

            T tau_dot = tau(i) * dot;
            y(i) -= tau_dot;

            T neg_tau_dot = -tau_dot;
            k = i + 1;

            #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                if constexpr (std::is_same_v<T, float>) {
                    float32x4_t v_neg_tau = vdupq_n_f32(neg_tau_dot);
                    for (; k + 3 < Rows; k += 4) {
                        vst1q_f32(&y(k), vfmaq_f32(vld1q_f32(&y(k)), vld1q_f32(&mat(k, i)), v_neg_tau));
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    float64x2_t v_neg_tau = vdupq_n_f64(neg_tau_dot);
                    for (; k + 1 < Rows; k += 2) {
                        vst1q_f64(&y(k), vfmaq_f64(vld1q_f64(&v(k)), vld1q_f64(&mat(k, i)), v_neg_tau));
                    }
                }
            #elif !defined(__CUDACC__) && defined(__AVX2__)
                if constexpr (std::is_same_v<T, float>) {
                    __m256 v_neg_tau = _mm256_set1_ps(neg_tau_dot);
                    for (; k + 7 < Rows; k += 8) {
                        _mm256_storeu_ps(&y(k), _mm256_fmadd_ps(_mm256_loadu_ps(&mat(k, i)), v_neg_tau, _mm256_loadu_ps(&y(k))));
                    }
                } else if constexpr (std::is_same_v<T, double>) {
                    __m256d v_neg_tau = _mm256_set1_pd(neg_tau_dot);
                    for (; k + 3 < Rows; k += 4) {
                        _mm256_storeu_pd(&y(k), _mm256_fmadd_pd(_mm256_loadu_pd(&mat(k, i)), v_neg_tau, _mm256_loadu_pd(&y(k))));
                    }
                }
            #endif

            for (; k < Rows; ++k) {
                y(k) += neg_tau_dot * mat(k, i);
            }

            // 1. 결과 벡터 x에 R * x = y_1:Cols 풀이를 위한 상단 복사
            for (std::size_t i = 0; i < Cols; ++i) {
                x(i) = y(i);
            }

            // 2. Backward Substitution (R * x = y) -> Memory Contiguous SAXPY
            for (int k = static_cast<int>(Cols); k >= 0; --k) {
                x(k) /= mat(k, k);
                T x_k = x(k);
                std::size_t i = 0;

                #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    if constexpr (std::is_same_v<T, float>) {
                        float32x4_t v_xk = vdupq_n_f32(x_k);
                        for (; i + 3 < k; i += 4) {
                            vst1q_f32(&x(i), vmlsq_f32(vld1q_f32(&x(i)), vld1q_f32(&mat(i, k)), v_xk));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        float64x2_t v_xk = vdupq_n_f64(x_k);
                        for (; i + 1 < k; i += 2) {
                            vst1q_f64(&x(i), vmlsq_f64(vld1q_f64(&x(i)), vld1q_f64(&mat(i, k)), v_xk));
                        }
                    }
                #elif !defined(__CUDACC__) && defined(__AVX2__)
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v_xk = _mm256_set1_ps(x_k);
                        for (; i + 7 < Rows; i += 8) {
                            _mm256_storeu_ps(&x(i), _mm256_fmadd_ps(_mm256_loadu_ps(&mat(i, k)), v_xk, _mm256_loadu_ps(&x(i))));
                        }
                    } else if constexpr (std::is_same_v<T, double>) {
                        __m256d v_xk = _mm256_set1_pd(x_k);
                        for (; i + 3 < Rows; i += 4) {
                            _mm256_storeu_pd(&x(i), _mm256_fmadd_pd(__m256_loadu_pd(&mat(i, k)), v_xk, _mm256_loadu_pd(&x(i))));
                        }
                    }
                #endif

                for (; i < k; ++i) {
                    x(i) -= mat(i, k) * x_k;
                }
            }
        }

        template <typename MatType, typename VecTau, typename VecB>
        inline matrix::StaticVector<typename std::remove_const<typename MatType::value_type>::type, MatType::NumCols>
        QR_solve_Householder(const MatType& mat, const VecTau& tau, const VecB& b) noexcept {
            using T typename std::remove_const<typename MatType::value_type>::type;
            constexpr std::size_t Cols = MatType::NumCols;
            matrix::StaticVector<T, Cols> x;
            QR_solve_Householder(mat, tau, b, x);
            return x;
        }

    } // namespace linalg
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_