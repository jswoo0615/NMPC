#ifndef OPTIMIZATION_MATRIX_STATIC_MATRIX_HPP_
#define OPTIMIZATION_MATRIX_STATIC_MATRIX_HPP_

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    #include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <type_traits>

#include "Optimization/Matrix/Core/StaticMatrixView.hpp"
#include "Optimization/Utils/CUDAMacros.hpp"

namespace Optimization {
    namespace matrix {
        template <typename T, std::size_t Rows, std::size_t Cols>
        class StaticMatrix;

        template <typename T, std::size_t N>
        using StaticVector = StaticMatrix<T, N, 1>;

        template <typename T, std::size_t Rows, std::size_t Cols>
        class alignas(64) StaticMatrix {
            private:
                alignas(64) T data_[Rows * Cols] {};
                // 하드웨어 가속 강제 복사 
                CUDA_CALLABLE inline void simd_copy_from(const T* src) noexcept {
                    constexpr std::size_t size = Rows * Cols;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 3 < size; i += 4) {
                                vst1q_f32(&data_[i], vld1q_f32(&src[i]));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 1 < size; i += 2) {
                                vst1q_f64(&data_[i], vld1q_f64(&src[i]));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 7 < size; i += 8) {
                                _mm256_store_ps(&data_[i], _mm256_load_ps(&src[i]));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 3 < size; i += 4) {
                                _mm256_store_pd(&data_[i], _mm256_load_pd(&src[i]));
                            }
                        }
                    #endif 

                    for (; i < size; ++i) {
                        data_[i] = src[i];
                    }
                }
            public:
                using value_type = T;
                static constexpr std::size_t NumRows = Rows;
                static constexpr std::size_t NumCols = Cols;

                CUDA_CALLABLE inline StaticMatrix() noexcept {
                    set_zero();
                }

                // SIMD 복사 생성자
                CUDA_CALLABLE inline StaticMatrix(const StaticMatrix& other) noexcept {
                    simd_copy_from(other.data_);
                }

                // 명시적 SIMD 복사 대입 연산자
                CUDA_CALLABLE inline StaticMatrix& operator=(const StaticMatrix& other) noexcept {
                    if (this != &other) {
                        simd_copy_from(other.data_);
                    }
                    return *this;
                }

                CUDA_CALLABLE inline T& operator()(int r, int c) noexcept {
                    assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
                    return data_[c * Rows + r];
                }
                CUDA_CALLABLE inline const T& operator()(int r, int c) const noexcept {
                    assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
                    return data_[c * Rows + r];
                }
                CUDA_CALLABLE inline T& operator()(std::size_t i) noexcept {
                    assert(i < Rows * Cols);
                    return data_[i];
                }
                CUDA_CALLABLE inline const T& operator()(std::size_t i) const noexcept {
                    assert(i < Rows * Cols);
                    return data_[i];
                }
                CUDA_CALLABLE inline T* data_ptr() noexcept {
                    return data_;
                }
                CUDA_CALLABLE inline const T* data_ptr() const noexcept {
                    return data_;
                }

                CUDA_CALLABLE inline StaticMatrix& set_zero() noexcept {
                    constexpr std::size_t size = Rows * Cols;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t zero = vdupq_n_f32(0.0f);
                            for (; i + 3 < size; i += 4) {
                                vst1q_f32(&data_[i], zero);
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t zero = vdupq_n_f64(0.0);
                            for (; i + 1 < size; i += 2) {
                                vst1q_f64(&data_[i], zero);
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 zero = _mm256_setzero_ps();
                            for (; i + 7 < size; i += 8) {
                                _mm256_store_ps(&data_[i], zero);
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d zero = _mm256_setzero_pd();
                            for (; i + 3 < size; i += 4) {
                                _mm256_store_pd(&data_[i], zero);
                            }
                        }
                    #endif

                    for (; i < size; ++i) {
                        data_[i] = static_cast<T>(0.0);
                    }
                    return *this;
                }

                CUDA_CALLABLE inline StaticMatrix& saxpy(T scalar, const StaticMatrix& rhs) noexcept {
                    constexpr std::size_t size = Rows * Cols;
                    std::size_t i = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_scalar = vdupq_n_f32(scalar);
                            for (; i + 3 < size; i += 4) {
                                float32x4_t v_rhs = vld1q_f32(&rhs.data_[i]);
                                float32x4_t v_res = vld1q_f32(&data_[i]);
                                vst1q_f32(&data_[i], vfmaq_f32(v_res, v_rhs, v_scalar));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_scalar = vdupq_n_f64(scalar);
                            for (; i + 1 < size; i += 2) {
                                float64x2_t v_rhs = vld1q_f64(&rhs.data_[i]);
                                float64x2_t v_res = vld1q_f64(&data_[i]);
                                vst1q_f64(&data_[i], vfmaq_f64(v_res, v_rhs, v_scalar));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_scalar = _mm256_set1_ps(scalar);
                            for (; i + 7 < size; i += 8) {
                                __m256 v_rhs = _mm256_load_ps(&rhs.data_[i]);
                                __m256 v_res = _mm256_load_ps(&data_[i]);
                                _mm256_store_ps(&data_[i], _mm256_fmadd_ps(v_rhs, v_scalar, v_res));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_scalar = _mm256_set1_pd(scalar);
                            for (; i + 3 < size; i += 4) {
                                __m256d v_rhs = _mm256_load_pd(&rhs.data_[i]);
                                __m256d v_res = _mm256_load_pd(&data_[i]);
                                _mm256_store_pd(&data_[i], _mm256_fmadd_pd(v_rhs, v_scalar, v_res));
                            }
                        }
                    #endif

                    for (; i < size; ++i) {
                        data_[i] += scalar * rhs.data_[i];
                    }
                    return *this;
                }

                CUDA_CALLABLE inline StaticMatrix& operator+=(const StaticMatrix& rhs) noexcept {
                    constexpr std::size_t total = Rows * Cols;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 3 < total; i += 4) {
                                vst1q_f32(&data_[i], vaddq_f32(vld1q_f32(&data_[i]), vld1q_f32(&rhs.data_[i])));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 1 < total; i += 2) {
                                vst1q_f64(&data_[i], vaddq_f64(vld1q_f64(&data_[i]), vld1q_f64(&rhs.data_[i])));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 7 < total; i += 8) {
                                _mm256_store_ps(&data_[i], _mm256_add_ps(_mm256_load_ps(&data_[i]), _mm256_load_ps(&rhs.data_[i])));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 3 < total; i += 4) {
                                _mm256_store_pd(&data_[i], _mm256_add_pd(_mm256_load_pd(&data_[i]), _mm256_load_pd(&rhs.data_[i])));
                            }
                        }
                    #endif
                    for (; i < total; ++i) {
                        data_[i] += rhs.data_[i];
                    }
                    return *this;
                }

                CUDA_CALLABLE inline StaticMatrix& operator-=(const StaticMatrix& rhs) noexcept {
                    constexpr std::size_t total = Rows * Cols;
                    std::size_t i = 0;

                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 3 < total; i += 4) {
                                vst1q_f32(&data_[i], vsubq_f32(vld1q_f32(&data_[i]), vld1q_f32(&rhs.data_[i])));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 1 < total; i += 2) {
                                vst1q_f64(&data_[i], vsubq_f64(vld1q_f64(&data_[i]), vld1q_f64(&rhs.data_[i])));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            for (; i + 7 < total; i += 8 ) {
                                _mm256_store_ps(&data_[i], _mm256_sub_ps(_mm256_load_ps(&data_[i]), _mm256_load_ps(&rhs.data_[i])));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            for (; i + 3 < total; i += 4) {
                                _mm256_store_pd(&data_[i], _mm256_sub_pd(_mm256_load_pd(&data_[i]), _mm256_load_pd(&rhs.data_[i])));
                            }
                        }
                    #endif

                    for (; i < total; ++i) {
                        data_[i] -= rhs.data_[i];
                    }
                    return *this;
                }
                CUDA_CALLABLE inline StaticMatrix& operator*=(T scalar) noexcept {
                    constexpr std::size_t total = Rows * Cols;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_s = vdupq_n_f32(scalar);
                            for (; i + 3 < total; i += 4) {
                                vst1q_f32(&data_[i], vmulq_f32(vld1q_f32(&data_[i]), v_s));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_s = vdupq_n_f64(scalar);
                            for (; i + 1 < total; i += 2) {
                                vst1q_f64(&data_[i], vmulq_f64(vld1q_f64(&data_[i]), v_s));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_s = _mm256_set1_ps(scalar);
                            for (; i + 7 < total; i += 8) {
                                _mm256_store_ps(&data_[i], _mm256_mul_ps(_mm256_load_ps(&data_[i]), v_s));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_s = _mm256_set1_pd(scalar);
                            for (; i + 3 < total; i += 4) {
                                _mm256_store_pd(&data_[i], _mm256_mul_pd(_mm256_load_pd(&data_[i]), v_s));
                            }
                        }
                    #endif

                    for (; i < total; ++i) {
                        data_[i] *= scalar;
                    }
                    return *this;
                }

                // 전치 연산 (Transpose)은 O(N^2) 복사 비용을 발생시킵니다
                // 이 함수의 호출 빈도를 0으로 유지합니다. 대신 linalg::multiply_AT_B 사용
                [[deprecated("Use linalg::multiply_AT_B to prevent memory allocation and copying")]]
                CUDA_CALLABLE StaticMatrix<T, Cols, Rows> transpose() const noexcept {
                    StaticMatrix<T, Cols, Rows> res;
                    for (std::size_t j = 0; j < Cols; ++j) {
                        for (std::size_t i = 0; i < Rows; ++i) {
                            res(static_cast<int>(j), static_cast<int>(i)) = (*this)(static_cast<int>(i), static_cast<int>(j));
                        }
                    }
                    return res;
                }
        };
    } // namespace matrix
} // namespace Optimization
#endif // OPTIMIZATION_MATRIX_STATIC_MATRIX_HPP_