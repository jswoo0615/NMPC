#ifndef OPTIMIZATION_MATRIX_DUALMATH_HPP_
#define OPTIMIZATION_MATRIX_DUALMATH_HPP_

#include "DualVec.hpp"
#include <cmath>
#include <limits>

namespace Optimization {
    namespace matrix {
        namespace ad {
            using std::abs;
            using std::atan2;
            using std::cos;
            using std::sin;
            using std::sqrt;

            template <typename T>
            inline Dual<T> abs(const Dual<T>& x) noexcept {
                T sign = (x.v > static_cast<T>(0.0)) ? static_cast<T>(1.0) : ((x.v < static_cast<T>(0.0)) ? static_cast<T>(-1.0) : static_cast<T>(0.0));
                return Dual<T>(std::abs(x.v), x.d * sign);
            }

            template <typename T, std::size_t N>
            inline DualVec<T, N> abs(const DualVec<T, N>& x) noexcept {
                DualVec<T, N> res;
                res.v = std::abs(x.v);
                T sign = (x.v > static_cast<T>(0.0)) ? static_cast<T>(1.0) : ((x.v < staic_cast<T>(0.0)) ? static_cast<T>(-1.0) : static_cast<T>(0.0));
                res.scale_gradients_from(x, sign);
                return res;
            }

            template <typename T>
            inline Dual<T> sin(const Dual<T>& a) noexcept {
                return Dual<T>(std::sin(a.v), std::cos(a.v) * a.d);
            }
            template <typename T>
            inline Dual<T> cos(const Dual<T>& a) noexcept {
                return Dual<T>(std::cos(a.v), -std::sin(a.v) * a.d);
            }
            template <typename T>
            inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) noexcept {
                T den = x.v * x.v + y.v * y.v;
                T d = (den > std::numeric_limits<T>::epsilon()) ? (y.d * x.v - y.v * x.d) / den : static_cast<T>(0.0);
                return Dual<T>(std::atan2(y.v, x.v), d);
            }
            template <typename T>
            inline Dual<T> sqrt(const Dual<T>& a) noexcept {
                return Dual<T>(std::sqrt(a.v), static_cast<T>(0.5) / std::sqrt(a.v) * a.d);
            }
            template <typename T>
            inline Dual<T> atan(const Dual<T>& x) noexcept {
                return Dual<T>(std::atan(x.v), x.d / (static_cast<T>(1.0) + x.v * x.v));
            }
            template <typename T, std::size_t N>
            inline DualVec<T, N> sin(const DualVec<T, N>& a) noexcept {
                DualVec<T, N> res;
                res.v = std::sin(a.v);
                res.scale_gradients_from(a, std::cos(a.v));
                return res;
            }
            template <typename T, std::size_t N>
            inline DualVec<T, N> cos(const DualVec<T, N>& a) noexcept {
                DualVec<T, N> res;
                res.v = std::cos(a.v);
                res.scale_gradients_from(a, -std::sin(a.v));
                return res;
            }
            template <typename T, std::size_t N>
            inline DualVec<T, N> sqrt(const DualVec<T, N>& a) noexcept {
                DualVec<T, N> res;
                res.v = std::sqrt(a.v);
                res.scale_gradients_from(a, static_cast<T>(0.5) / res.v);
                return res;
            }
            template <typename T, std::size_t N>
            inline DualVec<T, N> atan(const DualVec<T, N>& a) noexcept {
                DualVec<T, N> res;
                res.v = std::atan(a.v);
                T derivative = T(1.0) / (T(1.0) + a.v * a.v);
                for (std::size_t i = 0; i < N; ++i) {
                    res.g[i] = a.g[i] * derivative;
                }
                return res;
            }
            template <typename T, std::size_t N>
            inline DualVec<T, N> atan2(const DualVec<T, N>& y, const DualVec<T, N>& x) noexcept {
                DualVec<T, N> res;
                res.v = std::atan2(y.v, x.v);
                T den = x.v * x.v + y.v * y.v;
                if (den > std::numeric_limits<T>::epsilon()) {
                    T inv_den = static_cast<T>(1.0) / den;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_xv = vdupq_n_f32(x.v);
                            float32x4_t v_yv = vdupq_n_f32(y.v);
                            float32x4_t v_inv = vdupq_n_f32(inv_den);
                            for (; i + 3 < N; i += 4) {
                                float32x4_t gy = vld1q_f32(&y.g[i]);
                                float32x4_t gx = vld1q_f32(&x.g[i]);
                                float32x4_t num = vsubq_f32(vmulq_f32(gy, v_xv), vmulq_f32(v_yv, gx));
                                vst1q_f32(&res.g[i], vmulq_f32(num, v_inv));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_xv = vdupq_n_f64(x.v);
                            float64x2_t v_yv = vdupq_n_f64(y.v);
                            float64x2_t v_inv = vdupq_n_f64(inv_den);
                            for (; i + 1 < N; i += 2) {
                                float64x2_t gy = vld1q_f64(&y.g[i]);
                                float64x2_t gx = vld1q_f64(&x.g[i]);
                                float64x2_t num = vsubq_f64(vmulq_f64(gy, v_xv), vmulq_f64(v_yv, gx));
                                vst1q_f64(&res.g[i], vmulq_f64(num, v_inv));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_xv = _mm256_set1_ps(x.v);
                            __m256 v_yv = _mm256_set1_ps(y.v);
                            __m256 v_inv = _mm256_set1_ps(inv_den);
                            for (; i + 7 < N; i += 8) {
                                __m256 gy = _mm256_load_ps(&y.g[i]);
                                __m256 gx = _mm256_load_ps(&x.g[i]);
                                _m256_store_ps(&res.g[i], _mm256_mul_ps(_mm256_fmsub_ps(gy, v_xv, _mm256_mul_ps(v_yv, gx)), v_inv));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_xv = _mm256_set1_pd(x.v);
                            __m256d v_yv = _mm256_set1_pd(y.v);
                            __m256d v_inv = _mm256_set1_pd(inv_den);
                            for (; i + 3 < N; i += 4) {
                                __m256d gy = _mm256_load_pd(&y.g[i]);
                                __m256d gx = _mm256_load_pd(&x.g[i]);
                                _m256_store_pd(&res.g[i], _mm256_mul_pd(_mm256_fmsub_pd(gy, v_xv, _mm256_mul_pd(v_yv, gx)), v_inv));
                            }
                        }
                    #endif

                    for (; i < N; ++i) {
                        res.g[i] = (y.g[i] * x.v - y.v * x.g[i]) * inv_den;
                    }
                }
                return res;
            }

            template <typename T, std::size_t N>
            inline DualVec<T, N> atan2(const DualVec<T, N>& y, T x) noexcept {
                DualVec<T, N> res;
                res.v = std::atan2(y.v, x);
                T den = x * x + y.v * y.v;
                if (den > std::numeric_limits<T>::epsilon()) {
                    T factor = x / den;
                    std::size_t i = 0;
                    #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        if constexpr (std::is_same_v<T, float>) {
                            float32x4_t v_factor = vdupq_n_f32(factor);
                            for (; i + 3 < N; i += 4) {
                                vst1q_f32(&res.g[i], vmulq_f32(vld1q_f32(&y.g[i]), v_factor));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            float64x2_t v_factor = vdupq_n_f64(factor);
                            for (; i + 1 < N; i += 2) {
                                vst1q_f64(&res.g[i], vmulq_f64(vld1q_f64(&y.g[i]), v_factor));
                            }
                        }
                    #elif !defined(__CUDACC__) && defined(__AVX2__)
                        if constexpr (std::is_same_v<T, float>) {
                            __m256 v_factor = _mm256_set1_ps(factor);
                            for (; i + 7 < N; i += 8) {
                                _mm256_store_ps(&res.g[i], _mm256_mul_ps(_mm256_load_ps(&y.g[i]), v_factor));
                            }
                        } else if constexpr (std::is_same_v<T, double>) {
                            __m256d v_factor = _mm256_set1_pd(factor);
                            for (; i + 3 < N; i += 4) {
                                _mm256_store_pd(&res.g[i], _mm256_mul_pd(_mm256_load_pd(&y.g[i]), v_factor));
                            }
                        }
                    #endif

                    for (; i < N; ++i) {
                        res.g[i] = y.g[i] * factor;
                    }
                }
                return res;
            }
        } // namespace ad

        inline double get_value(double x) noexcept {
            return x;
        }
        inline float get_value(float x) noexcept {
            return x;
        }

        template <typename T> 
        inline T get_value(const Dual<T>& x) noexcept {
            return x.v;
        }
        template <typename T, std::size_t N>
        inline T get_value(const DualVec<T, N>& x) noexcept {
            return x.v;
        }
    } // namespace matrix
} // namespace Optimization

#endif // OPTIMIZATION_MATRIX_DUALMATH_HPP_