#ifndef OPTIMIZATION_MATRIX_DUALVEC_HPP_
#define OPTIMIZATION_MATRIX_DUALVEC_HPP_

#include "DualScalar.hpp"
#include <cstddef>

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Optimization {
namespace matrix {

/**
 * @brief NMPC용 다차원 SIMD 가속 Dual Number (DualVec)
 * @details
 * 미분 배열의 정렬 규격 64바이트 캐시 라인으로 고정 (alignas(64))
 * 초기화 과정의 스칼라 루프를 SIMD 제로 셋으로 치환하여 생성 (Instantiation)
 * 병목 소거
 */
template <typename T, std::size_t N> struct alignas(64) DualVec {
  T v;
  alignas(64) T g[N]; // 64-byte 캐시 라인 완벽 동기화

  inline void simd_set_zero_gradients() noexcept {
    std::size_t i = 0;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t zero = vdupq_n_f32(0.0f);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&g[i], zero);
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t zero = vdupq_n_f64(0.0);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&g[i], zero);
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 zero = _mm256_setzero_ps();
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&g[i], zero);
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d zero = _mm256_setzero_pd();
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&g[i], zero);
      }
    }
#endif

    for (; i < N; ++i) {
      g[i] = static_cast<T>(0.0);
    }
  }

  inline DualVec() noexcept : v(static_cast<T>(0.0)) {
    simd_set_zero_gradients();
  }
  inline DualVec(T val) noexcept : v(val) { simd_set_zero_gradients(); }
  inline DualVec &operator=(T val) noexcept {
    v = val;
    simd_set_zero_gradients();
    return *this;
  }

  DualVec(const DualVec&) = default;
  DualVec& operator=(const DualVec&) = default;

  static inline DualVec make_variable(T val, std::size_t idx) noexcept {
    DualVec res(val);
    if (idx < N) {
      res.g[idx] = static_cast<T>(1.0);
    }
    return res;
  }

  inline DualVec operator-() const noexcept {
    DualVec res;
    res.v = -v;
    std::size_t i = 0;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t zero = vdupq_n_f32(0.0f);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vsubq_f32(zero, vld1q_f32(&g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t zero = vdupq_n_f64(0.0);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vsubq_f64(zero, vld1q_f64(&g[i])));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 zero = _mm256_setzero_ps();
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&res.g[i], _mm256_sub_ps(zero, _mm256_load_ps(&g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d zero = _mm256_setzero_pd();
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&res.g[i], _mm256_sub_pd(zero, _mm256_load_pd(&g[i])));
      }
    }
#endif

    for (; i < N; ++i) {
      res.g[i] = -g[i];
    }
    return res;
  }

  inline DualVec operator+(const DualVec &rhs) const noexcept {
    DualVec res;
    res.v = v + rhs.v;
    std::size_t i = 0;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vaddq_f32(vld1q_f32(&g[i]), vld1q_f32(&rhs.g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vaddq_f64(vld1q_f64(&g[i]), vld1q_f64(&rhs.g[i])));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&res.g[i], _mm256_add_ps(_mm256_load_ps(&g[i]),
                                                 _mm256_load_ps(&rhs.g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&res.g[i], _mm256_add_pd(_mm256_load_pd(&g[i]),
                                                 _mm256_load_pd(&rhs.g[i])));
      }
    }
#endif

    for (; i < N; ++i) {
      res.g[i] = g[i] + rhs.g[i];
    }
    return res;
  }

  inline DualVec operator-(const DualVec &rhs) const noexcept {
    DualVec res;
    res.v = v - rhs.v;
    std::size_t i = 0;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vsubq_f32(vld1q_f32(&g[i]), vld1q_f32(&rhs.g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vsubq_f64(vld1q_f64(&g[i]), vld1q_f64(&rhs.g[i])));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&res.g[i], _mm256_sub_ps(_mm256_load_ps(&g[i]),
                                                 _mm256_load_ps(&rhs.g[i])));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&res.g[i], _mm256_sub_pd(_mm256_load_pd(&g[i]),
                                                 _mm256_load_pd(&rhs.g[i])));
      }
    }
#endif

    for (; i < N; ++i) {
      res.g[i] = g[i] - rhs.g[i];
    }
    return res;
  }

  inline DualVec operator*(const DualVec &rhs) const noexcept {
    DualVec res;
    res.v = v * rhs.v;
    std::size_t i = 0;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t v_a = vdupq_n_f32(v);
      float32x4_t v_b = vdupq_n_f32(rhs.v);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vfmaq_f32(vmulq_f32(vld1q_f32(&g[i]), v_b),
                                       vld1q_f32(&rhs.g[i]), v_a));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t v_a = vdupq_n_f64(v);
      float64x2_t v_b = vdupq_n_f64(rhs.v);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vfmaq_f64(vmulq_f64(vld1q_f64(&g[i]), v_b),
                                       vld1q_f64(&rhs.g[i]), v_a));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 v_a = _mm256_set1_ps(v);
      __m256 v_b = _mm256_set1_ps(rhs.v);
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(
            &res.g[i],
            _mm256_fmadd_ps(_mm256_load_ps(&g[i]), v_b,
                            _mm256_mul_ps(_mm256_load_ps(&rhs.g[i]), v_a)));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d v_a = _mm256_set1_pd(v);
      __m256d v_b = _mm256_set1_pd(rhs.v);
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(
            &res.g[i],
            _mm256_fmadd_pd(_mm256_load_pd(&g[i]), v_b,
                            _mm256_mul_pd(_mm256_load_pd(&rhs.g[i]), v_a)));
      }
    }
#endif

    for (; i < N; ++i) {
      res.g[i] = g[i] * rhs.v + v * rhs.g[i];
    }
    return res;
  }

  inline DualVec operator/(const DualVec &rhs) const noexcept {
    DualVec res;
    res.v = v / rhs.v;
    T inv_v2 = static_cast<T>(1.0) / (rhs.v * rhs.v);
    std::size_t i = 0;

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t v_a = vdupq_n_f32(v);
      float32x4_t v_b = vdupq_n_f32(rhs.v);
      float32x4_t v_inv = vdupq_n_f32(inv_v2);
      for (; i + 3 < N; i += 4) {
        float32x4_t num = vsubq_f32(vmulq_f32(vld1q_f32(&g[i]), v_b),
                                    vmulq_f32(vld1q_f32(&rhs.g[i]), v_a));
        vst1q_f32(&res.g[i], vmulq_f32(num, v_inv));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t v_a = vdupq_n_f64(v);
      float64x2_t v_b = vdupq_n_f64(rhs.v);
      float64x2_t v_inv = vdupq_n_f64(inv_v2);
      for (; i + 1 < N; i += 2) {
        float64x2_t num = vsubq_f64(vmulq_f64(vld1q_f64(&g[i]), v_b),
                                    vmulq_f64(vld1q_f64(&rhs.g[i]), v_a));
        vst1q_f64(&res.g[i], vmulq_f64(num, v_inv));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 v_a = _mm256_set1_ps(v);
      __m256 v_b = _mm256_set1_ps(rhs.v);
      __m256 v_inv = _mm256_set1_ps(inv_v2);
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(
            &res.g[i],
            _mm256_mul_ps(_mm256_fmsub_ps(_mm256_load_ps(&g[i]), v_b,
                                          _mm256_mul_ps(_mm256_load_ps(&rhs.g[i]), v_a)),
                          v_inv));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d v_a = _mm256_set1_pd(v);
      __m256d v_b = _mm256_set1_pd(rhs.v);
      __m256d v_inv = _mm256_set1_pd(inv_v2);
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(
            &res.g[i],
            _mm256_mul_pd(_mm256_fmsub_pd(_mm256_load_pd(&g[i]), v_b,
                                          _mm256_mul_pd(_mm256_load_pd(&rhs.g[i]), v_a)),
                          v_inv));
      }
    }
#endif

    for (; i < N; ++i) {
      res.g[i] = (g[i] * rhs.v - v * rhs.g[i]) * inv_v2;
    }
    return res;
  }

  inline DualVec operator+(T s) const noexcept {
    DualVec res = *this;
    res.v += s;
    return res;
  }
  inline DualVec operator-(T s) const noexcept {
    DualVec res = *this;
    res.v -= s;
    return res;
  }
  inline DualVec operator*(T s) const noexcept {
    DualVec res;
    res.v = v * s;
    std::size_t i = 0;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t vs = vdupq_n_f32(s);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vmulq_f32(vld1q_f32(&g[i]), vs));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t vs = vdupq_n_f64(s);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vmulq_f64(vld1q_f64(&g[i]), vs));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 vs = _mm256_set1_ps(s);
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&res.g[i], _mm256_mul_ps(_mm256_load_ps(&g[i]), vs));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d vs = _mm256_set1_pd(s);
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&res.g[i], _mm256_mul_pd(_mm256_load_pd(&g[i]), vs));
      }
    }
#endif
    for (; i < N; ++i) {
      res.g[i] = g[i] * s;
    }
    return res;
  }

  inline DualVec operator/(T s) const noexcept {
    DualVec res;
    res.v = v / s;
    T inv = static_cast<T>(1.0) / s;
    std::size_t i = 0;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t vinv = vdupq_n_f32(inv);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&res.g[i], vmulq_f32(vld1q_f32(&g[i]), vinv));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t vinv = vdupq_n_f64(inv);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&res.g[i], vmulq_f64(vld1q_f64(&g[i]), vinv));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 vinv = _mm256_set1_ps(inv);
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&res.g[i], _mm256_mul_ps(_mm256_load_ps(&g[i]), vinv));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d vinv = _mm256_set1_pd(inv);
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&res.g[i], _mm256_mul_pd(_mm256_load_pd(&g[i]), vinv));
      }
    }
#endif
    for (; i < N; ++i) {
      res.g[i] = g[i] * inv;
    }
    return res;
  }

  inline DualVec &operator+=(const DualVec &rhs) noexcept {
    *this = *this + rhs;
    return *this;
  }
  inline DualVec &operator-=(const DualVec &rhs) noexcept {
    *this = *this - rhs;
    return *this;
  }
  inline DualVec &operator*=(const DualVec &rhs) noexcept {
    *this = *this * rhs;
    return *this;
  }
  inline DualVec &operator/=(const DualVec &rhs) noexcept {
    *this = *this / rhs;
    return *this;
  }

  inline DualVec &operator+=(T s) noexcept {
    this->v += s;
    return *this;
  }
  inline DualVec &operator-=(T s) noexcept {
    this->v -= s;
    return *this;
  }
  inline DualVec &operator*=(T s) noexcept {
    *this = *this * s;
    return *this;
  }
  inline DualVec &operator/=(T s) noexcept {
    *this = *this / s;
    return *this;
  }

  inline void scale_gradients_from(const DualVec &src, T derivative) noexcept {
    std::size_t i = 0;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if constexpr (std::is_same_v<T, float>) {
      float32x4_t vd = vdupq_n_f32(derivative);
      for (; i + 3 < N; i += 4) {
        vst1q_f32(&g[i], vmulq_f32(vld1q_f32(&src.g[i]), vd));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      float64x2_t vd = vdupq_n_f64(derivative);
      for (; i + 1 < N; i += 2) {
        vst1q_f64(&g[i], vmulq_f64(vld1q_f64(&src.g[i]), vd));
      }
    }
#elif !defined(__CUDACC__) && defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      __m256 v_d = _mm256_set1_ps(derivative);
      for (; i + 7 < N; i += 8) {
        _mm256_store_ps(&g[i], _mm256_mul_ps(_mm256_load_ps(&src.g[i]), v_d));
      }
    } else if constexpr (std::is_same_v<T, double>) {
      __m256d v_d = _mm256_set1_pd(derivative);
      for (; i + 3 < N; i += 4) {
        _mm256_store_pd(&g[i], _mm256_mul_pd(_mm256_load_pd(&src.g[i]), v_d));
      }
    }
#endif
    for (; i < N; ++i) {
      g[i] = src.g[i] * derivative;
    }
  }
};

template <typename T, std::size_t N>
inline DualVec<T, N> operator+(T s, const DualVec<T, N> &d) noexcept {
  return d + s;
}
template <typename T, std::size_t N>
inline DualVec<T, N> operator-(T s, const DualVec<T, N> &d) noexcept {
  DualVec<T, N> res;
  res.v = s - d.v;
  for (std::size_t i = 0; i < N; ++i) {
    res.g[i] = -d.g[i];
  }
  return res;
}
template <typename T, std::size_t N>
inline DualVec<T, N> operator*(T s, const DualVec<T, N> &d) noexcept {
  return d * s;
}
template <typename T, std::size_t N>
inline DualVec<T, N> operator/(T s, const DualVec<T, N> &d) noexcept {
  DualVec<T, N> res;
  res.v = s / d.v;
  T factor = -s / (d.v * d.v);
  for (std::size_t i = 0; i < N; ++i) {
    res.g[i] = d.g[i] * factor;
  }
  return res;
}

} // namespace matrix
} // namespace Optimization

#endif // OPTIMIZATION_MATRIX_DUALVEC_HPP_
