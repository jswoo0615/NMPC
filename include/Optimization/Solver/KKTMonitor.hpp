#ifndef OPTIMIZATION_SOLVER_KKT_MONITOR_HPP_
#define OPTIMIZATION_SOLVER_KKT_MONITOR_HPP_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

// 플랫폼에 따른 네이티브 SIMD 명령어셋 적재
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Optimization {
namespace Solver {

/**
 * @brief KKT 조건 (Karush-Kuhn-Tucker Conditions) 고속 모니터링 클래스
 * @details EQP 및 최상위 티어 IPM(내점법) 솔버의 수렴성을 SIMD 가속으로 독립 검증합니다.
 * [Architect's Update]
 * std::initializer_list 스택 할당 오버헤드와 꼬리 루프 분기(Branch)를 완전히 제거하여
 * CPU 레지스터 단에서 평가가 종결되도록 극단적 최적화를 적용했습니다.
 */
template <std::size_t N_vars, std::size_t N_cons>
class KKTMonitor {
   public:
    static constexpr double TOLERANCE = 1e-6;

    struct KKT_Metrics {
        double stationarity_error;        // ||∇L||_inf (정류성 오차)
        double primal_feasibility_error;  // ||Au - b||_inf (동역학 실현가능성 결함)
        double dual_feasibility_error;    // min(0, λ) 위반량의 최댓값 (쌍대성 오차)
        double complementarity_error;     // ||S * λ - μ||_inf (상보성 여유 오차)
        bool is_optimal;                  // 4대 지표가 모두 TOLERANCE 이내인지 판결
    };

    /**
     * @brief [코어 엔진 1] 아키텍처 중심 고속 Infinity Norm 추출기
     */
    template <typename T, std::size_t N>
    static T fast_infinity_norm(const matrix::StaticVector<T, N>& vec) {
        std::size_t i = 0;
        T max_val = T(0.0);
        const T* data = vec.data_ptr();

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_max = vdupq_n_f32(0.0f);
            for (; i + 3 < N; i += 4) {
                float32x4_t v_val = vabsq_f32(vld1q_f32(&data[i]));
                v_max = vmaxq_f32(v_max, v_val);
            }
            // [Architect's Fix] 스택 메모리 경유(initializer_list) 차단
            max_val = std::max(std::max(vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1)),
                               std::max(vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)));
        } else if constexpr (std::is_same_v<T, double>) {
            float64x2_t v_max = vdupq_n_f64(0.0);
            for (; i + 1 < N; i += 2) {
                float64x2_t v_val = vabsq_f64(vld1q_f64(&data[i]));
                v_max = vmaxq_f64(v_max, v_val);
            }
            max_val = std::max(vgetq_lane_f64(v_max, 0), vgetq_lane_f64(v_max, 1));
        }
#elif !defined(__CUDACC__) && defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 v_max = _mm256_setzero_ps();
            for (; i + 7 < N; i += 8) {
                __m256 v_val = _mm256_loadu_ps(&data[i]);
                v_val = _mm256_and_ps(v_val, v_abs_mask);
                v_max = _mm256_max_ps(v_max, v_val);
            }
            __m128 hi = _mm256_extractf128_ps(v_max, 1);
            __m128 lo = _mm256_castps256_ps128(v_max);
            __m128 max128 = _mm_max_ps(hi, lo);
            __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 max64 = _mm_max_ps(max128, shuf);
            shuf = _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(2, 3, 0, 1));
            max_val = _mm_cvtss_f32(_mm_max_ps(max64, shuf));
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d v_max = _mm256_setzero_pd();
            for (; i + 3 < N; i += 4) {
                __m256d v_val = _mm256_loadu_pd(&data[i]);
                v_val = _mm256_and_pd(v_val, v_abs_mask);
                v_max = _mm256_max_pd(v_max, v_val);
            }
            __m128d hi = _mm256_extractf128_pd(v_max, 1);
            __m128d lo = _mm256_castpd256_pd128(v_max);
            __m128d max128 = _mm_max_pd(hi, lo);
            __m128d shuf = _mm_shuffle_pd(max128, max128, 1);
            max_val = _mm_cvtsd_f64(_mm_max_pd(max128, shuf));
        }
#endif

        // 잔여 자투리 처리 (Branchless)
        for (; i < N; ++i) {
            max_val = std::max(max_val, std::abs(data[i]));
        }
        return max_val;
    }

    /**
     * @brief [코어 엔진 2] 아키텍처 중심 상보성 여유 오차 고속 추출기
     */
    template <typename T, std::size_t N>
    static T fast_complementarity_norm(const matrix::StaticVector<T, N>& s_vec,
                                       const matrix::StaticVector<T, N>& lambda_vec, T target_mu) {
        std::size_t i = 0;
        T max_val = T(0.0);
        const T* s_data = s_vec.data_ptr();
        const T* l_data = lambda_vec.data_ptr();

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_mu = vdupq_n_f32(target_mu);
            float32x4_t v_max = vdupq_n_f32(0.0f);
            for (; i + 3 < N; i += 4) {
                float32x4_t v_s = vld1q_f32(&s_data[i]);
                float32x4_t v_l = vld1q_f32(&l_data[i]);
                float32x4_t v_prod = vmulq_f32(v_s, v_l);
                float32x4_t v_diff = vsubq_f32(v_prod, v_mu);
                float32x4_t v_abs = vabsq_f32(v_diff);
                v_max = vmaxq_f32(v_max, v_abs);
            }
            max_val = std::max(std::max(vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1)),
                               std::max(vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)));
        } else if constexpr (std::is_same_v<T, double>) {
            float64x2_t v_mu = vdupq_n_f64(target_mu);
            float64x2_t v_max = vdupq_n_f64(0.0);
            for (; i + 1 < N; i += 2) {
                float64x2_t v_s = vld1q_f64(&s_data[i]);
                float64x2_t v_l = vld1q_f64(&l_data[i]);
                float64x2_t v_prod = vmulq_f64(v_s, v_l);
                float64x2_t v_diff = vsubq_f64(v_prod, v_mu);
                float64x2_t v_abs = vabsq_f64(v_diff);
                v_max = vmaxq_f64(v_max, v_abs);
            }
            max_val = std::max(vgetq_lane_f64(v_max, 0), vgetq_lane_f64(v_max, 1));
        }
#elif !defined(__CUDACC__) && defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_mu = _mm256_set1_ps(target_mu);
            __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 v_max = _mm256_setzero_ps();
            for (; i + 7 < N; i += 8) {
                __m256 v_s = _mm256_loadu_ps(&s_data[i]);
                __m256 v_l = _mm256_loadu_ps(&l_data[i]);
                __m256 v_prod = _mm256_mul_ps(v_s, v_l);
                __m256 v_diff = _mm256_sub_ps(v_prod, v_mu);
                __m256 v_abs = _mm256_and_ps(v_diff, v_abs_mask);
                v_max = _mm256_max_ps(v_max, v_abs);
            }
            __m128 hi = _mm256_extractf128_ps(v_max, 1);
            __m128 lo = _mm256_castps256_ps128(v_max);
            __m128 max128 = _mm_max_ps(hi, lo);
            __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 max64 = _mm_max_ps(max128, shuf);
            shuf = _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(2, 3, 0, 1));
            max_val = _mm_cvtss_f32(_mm_max_ps(max64, shuf));
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d v_mu = _mm256_set1_pd(target_mu);
            __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d v_max = _mm256_setzero_pd();
            for (; i + 3 < N; i += 4) {
                __m256d v_s = _mm256_loadu_pd(&s_data[i]);
                __m256d v_l = _mm256_loadu_pd(&l_data[i]);
                __m256d v_prod = _mm256_mul_pd(v_s, v_l);
                __m256d v_diff = _mm256_sub_pd(v_prod, v_mu);
                __m256d v_abs = _mm256_and_pd(v_diff, v_abs_mask);
                v_max = _mm256_max_pd(v_max, v_abs);
            }
            __m128d hi = _mm256_extractf128_pd(v_max, 1);
            __m128d lo = _mm256_castpd256_pd128(v_max);
            __m128d max128 = _mm_max_pd(hi, lo);
            __m128d shuf = _mm_shuffle_pd(max128, max128, 1);
            max_val = _mm_cvtsd_f64(_mm_max_pd(max128, shuf));
        }
#endif

        for (; i < N; ++i) {
            max_val = std::max(max_val, std::abs(s_data[i] * l_data[i] - target_mu));
        }
        return max_val;
    }

    /**
     * @brief [코어 엔진 3] 아키텍처 중심 쌍대 실현 가능성 오차 고속 추출기
     */
    template <typename T, std::size_t N>
    static T fast_dual_feasibility_norm(const matrix::StaticVector<T, N>& lambda_vec) {
        std::size_t i = 0;
        T max_viol = T(0.0);
        const T* l_data = lambda_vec.data_ptr();

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_zero = vdupq_n_f32(0.0f);
            float32x4_t v_max_viol = vdupq_n_f32(0.0f);
            for (; i + 3 < N; i += 4) {
                float32x4_t v_l = vld1q_f32(&l_data[i]);
                float32x4_t v_viol = vsubq_f32(v_zero, v_l);
                v_viol = vmaxq_f32(v_zero, v_viol);
                v_max_viol = vmaxq_f32(v_max_viol, v_viol);
            }
            max_viol =
                std::max(std::max(vgetq_lane_f32(v_max_viol, 0), vgetq_lane_f32(v_max_viol, 1)),
                         std::max(vgetq_lane_f32(v_max_viol, 2), vgetq_lane_f32(v_max_viol, 3)));
        } else if constexpr (std::is_same_v<T, double>) {
            float64x2_t v_zero = vdupq_n_f64(0.0);
            float64x2_t v_max_viol = vdupq_n_f64(0.0);
            for (; i + 1 < N; i += 2) {
                float64x2_t v_l = vld1q_f64(&l_data[i]);
                float64x2_t v_viol = vsubq_f64(v_zero, v_l);
                v_viol = vmaxq_f64(v_zero, v_viol);
                v_max_viol = vmaxq_f64(v_max_viol, v_viol);
            }
            max_viol = std::max(vgetq_lane_f64(v_max_viol, 0), vgetq_lane_f64(v_max_viol, 1));
        }
#elif !defined(__CUDACC__) && defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_zero = _mm256_setzero_ps();
            __m256 v_max_viol = _mm256_setzero_ps();
            for (; i + 7 < N; i += 8) {
                __m256 v_l = _mm256_loadu_ps(&l_data[i]);
                __m256 v_viol = _mm256_sub_ps(v_zero, v_l);
                v_viol = _mm256_max_ps(v_zero, v_viol);
                v_max_viol = _mm256_max_ps(v_max_viol, v_viol);
            }
            __m128 hi = _mm256_extractf128_ps(v_max_viol, 1);
            __m128 lo = _mm256_castps256_ps128(v_max_viol);
            __m128 max128 = _mm_max_ps(hi, lo);
            __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 max64 = _mm_max_ps(max128, shuf);
            shuf = _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(2, 3, 0, 1));
            max_viol = _mm_cvtss_f32(_mm_max_ps(max64, shuf));
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d v_zero = _mm256_setzero_pd();
            __m256d v_max_viol = _mm256_setzero_pd();
            for (; i + 3 < N; i += 4) {
                __m256d v_l = _mm256_loadu_pd(&l_data[i]);
                __m256d v_viol = _mm256_sub_pd(v_zero, v_l);
                v_viol = _mm256_max_pd(v_zero, v_viol);
                v_max_viol = _mm256_max_pd(v_max_viol, v_viol);
            }
            __m128d hi = _mm256_extractf128_pd(v_max_viol, 1);
            __m128d lo = _mm256_castpd256_pd128(v_max_viol);
            __m128d max128 = _mm_max_pd(hi, lo);
            __m128d shuf = _mm_shuffle_pd(max128, max128, 1);
            max_viol = _mm_cvtsd_f64(_mm_max_pd(max128, shuf));
        }
#endif

        for (; i < N; ++i) {
            max_viol = std::max(max_viol, std::max(T(0.0), -l_data[i]));
        }
        return max_viol;
    }

    /**
     * @brief 기존 EQP 검증 인터페이스
     */
    static KKT_Metrics evaluate_EQP(const matrix::StaticMatrix<double, N_vars, N_vars>& P,
                                    const matrix::StaticVector<double, N_vars>& q,
                                    const matrix::StaticMatrix<double, N_cons, N_vars>& A,
                                    const matrix::StaticVector<double, N_cons>& b,
                                    const matrix::StaticVector<double, N_vars>& u_opt,
                                    const matrix::StaticVector<double, N_cons>& lambda_opt) {
        KKT_Metrics metrics;
        metrics.dual_feasibility_error = 0.0;
        metrics.complementarity_error = 0.0;

        matrix::StaticVector<double, N_vars> grad_L = q;
        matrix::StaticVector<double, N_vars> Pu;
        linalg::multiply(P, u_opt, Pu);
        grad_L += Pu;

        matrix::StaticVector<double, N_vars> AT_lambda;
        linalg::multiply_AT_B(A, lambda_opt, AT_lambda);
        grad_L += AT_lambda;

        metrics.stationarity_error = fast_infinity_norm(grad_L);

        matrix::StaticVector<double, N_cons> eq_res;
        linalg::multiply(A, u_opt, eq_res);
        eq_res.saxpy(-1.0, b);

        metrics.primal_feasibility_error = fast_infinity_norm(eq_res);

        metrics.is_optimal = (metrics.stationarity_error <= TOLERANCE) &&
                             (metrics.primal_feasibility_error <= TOLERANCE);

        return metrics;
    }

    /**
     * @brief 풀 스택 비선형 IPM(내점법) KKT 통합 검증기
     */
    template <std::size_t N_ineq>
    static KKT_Metrics evaluate_IPM(const matrix::StaticVector<double, N_vars>& grad_L_res,
                                    const matrix::StaticVector<double, N_cons>& primal_res,
                                    const matrix::StaticVector<double, N_ineq>& s_ineq,
                                    const matrix::StaticVector<double, N_ineq>& lambda_ineq,
                                    double target_mu) {
        KKT_Metrics metrics;

        metrics.stationarity_error = fast_infinity_norm(grad_L_res);
        metrics.primal_feasibility_error = fast_infinity_norm(primal_res);
        metrics.dual_feasibility_error = fast_dual_feasibility_norm(lambda_ineq);
        metrics.complementarity_error = fast_complementarity_norm(s_ineq, lambda_ineq, target_mu);

        metrics.is_optimal = (metrics.stationarity_error <= TOLERANCE) &&
                             (metrics.primal_feasibility_error <= TOLERANCE) &&
                             (metrics.dual_feasibility_error <= TOLERANCE) &&
                             (metrics.complementarity_error <= TOLERANCE);

        return metrics;
    }

    static void print_metrics(const KKT_Metrics& metrics) {
        std::cout << "========== [ KKT Monitor : Accelerated ] ==========\n";
        std::cout << "[1] Stationarity (∇L) Error : " << metrics.stationarity_error << "\n";
        std::cout << "[2] Primal Feasibility Error: " << metrics.primal_feasibility_error << "\n";
        std::cout << "[3] Dual Feasibility Error  : " << metrics.dual_feasibility_error << "\n";
        std::cout << "[4] Complementarity Error   : " << metrics.complementarity_error << "\n";
        std::cout << ">>> Status: " << (metrics.is_optimal ? "OPTIMAL ✅" : "VIOLATED ❌") << "\n";
        std::cout << "===================================================\n\n";
    }
};


}  // namespace Solver

// 하위 호환성 유지를 위한 Alias 추가
using Solver::KKTMonitor;

}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_KKT_MONITOR_HPP_