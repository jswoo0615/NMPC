#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_
#define OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_

#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include <cmath>
#include <algorithm>
#include <type_traits>

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    #include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace Optimization {
    namespace solver {
        /**
         * @brief 고속 Levenberg-Marquardt (LM) 비선형 최소자승 솔버
         * @details
         * SIMD 헤시안 어셈블리 과정에서 Column-Major 메모리를 Row 방향으로 읽던 캐시 결함을 
         * H(j, i) 연속 기입 구조로 뒤집어 SIMD 파이프라인 효울을 100%로 상승
         * 템플릿 타입 T 도입 및 if constexpr 분기 적용으로 SIMD 하드웨어 충돌 방지
         */
        template <typename T, std::size_t N, std::size_t M, typename Functor>
        inline SolverStatus solve_LM(matrix::StaticVector<T, N>& x_opt,
                                     const Functor& calc_residuals,
                                     int max_iter = 50,
                                     T tol = T(1e-6),
                                     T initial_lambda = T(1e-3)) {
            static_assert(M >= N, "Residual M must be greater than or equal to variables N");
            static_assert(std::is_floating_point_v<T>, "Type T must be a floating-point type");

            using ADVar = DualVar<T, N>;
            T lambda = initial_lambda;

            matrix::StaticVector<T, N> x_new;
            matrix::StaticMatrix<T, N, N> H;
            matrix::StaticVector<T, N> g;
            matrix::StaticVector<T, N> dx;
            matrix::StaticVector<ADVar, N> x_dual;
            matrix::StaticVector<ADVar, N> x_new_dual;

            T current_cost = T(0.0);

            for (int iter = 0; iter < max_iter; ++iter) {
                for (std::size_t i = 0; i < N; ++i) {
                    x_dual(i) = ADVar::make_variable(x_opt(i), i);
                }
                
                matrix::StaticVector<ADVar, M> residuals = calc_residuals(x_dual);

                current_cost = T(0.0);
                for (std::size_t k = 0; k < M; ++k) {
                    T r_val = get_value(residuals(k));
                    current_cost += T(0.5) * r_val * r_val;
                }

                H.set_zero();
                g.set_zero();

                for (std::size_t k = 0; k < M; ++k) {
                    const T r_val = residuals(k).v;
                    const T* J_k = residuals(k).g;

                    for (std::size_t i = 0; i < N; ++i) {
                        const T J_ki = J_k[i];
                        g(i) += J_ki * r_val;

                        std::size_t j = i;

                        // H(j, i) 연속 쓰기로 Column-Major SIMD 정렬 보장
                        #if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                            if constexpr (std::is_same_v<T, float>) {
                                float32x4_t v_jki = vdupq_n_f32(J_ki);
                                for (; i + 3 < N; j += 4) {
                                    float32x4_t v_jkj = vld1q_f32(&H_k[j]);
                                    float32x4_t v_h = vld1q_f32(&H(j, i));
                                    vst1q_f32(&H(j, i), vfmaq_f32(v_h, v_jki, v_jkj));
                                }
                            } else if constexpr (std::is_same_v<T, double>) {
                                float64x2_t v_jki = vdupq_n_f64(J_ki);
                                for (; j + 1 < N; j += 2) {
                                    float64x2_t v_jkj = vld1q_f64(&J_k[j]);
                                    float64x2_t v_h = vld1q_f64(&H(j, i));
                                    vst1q_f64(&H(j, i), vfmaq_f64(v_h, v_jki, v_jkj));
                                }
                            }
                        #elif !defined(__CUDACC__) && defined(__AVX2__)
                            if constexpr (std::is_same_v<T, float>) {
                                __m256 v_jki = _mm256_set1_ps(J_ki);
                                for (; j + 7 < N; j += 8) {
                                    __m256 v_jkj = _mm256_loadu_ps(&J_k[j]);
                                    __m256 v_h = _mm256_loadu_ps(&H(j, i));
                                    // AVX2 FMA : (J_ki * J_kj) + H
                                    _mm256_storeu_ps(&H(j, i), _mm256_fmadd_ps(v_jki, v_jkj, v_h));
                                }
                            } else if constexpr (std::is_same_v<T, double>) {
                                __m256d v_jki = _mm256_set1_pd(J_ki);
                                for (; j + 3 < N; j += 4) {
                                    __m256d v_jkj = _mm256_loadu_pd(&J_k[j]);
                                    __m256d v_h = _mm256_loadu_pd(&H(j, i));
                                    // AVX2 FMA : (J_ki * J_kj) + H
                                    _mm256_storeu_pd(&H(j, i), _mm256_fmadd_pd(v_jki, v_jkj, v_h));
                                }
                            }
                        #endif

                        // SIMD 처리 후 남은 스칼라 루프 처리
                        for (; j < N; ++j) {
                            H(j, i) += J_ki * J_k[j];
                        }
                    }
                }

                // H 대칭성 복구 (하삼각에 쓰인 값을 상삼각으로 복사)
                for (std::size_t i = 0; i < N; ++i) {
                    for (std::size_t j = i + 1; j < N; ++j) {
                        H(i, j) = H(j, i);
                    }
                }

                T g_max = T(0.0);
                for (std::size_t i = 0; i < N; ++i) {
                    T abs_q = std::abs(g(i));
                    if (abs_q > g_max) {
                        g_max = abs_q;
                    }
                }

                if (g_max < tol) {
                    return SolverStatus::SUCCESS;
                }

                bool step_accepted = false;
                for (int retry = 0; retry < 10; ++retry) {
                    matrix::StaticMatrix<T, N, N> H_damped = H;

                    // LM 정착화 (Marquardt 전략 적용)
                    for (std::size_t i = 0; i < N; ++i) {
                        H_damped(i, i) += lambda * (H(i, i) + T(1e-6));
                    }

                    if (linalg::LDLT_decompose(H_damped) == MathStatus::SUCCESS) {
                        matrix::StaticVector<T, N> neg_g;
                        for (std::size_t i = 0; i < N; ++i) {
                            neg_g(i) = -g(i);
                        }
                        linalg::LDLT_solve(H_damped, neg_g, dx);
                    } else {
                        lambda *= T(10.0);
                        continue;
                    }

                    x_new = x_opt + dx;

                    for (std::size_t i = 0; i < N; ++i) {
                        x_new_dual(i) = ADVar(x_new(i));
                    }

                    matrix::StaticVector<ADVar, M> new_residuals = calc_residuals(x_new_dual);

                    T new_cost = T(0.0);
                    for (std::size_t k = 0; k < M; ++k) {
                        T r_val = get_value(new_residuals(k));
                        new_cost += T(0.5) * r_val * r_val;
                    }

                    if (new_cost < current_cost) {
                        x_opt = x_new;
                        lambda = std::max(T(1e-7), lambda / T(10.0));
                        step_accepted = true;
                        break;
                    } else {
                        lambda *= T(10.0);
                    }
                }

                if (!step_accepted) {
                    return SolverStatus::MAX_ITERATION_REACHED;
                }
            }
            return SolverStatus::MAX_ITERATION_REACHED;
        }
    } // namespace solver
} // namespace Optimization



#endif // OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_