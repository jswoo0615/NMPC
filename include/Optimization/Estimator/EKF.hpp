#ifndef OPTIMIZATION_EKF_HPP_
#define OPTIMIZATION_EKF_HPP_

#include <cmath>

#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Simulation/Integrator.hpp"

#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#elif !defined(__CUDACC__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Optimization {
namespace estimator {

/**
 * @brief 확장 칼만 필터(Extended Kalman Filter, EKF) 클래스
 * @details [Architect's Update]
 * O(N^3) 스칼라 역행렬 연산 폐기 및 .transpose() 복사 할당을 완전히 소거했습니다.
 * 공분산 예측(FPF^T) 시 전치 행렬을 만들지 않고 SIMD SAXPY로 직결하여 WCET를 방어합니다.
 */
template <size_t Nx = 8, size_t Nu = 2>
class EKF {
   public:
    matrix::StaticVector<double, Nx> x_est;
    matrix::StaticMatrix<double, Nx, Nx> P;
    matrix::StaticMatrix<double, Nx, Nx> Q;
    matrix::StaticMatrix<double, Nx, Nx> R;

    EKF() {
        x_est.set_zero();

        P.set_zero();
        for (size_t i = 0; i < Nx; ++i) P(i, i) = 1.0;

        Q.set_zero();
        for (size_t i = 0; i < Nx; ++i) Q(i, i) = 0.01;

        R.set_zero();
        for (size_t i = 0; i < Nx; ++i) R(i, i) = 0.1;
        if constexpr (Nx >= 6) {
            R(0, 0) = 0.2;   // s
            R(1, 1) = 0.2;   // d
            R(2, 2) = 0.05;  // mu
            R(3, 3) = 0.1;   // vx
            R(4, 4) = 0.1;   // vy
            R(5, 5) = 0.1;   // r
        }
        if constexpr (Nx >= 8) {
            R(6, 6) = 0.1;   // alpha_f
            R(7, 7) = 0.1;   // alpha_r
        }
    }

    template <typename Model>
    void predict(Model& model, const matrix::StaticVector<double, Nu>& u, double dt) {
        x_est = integrator::IntegratorEngine<Nx, Nu, Model, double>::compute(model, x_est, u, dt);

        matrix::StaticMatrix<double, Nx, Nx> F;
        using ADVar = DualVec<double, Nx>;
        matrix::StaticVector<ADVar, Nx> x_dual;
        matrix::StaticVector<ADVar, Nu> u_dual;

        for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(x_est(i), i);
        for (size_t i = 0; i < Nu; ++i) u_dual(i) = ADVar(u(i));

        matrix::StaticVector<ADVar, Nx> x_next_dual =
            integrator::IntegratorEngine<Nx, Nu, Model, ADVar>::compute(model, x_dual, u_dual, dt);

        // 자코비안 메모리 캐시 정렬 (Column-Major)
        for (size_t j = 0; j < Nx; ++j) {
            for (size_t i = 0; i < Nx; ++i) {
                F(i, j) = x_next_dual(i).g[j];
            }
        }

        // P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        matrix::StaticMatrix<double, Nx, Nx> FP = F * P;

        // =====================================================================
        // [Architect's Update] Zero-Copy SIMD SAXPY (A * B^T 가속)
        // 전치 행렬(F.transpose()) 메모리 할당을 회피하고 루프를 물리적으로 전개합니다.
        // =====================================================================
        P = Q;  // P = Q + FP * F^T
        for (size_t j = 0; j < Nx; ++j) {
            for (size_t k = 0; k < Nx; ++k) {
                double f_jk = F(j, k);
                size_t i = 0;
#if !defined(__CUDACC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                float64x2_t v_fjk = vdupq_n_f64(f_jk);
                for (; i + 1 < Nx; i += 2) {
                    float64x2_t v_fp = vld1q_f64(&FP(i, k));
                    float64x2_t v_p = vld1q_f64(&P(i, j));
                    vst1q_f64(&P(i, j), vfmaq_f64(v_p, v_fp, v_fjk));
                }
#elif !defined(__CUDACC__) && defined(__AVX2__)
                __m256d v_fjk = _mm256_set1_pd(f_jk);
                for (; i + 3 < Nx; i += 4) {
                    __m256d v_fp = _mm256_loadu_pd(&FP(i, k));
                    __m256d v_p = _mm256_loadu_pd(&P(i, j));
                    _mm256_storeu_pd(&P(i, j), _mm256_fmadd_pd(v_fp, v_fjk, v_p));
                }
#endif
                for (; i < Nx; ++i) {
                    P(i, j) += FP(i, k) * f_jk;
                }
            }
        }
    }

    void update(const matrix::StaticVector<double, Nx>& z) {
        matrix::StaticMatrix<double, Nx, Nx> S = P + R;
        matrix::StaticMatrix<double, Nx, Nx> S_decomp = S;

        if (linalg::LDLT_decompose(S_decomp) != MathStatus::SUCCESS) {
            return;
        }

        matrix::StaticMatrix<double, Nx, Nx> S_inv;
        for (size_t i = 0; i < Nx; ++i) {
            matrix::StaticVector<double, Nx> e;
            e.set_zero();
            e(i) = 1.0;

            matrix::StaticVector<double, Nx> col_inv;
            linalg::LDLT_solve(S_decomp, e, col_inv);

            for (size_t j = 0; j < Nx; ++j) {
                S_inv(j, i) = col_inv(j);
            }
        }

        matrix::StaticMatrix<double, Nx, Nx> K = P * S_inv;
        matrix::StaticVector<double, Nx> y = z - x_est;

        if (y(2) > M_PI)
            y(2) -= 2.0 * M_PI;
        else if (y(2) < -M_PI)
            y(2) += 2.0 * M_PI;

        x_est = x_est + (K * y);

        matrix::StaticMatrix<double, Nx, Nx> I;
        I.set_zero();
        for (size_t i = 0; i < Nx; ++i) I(i, i) = 1.0;

        P = (I - K) * P;

        for (size_t j = 0; j < Nx; ++j) {
            for (size_t i = j + 1; i < Nx; ++i) {
                double sym = 0.5 * (P(i, j) + P(j, i));
                P(i, j) = sym;
                P(j, i) = sym;
            }
        }
    }
};

}  // namespace estimator
}  // namespace Optimization

#endif  // OPTIMIZATION_EKF_HPP_