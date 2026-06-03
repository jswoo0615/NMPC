#ifndef OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_
#define OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_

#include <array>
#include <cmath>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

// 사용자 정의 Dual 타입 (모델의 템플릿 호환성을 위해)
using Optimization::Dual;

namespace Optimization {
namespace control {

/**
 * @brief NMPC (Nonlinear Model Predictive Control) 메인 컨트롤러
 * @details [Architect's Design]
 * - 다중 사격법(Multiple Shooting) 기반 무제약 SQP / DDP 아우터 루프.
 * - 동적 메모리 할당(Heap)을 완전 배제한 정적 파이프라인.
 * @tparam Model 동역학 모델 펑터 (RealTimeDynamicsModel 또는 HighFidelityDynamicsModel)
 * @tparam H 예측 구간 (Horizon)
 * @tparam Nx 상태 벡터 차원 (기본 8)
 * @tparam Nu 제어 입력 차원 (기본 2)
 */
template <typename Model, size_t H, size_t Nx = 8, size_t Nu = 2>
class NMPC_Controller {
   public:
    // =========================================================================
    // [1] 핵심 코어 부품
    // =========================================================================
    Model dynamics_model;
    solver::RiccatiSolver<H, Nx, Nu> riccati;

    // =========================================================================
    // [2] 튜닝 파라미터 (가중치 행렬)
    // =========================================================================
    matrix::StaticMatrix<double, Nx, Nx> Q_weight;
    matrix::StaticMatrix<double, Nx, Nx> Q_terminal;
    matrix::StaticMatrix<double, Nu, Nu> R_weight;

    // =========================================================================
    // [3] 내부 상태 궤적 (Current Guess) 및 목표 궤적 (Reference)
    // =========================================================================
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_guess;
    std::array<matrix::StaticVector<double, Nu>, H> u_guess;
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;

    // 초기화 함수
    void init(const matrix::StaticMatrix<double, Nx, Nx>& Q,
              const matrix::StaticMatrix<double, Nx, Nx>& Q_term,
              const matrix::StaticMatrix<double, Nu, Nu>& R) {
        Q_weight = Q;
        Q_terminal = Q_term;
        R_weight = R;

        for (size_t k = 0; k <= H; ++k) {
            x_guess[k].set_zero();
            x_ref[k].set_zero();
            if (k < H) u_guess[k].set_zero();
        }
    }

    /**
     * @brief 목표 궤적 주입 (Local Planner로부터 수신)
     */
    void set_reference_trajectory(const std::array<matrix::StaticVector<double, Nx>, H + 1>& ref) {
        x_ref = ref;
    }

    // =========================================================================
    // [4] SQP 아우터 루프 (Outer-Loop)
    // =========================================================================
    /**
     * @brief NMPC 최적화 수행
     * @param x0 현재 차량의 관측 상태 (Initial State)
     * @param max_iter SQP 최대 반복 횟수
     * @param tol 수렴 판정 허용 오차
     * @return 계산된 최적의 첫 번째 제어 입력
     */
    matrix::StaticVector<double, Nu> compute_control(const matrix::StaticVector<double, Nx>& x0, 
                                                     int max_iter = 5, 
                                                     double tol = 1e-4) {
        // 1. 초기 조건 강제 할당 (다중 사격법의 0번 노드는 항상 현재 상태)
        x_guess[0] = x0;

        for (int iter = 0; iter < max_iter; ++iter) {
            double max_defect = 0.0;

            // -----------------------------------------------------------------
            // Phase 1: 선형화 및 KKT 시스템 조립 (Evaluation & Linearization)
            // -----------------------------------------------------------------
            for (size_t k = 0; k < H; ++k) {
                // 1-1. 상태 1차항 가중치 (Gradient): q_k = Q * (x_guess_k - x_ref_k)
                matrix::StaticVector<double, Nx> x_err;
                for(size_t i = 0; i < Nx; ++i) {
                    x_err(i) = x_guess[k](i) - x_ref[k](i);
                }
                linalg::multiply(Q_weight, x_err, riccati.q[k]);
                riccati.Q[k] = Q_weight;

                // 1-2. 제어 1차항 가중치 (Gradient): r_k = R * u_guess_k
                linalg::multiply(R_weight, u_guess[k], riccati.r[k]);
                riccati.R[k] = R_weight;

                // 1-3. 자동 미분(Dual)을 통한 해석적 야코비안 (A_k, B_k) 및 잔차(d_k) 추출
                // A 행렬 (상태 야코비안) 추출
                for (size_t i = 0; i < Nx; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) {
                        x_dual(j) = Dual<double>(x_guess[k](j), (i == j) ? 1.0 : 0.0);
                    }
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) {
                        u_dual(j) = Dual<double>(u_guess[k](j), 0.0);
                    }

                    auto x_next_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        riccati.A[k](j, i) = x_next_dual(j).d; // 미분값(야코비안) 저장
                    }
                }

                // B 행렬 (입력 야코비안) 추출 및 동역학 잔차(Defect) 추출
                for (size_t i = 0; i < Nu; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) {
                        x_dual(j) = Dual<double>(x_guess[k](j), 0.0);
                    }
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) {
                        u_dual(j) = Dual<double>(u_guess[k](j), (i == j) ? 1.0 : 0.0);
                    }

                    auto x_next_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        riccati.B[k](j, i) = x_next_dual(j).d; // 미분값(야코비안) 저장
                    }

                    // i == 0 일 때 1회만 물리 법칙 비선형 평가 결과(v)를 가져와 잔차 계산
                    if (i == 0) {
                        for (size_t j = 0; j < Nx; ++j) {
                            // d_k = f(x_k, u_k) - x_{k+1}
                            riccati.d[k](j) = x_next_dual(j).v - x_guess[k + 1](j);
                            
                            double abs_d = std::abs(riccati.d[k](j));
                            if (abs_d > max_defect) max_defect = abs_d;
                        }
                    }
                }
            }

            // 종단(Terminal) 비용 설정
            matrix::StaticVector<double, Nx> x_err_H;
            for(size_t i = 0; i < Nx; ++i) x_err_H(i) = x_guess[H](i) - x_ref[H](i);
            linalg::multiply(Q_terminal, x_err_H, riccati.q[H]);
            riccati.Q[H] = Q_terminal;

            // -----------------------------------------------------------------
            // Phase 2: 이너 루프 (Riccati Solver 실행)
            // -----------------------------------------------------------------
            // 정칙화(Regularization) 주입으로 고유값 붕괴 방어
            SolverStatus status = riccati.solve(1e-4, 1e-6);
            if (status != SolverStatus::SUCCESS) {
                // 수치적 발산 (LDLT 실패) 발생 시 궤환 중단, 안전한 이전 제어 반환 (Fallback)
                return u_guess[0];
            }

            // -----------------------------------------------------------------
            // Phase 3: 궤적 갱신 (Trajectory Update with Backtracking Line Search 뼈대)
            // -----------------------------------------------------------------
            double alpha = 1.0; // 향후 Merit Function 기반 감쇠를 위한 변수
            double max_dx = 0.0;

            for (size_t k = 0; k < H; ++k) {
                for (size_t i = 0; i < Nx; ++i) {
                    double step_x = alpha * riccati.dx[k + 1](i);
                    x_guess[k + 1](i) += step_x;
                    
                    double abs_dx = std::abs(step_x);
                    if (abs_dx > max_dx) max_dx = abs_dx;
                }
                for (size_t i = 0; i < Nu; ++i) {
                    u_guess[k](i) += alpha * riccati.du[k](i);
                }
            }

            // -----------------------------------------------------------------
            // Phase 4: 수렴 판정 (Convergence Check)
            // -----------------------------------------------------------------
            // 다중 사격법의 동역학 위반량(Defect)과 보폭(dx)이 모두 허용치 이내면 조기 종료
            if (max_defect < tol && max_dx < tol) {
                break;
            }
        } // End of max_iter loop

        return u_guess[0];
    } // End of compute_control function

    /**
     * @brief 다음 제어 주기를 위한 Warm-Start (궤적 한 스텝 당기기)
     */
    void shift_trajectory() {
        for (size_t k = 0; k < H - 1; ++k) {
            x_guess[k] = x_guess[k + 1];
            u_guess[k] = u_guess[k + 1];
        }
        x_guess[H - 1] = x_guess[H];
        u_guess[H - 1].set_zero();
    }
};

}  // namespace control
}  // namespace Optimization

#endif  // OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_