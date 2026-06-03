#ifndef OPTIMIZATION_SOLVER_RICCATI_SOLVER_HPP_
#define OPTIMIZATION_SOLVER_RICCATI_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
    namespace solver {
        /**
         * @brief 고속 이산시간 리카티 재귀 솔버 (Discrete-time Riccati Recursion Solver)
         * @details
         * linalg::multiply_AT_B 하드웨어 가속기로 직접 연결하여 O(H * N^3) 복사 지연 제거
         * @tparam H 예측 구간 (Horizon) 길이 (총 스텝 수)
         * @tparam Nx 상태 (State) 벡터의 차원 (예: 차량의 x, y, yaw, v 등)
         * @tparam Nu 제어 (Control) 벡터의 차원 (예 : 가속도, 조향각 등)
         */
        template <std::size_t M, std::size_t Nx, std::size_t Nu>
        class RiccatiSolver {
            public:
                /**
                 * @brief [1] 선형화된 시스템 동역학 (Linearized System Dynamics)
                 * @details
                 * 모델 수식 : x_{k+1} = A_{k} * x_{k} + B_{k} * u_{k} + d_{k}
                 */

                 /** @brief 상태 천이 행렬 (State Transition Matrix), 크기 : Nx x Nx */
                std::array<matrix::StaticMatrix<double, Nx, Nx>, H> A;
                /** @brief 입력 행렬 (Control Input Matrix), 크기 : Nx x Nu */
                std::array<matrix::StaticMatrix<double, Nx, Nu>, H> B;
                /** @brief 동역학 상수항 및 잔차 (Affine Residual Vector), 크기 : Nx */
                std::array<matrix::StaticVector<double, Nx>, H> d;

                /**
                 * @brief [2] 2차 비용 함수 매개변수 (Quadratic Cost Parameters)
                 * @details
                 * 수식 : J = (1/2) * x_{H}^{T} * Q_{H} * x_{H} + q_{H}^{T} * x_{H}
                 *            + sum_{k=0}^{H-1} [(1/2) * x_{k}^{T} * Q_{k} * x_{k} 
                 *            + q_{k}^{T} * x_{k} + (1/2) * u_{k}^{T} * R_{k} * u_{k} 
                 *            + r_{k}^{T} * u_{k}]
                 */

                /** @brief 상태 가중치 행렬 (State Cost Matrix), 크기 : Nx x Nx (스텝 (0 ~ H)) */
                std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1> Q;
                /** @brief 제어 입력 가중치 행렬 (Control Cost Matrix), 크기 : Nu x Nu (스텝 0 ~ H-1) */
                std::array<matrix::StaticMatrix<double, Nu, Nu>, H> R;
                /** @brief 상태 1차항 가중치 (State Linear Cost Vector), 크기 : Nx (스텝 0 ~ H) */
                std::array<matrix::StaticVector<double, Nx>, H + 1> q;
                /** @brief 제어 입력 1차항 가중치 (Control Linear Cost Vector), 크기 : Nu (스텝 0 ~ H-1) */
                std::array<matrix::StaticVector<double, Nu>, H> r;

                /**
                 * @brief [3] Value Function (Cost-to-Go) 매개변수
                 * @details
                 * 수식 : V_{k}(x) = (1/2) * x^{T} * P_{k} * x + p_{k}^{T} * x
                 * 역방향 소거, Backward Pass 시에 계산
                 */

                /** @brief Cost-to-go 2차항 행렬 (Hessian of value function), 크기 : Nx * Nx */
                std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1> P;
                /** @brief Cost-to-go 1차항 벡터 (Gradient of value function), 크기 : Nx */
                std::array<matrix::StaticVector<double, Nx>, H + 1> p;

                /**
                 * @brief [4] 최적 제어 정책 (Optimal Control Policy)
                 * @details
                 * 수식 : du_{k} = -K_{k} * dx_{k} - k_ff_{k}
                 * 패드백 게인과 피드포워드 게인
                 */

                /** @brief 피드백 게인 행렬 (Feedback Gain Matrix), 크기 : Nu x Nx */
                std::array<matrix::StaticMatrix<double, Nu, Nx>, H> K;
                /** @brief 피드포워드 제어 입력 (Feedforward Control Vector), 크기 : Nu */
                std::array<matrix::StaticVector<double, Nu>, H> k_ff;

                /**
                 * @brief [5] 상태 및 제어 업데이트 궤적 (Optimal Trajectories)
                 * @details
                 * 순방향 탐색, Forward Pass 시에 계산
                 */

                /** @brief 상태 변화량 궤적 (Optimal State Update Trajectory), 크기 : Nx */
                std::array<matrix::StaticVector<double, Nx>, H + 1> dx;
                /** @brief 제어 입력 변화량 궤적 (Optimal Control Update Trajectory), zmrl : NU */
                std::array<matrix::StaticVector<double, Nu>, H> du;

                /**
                 * @brief Riccati Recursion 
                 * @param reg_u 제어 입력 Hessian (Quu) 정착화 파라미터 (LM Clamping Factor)
                 * @param reg_x 상태 Hessian (Qxx) 정착화 파라미터
                 * @return SolverStatus 수학적 분해 성공 여부 반환
                 */
                SolverStatus solve(double reg_u = 1e-6, double reg_x = 0.0) {
                    P[H] = Q[H];
                    p[H] = q[H];

                    // 종단 Value Function 정착화
                    for (std::size_t i = 0; i < Nx; ++i) {
                        P[H](i, j) += reg_x;
                    }

                    /**
                     * @brief 1. 역방향 소거 (Backward Pass) Zero-Allocation 전거
                     */
                    for (int k = H - 1; k >= 0; --k) {
                        // P_next_A = P[k+1] * A[k]
                        matrix::StaticMatrix<double, Nx, Nx> P_next_A;
                        linalg::multiply(P[k + 1], A[k], P_next_A);

                        // P_next_B = P[k+1] * B[k]
                        matrix::StaticMatrix<double, Nx, Nu> P_next_B;
                        linalg::multiply(P[k + 1], B[k], P_next_B);

                        // Quu = R[k] + B^{T} * P_next_B
                        matrix::StaticMatrix<double, Nu, Nu> Quu = R[k];
                        matrix::StaticMatrix<double, Nu, Nu> BTPB;
                        linalg::multiply_AT_B(B[k], P_next_B, BTPB);
                        Quu += BTPB;

                        // Q_ux = B^T * P_next_A
                        matrix::StaticMatrix<double, Nu, Nx> Q_ux;
                        linalg::multiply_AT_B(B[k], P_next_A, Q_ux);

                        // p_next_d = p[k+1] + P[k+1] * d[k]
                        matrix::StaticVector<double, Nx> p_next_d;
                        linalg::multiply(P[k+1], d[k], p_next_d);
                        p_next_d += p[k+1];

                        // q_u = r[k] + B^T * p_next_d
                        matrix::StaticVector<double, Nu> q_u = r[k];
                        matrix::StaticVector<double, Nu> BT_pnd;
                        linalg::multiply_AT_B(B[k], p_next_d, BT_pnd);
                        q_u += BT_pnd;

                        // Levenberg-Marquardt 제어 정착화
                        for (std::size_t i = 0; i < Nu; ++i) {
                            Quu(i, i) += reg_u;
                        }

                        // Quu 분해 (Positive Definite 실패 시 상위 LM 루프에 에러 반환)
                        MathStatus linalg_status = linalg::LDLT_decompose(Quu);
                        if (linalg_status != MathStatus::SUCCESS) {
                            return SolverStatus::MATH_ERROR;
                        }

                        // 피드백 게인 K 계산 : K = -Quu^{-1} * Q_ux
                        for (std::size_t i = 0; i < Nx; ++i) {
                            matrix::StaticVector<double, Nu> Qux_col;
                            for (std::size_t j = 0; j < Nu; ++j) {
                                Qux_col(j) = -Q_ux(j, i);
                            }
                            matrix::StaticVector<double, Nu> K_col;

                            linalg::LDLT_solve(Quu, Qux_col, K_col);
                            for (std::size_t j = 0; j < Nu; ++j) {
                                K[k](j, i) = K_col(j);
                            }
                        }

                        // 피드포워드 게인 k_ff 계산 : k_ff = -Quu^{-1} * q_u
                        matrix::StaticVector<double, Nu> neg_qu;
                        for (std::size_t i = 0; i < Nu; ++i) {
                            neg_qu(i) = -q_u(i);
                        }
                        linalg::LDLT_solve(Quu, neg_qu, k_ff[k]);

                        // Value Function Hessian 업데이트 : P[k] = Q[k] + A^{T} * P_next_A + K^{T} * Q_ux
                        matrix::StaticMatrix<double, Nx, Nx> AT_PA;
                        linalg::multiply_AT_B(A[k], P_next_A, AT_PA);
                        matrix::StaticMatrix<double, Nx, Nx> KT_Qux;
                        linalg::multiply_AT_B(K[k], Q_ux, KT_Qux);

                        P[k] = Q[k];
                        P[k] += AT_PA;
                        P[k] += KT_Qux;

                        // Levenberg-Marquardt 상태 정착화
                        for (std::size_t i = 0; i < Nx; ++i) {
                            P[k](i, i) += reg_x;
                        }

                        // Value Function 대칭성 강제 보장 (수치적 누적 오차 방지)
                        for (std::size_t i = 0; i < Nx; ++i) {
                            for (std::size_t j = i + 1; j < Nx; ++j) {
                                double sym_val = 0.5 * (P[k](i, j) + P[k](j, i));
                                P[k](i, j) = sym_val;
                                P[k](j, i) = sym_val;
                            }
                        }

                        // Value Function Gradient 업데이트 : p[k] = q[k] + A^{T} * p_next_d + Q_ux^{T} * k_ff[k]
                        matrix::StaticVector<double, Nx> AT_Pnd;
                        linalg::multiply_AT_B(A[k], p_next_d, AT_Pnd);
                        matrix::StaticVector<double, Nx> QuxT_kff;
                        linalg::multiply_AT_B(Q_ux, k_ff[k], QuxT_kff);

                        p[k] = q[k];
                        p[k] += AT_Pnd;
                        p[k] += QuxT_kff;
                    }

                    /**
                     * @brief 2. 순방향 전진 (Forward Pass)
                     */
                    dx[0].set_zero();

                    for (std::size_t k = 0; k < H; ++k) {
                        // du[k] = K[k] * dx[k] + k_ff[k]
                        linalg::multiply(K[k], dx[k], du[k]);
                        du[k] += k_ff[k];

                        // dx[k+1] = A[k] * dx[k] + B[k] * du[k] + d[k]
                        matrix::StaticVector<double, Nx> Adx;
                        linalg::multiply(A[k], dx[k], Adx);
                        matrix::StaticVector<double, Nx> Bdu;
                        linalg::multiply(B[k], du[k], Bdu);

                        dx[k + 1] = Adx;
                        dx[k + 1] += Bdu;
                        dx[k + 1] += d[k];
                    }
                    return SolverStatus::SUCCESS;
                }
        };
    } // namespace solver
} // namespace Optimization
#endif // OPTIMIZATION_SOLVER_RICCATI_SOLVER_HPP_