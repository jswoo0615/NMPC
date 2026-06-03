#ifndef OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_
#define OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_

#include <array>
#include <cmath>
#include <algorithm>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 Real-Time QP 솔버 (IPM + Riccati 융합 아키텍처)
 * @details [Architect's Update]
 * 밀집 행렬(Dense) LDLT 분해를 철거하고 RiccatiSolver를 이너 루프로 이식하여 O(H) 속도 달성.
 * Primal-Dual IPM의 엄밀한 Newton Step 추적 및 잔차 누적(Accumulation) 로직 이식 완료.
 * @tparam H 예측 구간
 * @tparam Nx 상태 차원
 * @tparam Nu 제어 차원
 * @tparam Ncx 스텝당 상태 부등식 제약 조건 수 (Cx * dx + dcx <= 0)
 * @tparam Ncu 스텝당 입력 부등식 제약 조건 수 (Cu * du + dcu <= 0)
 */
template <size_t H, size_t Nx, size_t Nu, size_t Ncx, size_t Ncu>
class RT_QPSolver {
   public:
    static constexpr double SIGMA = 0.5;
    static constexpr double TAU = 0.995;
    static constexpr double MIN_SLACK = 1e-8;

    // 코어 엔진
    RiccatiSolver<H, Nx, Nu> riccati;

    // =========================================================================
    // 제약 조건 파라미터 (SQP 아우터 루프에서 매 스텝 주입)
    // =========================================================================
    std::array<matrix::StaticMatrix<double, Ncx, Nx>, H + 1> Cx;
    std::array<matrix::StaticVector<double, Ncx>, H + 1> dcx;
    
    std::array<matrix::StaticMatrix<double, Ncu, Nu>, H> Cu;
    std::array<matrix::StaticVector<double, Ncu>, H> dcu;

    // =========================================================================
    // 내부점 (Interior-Point) 변수
    // =========================================================================
    std::array<matrix::StaticVector<double, Ncx>, H + 1> sx, zx; 
    std::array<matrix::StaticVector<double, Ncu>, H> su, zu;     

    // 해답 궤적 (Primal Variables)
    std::array<matrix::StaticVector<double, Nx>, H + 1> dx_opt;
    std::array<matrix::StaticVector<double, Nu>, H> du_opt;

    void init_interior_points() {
        for (size_t k = 0; k <= H; ++k) {
            for (size_t i = 0; i < Nx; ++i) dx_opt[k](i) = 0.0; // [핵심 교정] 탐색 시작점 0으로 엄밀히 초기화
            for (size_t i = 0; i < Ncx; ++i) { sx[k](i) = 1.0; zx[k](i) = 1.0; }
            if (k < H) {
                for (size_t i = 0; i < Nu; ++i) du_opt[k](i) = 0.0;
                for (size_t i = 0; i < Ncu; ++i) { su[k](i) = 1.0; zu[k](i) = 1.0; }
            }
        }
    }

    SolverStatus solve(
        const std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1>& Q,
        const std::array<matrix::StaticMatrix<double, Nu, Nu>, H>& R,
        const std::array<matrix::StaticVector<double, Nx>, H + 1>& q,
        const std::array<matrix::StaticVector<double, Nu>, H>& r,
        const std::array<matrix::StaticMatrix<double, Nx, Nx>, H>& A,
        const std::array<matrix::StaticMatrix<double, Nx, Nu>, H>& B,
        const std::array<matrix::StaticVector<double, Nx>, H>& d,
        int max_iter = 20, double tol = 1e-4) {

        for (int iter = 0; iter < max_iter; ++iter) {
            double mu = 0.0;
            size_t total_constraints = (H + 1) * Ncx + H * Ncu;
            
            // -----------------------------------------------------------------
            // 1. KKT 잔차 계산 및 Duality Measure (mu) 산출
            // -----------------------------------------------------------------
            std::array<matrix::StaticVector<double, Ncx>, H + 1> r_cx, r_dx;
            std::array<matrix::StaticVector<double, Ncu>, H> r_cu, r_du;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cx_dx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) cx_dx += Cx[k](i, j) * dx_opt[k](j);
                    
                    mu += sx[k](i) * zx[k](i);
                    // [핵심 교정] 현재까지의 궤적(dx_opt)이 반영된 엄밀한 Primal 잔차 계산
                    r_dx[k](i) = cx_dx + dcx[k](i) + sx[k](i); 
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cu_du = 0.0;
                        for (size_t j = 0; j < Nu; ++j) cu_du += Cu[k](i, j) * du_opt[k](j);

                        mu += su[k](i) * zu[k](i);
                        r_du[k](i) = cu_du + dcu[k](i) + su[k](i);
                    }
                }
            }
            mu /= static_cast<double>(total_constraints);

            double max_res = 0.0;
            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    r_cx[k](i) = sx[k](i) * zx[k](i) - SIGMA * mu;
                    if (std::abs(r_dx[k](i)) > max_res) max_res = std::abs(r_dx[k](i));
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        r_cu[k](i) = su[k](i) * zu[k](i) - SIGMA * mu;
                        if (std::abs(r_du[k](i)) > max_res) max_res = std::abs(r_du[k](i));
                    }
                }
            }

            if (max_res < tol && mu < tol) {
                return SolverStatus::SUCCESS;
            }

            // -----------------------------------------------------------------
            // 2. Riccati 엔진에 주입할 변조된 비용 함수(Schur Complement) 조립
            // -----------------------------------------------------------------
            for (size_t k = 0; k <= H; ++k) {
                riccati.Q[k] = Q[k];
                
                // [핵심 교정] 현재까지의 Q * dx_opt 기울기 반영
                for (size_t i = 0; i < Nx; ++i) {
                    double q_dx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) q_dx += Q[k](i, j) * dx_opt[k](j);
                    riccati.q[k](i) = q[k](i) + q_dx;
                }

                for (size_t i = 0; i < Ncx; ++i) {
                    double inv_s = 1.0 / MathTraits<double>::max(sx[k](i), MIN_SLACK);
                    double w_x = zx[k](i) * inv_s;
                    double t_x = (zx[k](i) * r_dx[k](i) - r_cx[k](i)) * inv_s;

                    for (size_t r_idx = 0; r_idx < Nx; ++r_idx) {
                        riccati.q[k](r_idx) += Cx[k](i, r_idx) * (zx[k](i) + t_x);
                        for (size_t c_idx = 0; c_idx < Nx; ++c_idx) {
                            riccati.Q[k](r_idx, c_idx) += Cx[k](i, r_idx) * w_x * Cx[k](i, c_idx);
                        }
                    }
                }

                if (k < H) {
                    riccati.R[k] = R[k];
                    
                    for (size_t i = 0; i < Nu; ++i) {
                        double r_du = 0.0;
                        for (size_t j = 0; j < Nu; ++j) r_du += R[k](i, j) * du_opt[k](j);
                        riccati.r[k](i) = r[k](i) + r_du;
                    }

                    for (size_t i = 0; i < Nx; ++i) {
                        double a_dx = 0.0, b_du = 0.0;
                        for (size_t j = 0; j < Nx; ++j) a_dx += A[k](i, j) * dx_opt[k](j);
                        for (size_t j = 0; j < Nu; ++j) b_du += B[k](i, j) * du_opt[k](j);
                        riccati.d[k](i) = a_dx + b_du + d[k](i) - dx_opt[k+1](i);
                    }
                    
                    riccati.A[k] = A[k];
                    riccati.B[k] = B[k];

                    for (size_t i = 0; i < Ncu; ++i) {
                        double inv_s = 1.0 / MathTraits<double>::max(su[k](i), MIN_SLACK);
                        double w_u = zu[k](i) * inv_s;
                        double t_u = (zu[k](i) * r_du[k](i) - r_cu[k](i)) * inv_s;

                        for (size_t r_idx = 0; r_idx < Nu; ++r_idx) {
                            riccati.r[k](r_idx) += Cu[k](i, r_idx) * (zu[k](i) + t_u);
                            for (size_t c_idx = 0; c_idx < Nu; ++c_idx) {
                                riccati.R[k](r_idx, c_idx) += Cu[k](i, r_idx) * w_u * Cu[k](i, c_idx);
                            }
                        }
                    }
                }
            }

            // -----------------------------------------------------------------
            // 3. 초고속 Riccati 선형 시스템 풀이 (O(H))
            // -----------------------------------------------------------------
            if (riccati.solve(1e-6, 1e-6) != SolverStatus::SUCCESS) {
                return SolverStatus::MATH_ERROR;
            }

            // -----------------------------------------------------------------
            // 4. 슬랙(Slack) 및 쌍대(Dual) 변수 Back-substitution (탐색 방향 도출)
            // -----------------------------------------------------------------
            std::array<matrix::StaticVector<double, Ncx>, H + 1> dsx, dzx;
            std::array<matrix::StaticVector<double, Ncu>, H> dsu, dzu;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cx_ddx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) cx_ddx += Cx[k](i, j) * riccati.dx[k](j);

                    dsx[k](i) = -r_dx[k](i) - cx_ddx;
                    double inv_s = 1.0 / MathTraits<double>::max(sx[k](i), MIN_SLACK);
                    dzx[k](i) = -(r_cx[k](i) + zx[k](i) * dsx[k](i)) * inv_s;
                }

                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cu_ddu = 0.0;
                        for (size_t j = 0; j < Nu; ++j) cu_ddu += Cu[k](i, j) * riccati.du[k](j);

                        dsu[k](i) = -r_du[k](i) - cu_ddu;
                        double inv_s = 1.0 / MathTraits<double>::max(su[k](i), MIN_SLACK);
                        dzu[k](i) = -(r_cu[k](i) + zu[k](i) * dsu[k](i)) * inv_s;
                    }
                }
            }

            // -----------------------------------------------------------------
            // 5. Fraction-to-Boundary Rule (Branchless Step Size)
            // -----------------------------------------------------------------
            double alpha_p = 1.0, alpha_d = 1.0;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cp = (dsx[k](i) < 0.0) ? (-TAU * sx[k](i) / dsx[k](i)) : 1.0;
                    alpha_p = MathTraits<double>::min(alpha_p, cp);
                    double cd = (dzx[k](i) < 0.0) ? (-TAU * zx[k](i) / dzx[k](i)) : 1.0;
                    alpha_d = MathTraits<double>::min(alpha_d, cd);
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cp = (dsu[k](i) < 0.0) ? (-TAU * su[k](i) / dsu[k](i)) : 1.0;
                        alpha_p = MathTraits<double>::min(alpha_p, cp);
                        double cd = (dzu[k](i) < 0.0) ? (-TAU * zu[k](i) / dzu[k](i)) : 1.0;
                        alpha_d = MathTraits<double>::min(alpha_d, cd);
                    }
                }
            }

            // -----------------------------------------------------------------
            // 6. 궤적 및 내부점 '누적 업데이트' (Newton Step Integration)
            // -----------------------------------------------------------------
            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Nx; ++i) dx_opt[k](i) += alpha_p * riccati.dx[k](i); 
                for (size_t i = 0; i < Ncx; ++i) {
                    sx[k](i) += alpha_p * dsx[k](i);
                    zx[k](i) += alpha_d * dzx[k](i);
                }
                if (k < H) {
                    for (size_t i = 0; i < Nu; ++i) du_opt[k](i) += alpha_p * riccati.du[k](i); // [핵심 교정] 누적
                    for (size_t i = 0; i < Ncu; ++i) {
                        su[k](i) += alpha_p * dsu[k](i);
                        zu[k](i) += alpha_d * dzu[k](i);
                    }
                }
            }
        }
        return SolverStatus::MAX_ITERATION_REACHED;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_