#ifndef OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_
#define OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_

#include <array>
#include <cmath>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include "Optimization/Solver/MeritLineSearch.hpp"

using Optimization::Dual;

namespace Optimization {
namespace control {

template <typename Model, size_t H, size_t Nx = 8, size_t Nu = 2>
class NMPC_Controller {
   public:
    Model dynamics_model;
    solver::RiccatiSolver<H, Nx, Nu> riccati;

    matrix::StaticMatrix<double, Nx, Nx> Q_weight;
    matrix::StaticMatrix<double, Nx, Nx> Q_terminal;
    matrix::StaticMatrix<double, Nu, Nu> R_weight;

    std::array<matrix::StaticVector<double, Nx>, H + 1> x_guess;
    std::array<matrix::StaticVector<double, Nu>, H> u_guess;
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;

    // [Architect's Update] 제어 주기(dt) 추가
    double dt_ = 0.05;
    bool is_first_run_ = true;

    // init 시 dt 파라미터 추가
    void init(const matrix::StaticMatrix<double, Nx, Nx>& Q,
              const matrix::StaticMatrix<double, Nx, Nx>& Q_term,
              const matrix::StaticMatrix<double, Nu, Nu>& R,
              double dt = 0.05) {
        Q_weight = Q;
        Q_terminal = Q_term;
        R_weight = R;
        dt_ = dt;

        for (size_t k = 0; k <= H; ++k) {
            x_guess[k].set_zero();
            x_ref[k].set_zero();
            if (k < H) u_guess[k].set_zero();
        }
        is_first_run_ = true;
    }

    void set_reference_trajectory(const std::array<matrix::StaticVector<double, Nx>, H + 1>& ref) {
        x_ref = ref;
    }

    matrix::StaticVector<double, Nu> compute_control(const matrix::StaticVector<double, Nx>& x0, 
                                                     int max_iter = 5, 
                                                     double tol = 1e-4) {
        // [Architect's Update] 첫 실행 시 초기 상태(x0)를 Horizon 전체에 복사 (Warm-Start 초기화)
        // 정지 상태(v=0)의 특이점(Singularity) 방지 및 초기 Defect 폭주 방지
        if (is_first_run_) {
            for (size_t k = 0; k <= H; ++k) x_guess[k] = x0;
            is_first_run_ = false;
        }
        
        x_guess[0] = x0;

        for (int iter = 0; iter < max_iter; ++iter) {
            double max_defect = 0.0;

            // -----------------------------------------------------------------
            // Phase 1: 선형화 및 KKT 시스템 조립 (Euler Discretization 적용)
            // -----------------------------------------------------------------
            for (size_t k = 0; k < H; ++k) {
                matrix::StaticVector<double, Nx> x_err;
                for(size_t i = 0; i < Nx; ++i) x_err(i) = x_guess[k](i) - x_ref[k](i);
                linalg::multiply(Q_weight, x_err, riccati.q[k]);
                riccati.Q[k] = Q_weight;

                linalg::multiply(R_weight, u_guess[k], riccati.r[k]);
                riccati.R[k] = R_weight;

                // [Architect's Update] A_discrete = I + A_continuous * dt
                for (size_t i = 0; i < Nx; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) x_dual(j) = Dual<double>(x_guess[k](j), (i == j) ? 1.0 : 0.0);
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) u_dual(j) = Dual<double>(u_guess[k](j), 0.0);

                    auto x_dot_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        double I_mat = (i == j) ? 1.0 : 0.0;
                        riccati.A[k](j, i) = I_mat + x_dot_dual(j).d * dt_; 
                    }
                }

                // [Architect's Update] B_discrete = B_continuous * dt
                for (size_t i = 0; i < Nu; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) x_dual(j) = Dual<double>(x_guess[k](j), 0.0);
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) u_dual(j) = Dual<double>(u_guess[k](j), (i == j) ? 1.0 : 0.0);

                    auto x_dot_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        riccati.B[k](j, i) = x_dot_dual(j).d * dt_; 
                    }

                    if (i == 0) {
                        for (size_t j = 0; j < Nx; ++j) {
                            // [Architect's Update] Discrete Defect: x_{k} + \dot{x}*dt - x_{k+1}
                            double x_next_sim = x_guess[k](j) + x_dot_dual(j).v * dt_;
                            riccati.d[k](j) = x_next_sim - x_guess[k + 1](j);
                            
                            double abs_d = std::abs(riccati.d[k](j));
                            if (abs_d > max_defect) max_defect = abs_d;
                        }
                    }
                }
            }

            matrix::StaticVector<double, Nx> x_err_H;
            for(size_t i = 0; i < Nx; ++i) x_err_H(i) = x_guess[H](i) - x_ref[H](i);
            linalg::multiply(Q_terminal, x_err_H, riccati.q[H]);
            riccati.Q[H] = Q_terminal;

            // -----------------------------------------------------------------
            // Phase 2: 이너 루프 (Riccati Solver 실행)
            // -----------------------------------------------------------------
            SolverStatus status = riccati.solve(1e-4, 1e-6);
            if (status != SolverStatus::SUCCESS) {
                return u_guess[0]; // 발산 방지 폴백
            }

            // -----------------------------------------------------------------
            // Phase 3: 궤적 갱신 (Armijo Backtracking Line Search)
            // -----------------------------------------------------------------
            auto calc_merit = [&](const std::array<matrix::StaticVector<double, Nx>, H + 1>& x_traj,
                                  const std::array<matrix::StaticVector<double, Nu>, H>& u_traj) -> double {
                double merit = 0.0;
                constexpr double SIGMA_PENALTY = 100.0;

                for (size_t k = 0; k < H; ++k) {
                    matrix::StaticVector<double, Nx> x_err;
                    for (size_t i = 0; i < Nx; ++i) x_err(i) = x_traj[k](i) - x_ref[k](i);
                    
                    matrix::StaticVector<double, Nx> Qx;
                    linalg::multiply(Q_weight, x_err, Qx);
                    for (size_t i = 0; i < Nx; ++i) merit += 0.5 * x_err(i) * Qx(i);

                    matrix::StaticVector<double, Nu> Ru;
                    linalg::multiply(R_weight, u_traj[k], Ru);
                    for (size_t i = 0; i < Nu; ++i) merit += 0.5 * u_traj[k](i) * Ru(i);

                    // [Architect's Update] Merit Function에도 동일하게 Euler 적분 적용
                    matrix::StaticVector<double, Nx> x_curr = x_traj[k];
                    matrix::StaticVector<double, Nu> u_curr = u_traj[k];
                    matrix::StaticVector<double, Nx> x_dot_sim = dynamics_model(x_curr, u_curr);
                    
                    for (size_t i = 0; i < Nx; ++i) {
                        double x_next_sim = x_curr(i) + x_dot_sim(i) * dt_;
                        double defect = x_next_sim - x_traj[k + 1](i);
                        merit += SIGMA_PENALTY * std::abs(defect);
                    }
                }
                
                matrix::StaticVector<double, Nx> x_err_H_term;
                for (size_t i = 0; i < Nx; ++i) x_err_H_term(i) = x_traj[H](i) - x_ref[H](i);
                matrix::StaticVector<double, Nx> Qx_H;
                linalg::multiply(Q_terminal, x_err_H_term, Qx_H);
                for (size_t i = 0; i < Nx; ++i) merit += 0.5 * x_err_H_term(i) * Qx_H(i);

                return merit;
            };

            double current_merit = calc_merit(x_guess, u_guess);

            auto evaluator = [&](double alpha) -> double {
                std::array<matrix::StaticVector<double, Nx>, H + 1> x_temp = x_guess;
                std::array<matrix::StaticVector<double, Nu>, H> u_temp = u_guess;

                for (size_t k = 0; k < H; ++k) {
                    for (size_t i = 0; i < Nx; ++i) x_temp[k + 1](i) += alpha * riccati.dx[k + 1](i);
                    for (size_t i = 0; i < Nu; ++i) u_temp[k](i) += alpha * riccati.du[k](i);
                }
                return calc_merit(x_temp, u_temp);
            };

            double opt_alpha = solver::MeritLineSearch::run(evaluator, current_merit, 0.0);

            double max_dx = 0.0;
            for (size_t k = 0; k < H; ++k) {
                for (size_t i = 0; i < Nx; ++i) {
                    double step_x = opt_alpha * riccati.dx[k + 1](i);
                    x_guess[k + 1](i) += step_x;
                    
                    if (std::abs(step_x) > max_dx) max_dx = std::abs(step_x);
                }
                for (size_t i = 0; i < Nu; ++i) {
                    u_guess[k](i) += opt_alpha * riccati.du[k](i);
                }
            }

            // -----------------------------------------------------------------
            // Phase 4: 수렴 판정 (Convergence Check)
            // -----------------------------------------------------------------
            if (max_defect < tol && max_dx < tol) {
                break;
            }
        } 

        return u_guess[0];
    } 

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