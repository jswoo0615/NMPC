#ifndef OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_
#define OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_

#include <array>
#include <cmath>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Solver/RTQPSolver.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include "Optimization/Solver/MeritLineSearch.hpp"

using Optimization::Dual;

namespace Optimization {
namespace control {

/**
 * @brief NMPC 메인 컨트롤러 (Obstacle Avoidance 탑재형)
 * @tparam Ncx 상태 제약 개수 (장애물 1개 = 1)
 * @tparam Ncu 입력 제약 개수 (조향 2개, 가속 2개 = 4)
 */
template <typename Model, size_t H, size_t Nx = 8, size_t Nu = 2, size_t Ncx = 1, size_t Ncu = 4>
class NMPC_Controller {
   public:
    Model dynamics_model;
    solver::RT_QPSolver<H, Nx, Nu, Ncx, Ncu> rt_qp;

    matrix::StaticMatrix<double, Nx, Nx> Q_weight;
    matrix::StaticMatrix<double, Nx, Nx> Q_terminal;
    matrix::StaticMatrix<double, Nu, Nu> R_weight;

    matrix::StaticVector<double, Nu> u_min;
    matrix::StaticVector<double, Nu> u_max;

    // [Architect's Update] 장애물 파라미터
    double obs_s = -1000.0; // 초기화 전에는 멀리 치워둠
    double obs_d = 0.0;
    double obs_r = 0.0;

    std::array<matrix::StaticVector<double, Nx>, H + 1> x_guess;
    std::array<matrix::StaticVector<double, Nu>, H> u_guess;
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;

    std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1> Q_seq;
    std::array<matrix::StaticMatrix<double, Nu, Nu>, H> R_seq;
    std::array<matrix::StaticVector<double, Nx>, H + 1> q_seq;
    std::array<matrix::StaticVector<double, Nu>, H> r_seq;
    std::array<matrix::StaticMatrix<double, Nx, Nx>, H> A_seq;
    std::array<matrix::StaticMatrix<double, Nx, Nu>, H> B_seq;
    std::array<matrix::StaticVector<double, Nx>, H> d_seq;

    double dt_ = 0.05;
    bool is_first_run_ = true;

    void init(const matrix::StaticMatrix<double, Nx, Nx>& Q,
              const matrix::StaticMatrix<double, Nx, Nx>& Q_term,
              const matrix::StaticMatrix<double, Nu, Nu>& R,
              const matrix::StaticVector<double, Nu>& u_lb,
              const matrix::StaticVector<double, Nu>& u_ub,
              double dt = 0.05) {
        Q_weight = Q;
        Q_terminal = Q_term;
        R_weight = R;
        u_min = u_lb;
        u_max = u_ub;
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

    // [Architect's Update] 장애물 설정 인터페이스
    void set_obstacle(double s, double d, double r_safe) {
        obs_s = s;
        obs_d = d;
        obs_r = r_safe;
    }

    matrix::StaticVector<double, Nu> compute_control(const matrix::StaticVector<double, Nx>& x0, 
                                                     int max_iter = 5, 
                                                     double tol = 1e-4) {
        if (is_first_run_) {
            for (size_t k = 0; k <= H; ++k) x_guess[k] = x0;
            is_first_run_ = false;
        }
        
        x_guess[0] = x0;

        for (int iter = 0; iter < max_iter; ++iter) {
            double max_defect = 0.0;

            for (size_t k = 0; k < H; ++k) {
                matrix::StaticVector<double, Nx> x_err;
                for(size_t i = 0; i < Nx; ++i) x_err(i) = x_guess[k](i) - x_ref[k](i);
                linalg::multiply(Q_weight, x_err, q_seq[k]);
                Q_seq[k] = Q_weight;

                linalg::multiply(R_weight, u_guess[k], r_seq[k]);
                R_seq[k] = R_weight;

                for (size_t i = 0; i < Nx; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) x_dual(j) = Dual<double>(x_guess[k](j), (i == j) ? 1.0 : 0.0);
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) u_dual(j) = Dual<double>(u_guess[k](j), 0.0);

                    auto x_dot_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        double I_mat = (i == j) ? 1.0 : 0.0;
                        A_seq[k](j, i) = I_mat + x_dot_dual(j).d * dt_; 
                    }
                }

                for (size_t i = 0; i < Nu; ++i) {
                    matrix::StaticVector<Dual<double>, Nx> x_dual;
                    for (size_t j = 0; j < Nx; ++j) x_dual(j) = Dual<double>(x_guess[k](j), 0.0);
                    matrix::StaticVector<Dual<double>, Nu> u_dual;
                    for (size_t j = 0; j < Nu; ++j) u_dual(j) = Dual<double>(u_guess[k](j), (i == j) ? 1.0 : 0.0);

                    auto x_dot_dual = dynamics_model(x_dual, u_dual);
                    
                    for (size_t j = 0; j < Nx; ++j) {
                        B_seq[k](j, i) = x_dot_dual(j).d * dt_; 
                    }

                    if (i == 0) {
                        for (size_t j = 0; j < Nx; ++j) {
                            double x_next_sim = x_guess[k](j) + x_dot_dual(j).v * dt_;
                            d_seq[k](j) = x_next_sim - x_guess[k + 1](j);
                            if (std::abs(d_seq[k](j)) > max_defect) max_defect = std::abs(d_seq[k](j));
                        }
                    }
                }
            }

            matrix::StaticVector<double, Nx> x_err_H;
            for(size_t i = 0; i < Nx; ++i) x_err_H(i) = x_guess[H](i) - x_ref[H](i);
            linalg::multiply(Q_terminal, x_err_H, q_seq[H]);
            Q_seq[H] = Q_terminal;

            // -----------------------------------------------------------------
            // Phase 1.5: 입력 제약 및 상태 제약(장애물) 배관
            // -----------------------------------------------------------------
            for (size_t k = 0; k < H; ++k) {
                // 입력 제약 (Hard Bounds)
                rt_qp.Cu[k].set_zero();
                rt_qp.Cu[k](0, 0) = 1.0;  rt_qp.dcu[k](0) = u_guess[k](0) - u_max(0);
                rt_qp.Cu[k](1, 0) = -1.0; rt_qp.dcu[k](1) = u_min(0) - u_guess[k](0);
                rt_qp.Cu[k](2, 1) = 1.0;  rt_qp.dcu[k](2) = u_guess[k](1) - u_max(1);
                rt_qp.Cu[k](3, 1) = -1.0; rt_qp.dcu[k](3) = u_min(1) - u_guess[k](1);

                // [Architect's Update] 상태 제약 (장애물 회피 선형화)
                if constexpr (Ncx > 0) {
                    double s_k = x_guess[k](0);
                    double d_k = x_guess[k](1);
                    
                    double dist_sq = (s_k - obs_s) * (s_k - obs_s) + (d_k - obs_d) * (d_k - obs_d);
                    
                    // dcx = R^2 - dist^2 <= 0
                    rt_qp.dcx[k](0) = (obs_r * obs_r) - dist_sq;
                    
                    // Cx 야코비안 조립
                    rt_qp.Cx[k].set_zero();
                    rt_qp.Cx[k](0, 0) = -2.0 * (s_k - obs_s); // ∂/∂s
                    rt_qp.Cx[k](0, 1) = -2.0 * (d_k - obs_d); // ∂/∂d
                }
            }

            rt_qp.init_interior_points(); 
            
            SolverStatus status = rt_qp.solve(Q_seq, R_seq, q_seq, r_seq, A_seq, B_seq, d_seq, 20, 1e-4);
            if (status == SolverStatus::MATH_ERROR) {
                return u_guess[0]; 
            }

            // -----------------------------------------------------------------
            // Phase 3: 궤적 갱신 (Merit Function에 장애물 페널티 추가)
            // -----------------------------------------------------------------
            auto calc_merit = [&](const std::array<matrix::StaticVector<double, Nx>, H + 1>& x_traj,
                                  const std::array<matrix::StaticVector<double, Nu>, H>& u_traj) -> double {
                double merit = 0.0;
                constexpr double SIGMA_DYN = 100.0;
                constexpr double SIGMA_OBS = 500.0; // 장애물 충돌 시 극단적 페널티

                for (size_t k = 0; k < H; ++k) {
                    matrix::StaticVector<double, Nx> x_err;
                    for (size_t i = 0; i < Nx; ++i) x_err(i) = x_traj[k](i) - x_ref[k](i);
                    matrix::StaticVector<double, Nx> Qx;
                    linalg::multiply(Q_weight, x_err, Qx);
                    for (size_t i = 0; i < Nx; ++i) merit += 0.5 * x_err(i) * Qx(i);

                    matrix::StaticVector<double, Nu> Ru;
                    linalg::multiply(R_weight, u_traj[k], Ru);
                    for (size_t i = 0; i < Nu; ++i) merit += 0.5 * u_traj[k](i) * Ru(i);

                    matrix::StaticVector<double, Nx> x_curr = x_traj[k];
                    matrix::StaticVector<double, Nu> u_curr = u_traj[k];
                    matrix::StaticVector<double, Nx> x_dot_sim = dynamics_model(x_curr, u_curr);
                    
                    for (size_t i = 0; i < Nx; ++i) {
                        double x_next_sim = x_curr(i) + x_dot_sim(i) * dt_;
                        double defect = x_next_sim - x_traj[k + 1](i);
                        merit += SIGMA_DYN * std::abs(defect);
                    }

                    // [Architect's Update] 장애물 충돌 검사
                    if constexpr (Ncx > 0) {
                        double s_k = x_curr(0);
                        double d_k = x_curr(1);
                        double dist_sq = (s_k - obs_s) * (s_k - obs_s) + (d_k - obs_d) * (d_k - obs_d);
                        double violation = (obs_r * obs_r) - dist_sq;
                        if (violation > 0.0) merit += SIGMA_OBS * violation;
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
                    for (size_t i = 0; i < Nx; ++i) x_temp[k + 1](i) += alpha * rt_qp.dx_opt[k + 1](i);
                    for (size_t i = 0; i < Nu; ++i) u_temp[k](i) += alpha * rt_qp.du_opt[k](i);
                }
                return calc_merit(x_temp, u_temp);
            };

            double opt_alpha = solver::MeritLineSearch::run(evaluator, current_merit, 0.0);

            double max_dx = 0.0;
            for (size_t k = 0; k < H; ++k) {
                for (size_t i = 0; i < Nx; ++i) {
                    double step_x = opt_alpha * rt_qp.dx_opt[k + 1](i);
                    x_guess[k + 1](i) += step_x;
                    if (std::abs(step_x) > max_dx) max_dx = std::abs(step_x);
                }
                for (size_t i = 0; i < Nu; ++i) {
                    u_guess[k](i) += opt_alpha * rt_qp.du_opt[k](i);
                }
            }

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