#ifndef OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_
#define OPTIMIZATION_CONTROL_NMPC_CONTROLLER_HPP_

#include <array>
#include <cmath>
#include <iostream>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Solver/RTQPSolver.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include "Optimization/Solver/MeritLineSearch.hpp"

// 강건성 모니터링 및 비상 제어 엔진 인스턴스 인클루드
#include "Optimization/Solver/KKTMonitor.hpp"
#include "Optimization/Solver/FallbackControl.hpp"

using Optimization::Dual;

namespace Optimization {
namespace control {

/**
 * @brief NMPC 메인 컨트롤러 (다중 원형 회피 + KKT Fallback 의미론 교정본)
 */
template <typename Model, size_t H, size_t Nx = 8, size_t Nu = 2, size_t Ncx = 2, size_t Ncu = 4>
class NMPC_Controller {
   public:
    Model dynamics_model;
    solver::RT_QPSolver<H, Nx, Nu, Ncx, Ncu> rt_qp;

    matrix::StaticMatrix<double, Nx, Nx> Q_weight;
    matrix::StaticMatrix<double, Nx, Nx> Q_terminal;
    matrix::StaticMatrix<double, Nu, Nu> R_weight;

    matrix::StaticVector<double, Nu> u_min;
    matrix::StaticVector<double, Nu> u_max;

    double obs_s = -1000.0; 
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
                        A_seq[k](j, i) = ((i == j) ? 1.0 : 0.0) + x_dot_dual(j).d * dt_; 
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
                            d_seq[k](j) = (x_guess[k](j) + x_dot_dual(j).v * dt_) - x_guess[k + 1](j);
                            if (std::abs(d_seq[k](j)) > max_defect) max_defect = std::abs(d_seq[k](j));
                        }
                    }
                }
            }

            matrix::StaticVector<double, Nx> x_err_H;
            for(size_t i = 0; i < Nx; ++i) x_err_H(i) = x_guess[H](i) - x_ref[H](i);
            linalg::multiply(Q_terminal, x_err_H, q_seq[H]);
            Q_seq[H] = Q_terminal;

            for (size_t k = 0; k < H; ++k) {
                rt_qp.Cu[k].set_zero();
                rt_qp.Cu[k](0, 0) = 1.0;  rt_qp.dcu[k](0) = u_guess[k](0) - u_max(0);
                rt_qp.Cu[k](1, 0) = -1.0; rt_qp.dcu[k](1) = u_min(0) - u_guess[k](0);
                rt_qp.Cu[k](2, 1) = 1.0;  rt_qp.dcu[k](2) = u_guess[k](1) - u_max(1);
                rt_qp.Cu[k](3, 1) = -1.0; rt_qp.dcu[k](3) = u_min(1) - u_guess[k](1);

                if constexpr (Ncx >= 2) {
                    double s_k = x_guess[k](0);
                    double d_k = x_guess[k](1);
                    double mu_k = x_guess[k](2);

                    constexpr double L_f = 1.2; 
                    constexpr double L_r = 1.6; 
                    
                    rt_qp.Cx[k].set_zero(); 

                    // 1. Front Circle
                    double sf = s_k + L_f * std::cos(mu_k);
                    double df = d_k + L_f * std::sin(mu_k);
                    double dist_f_s = sf - obs_s;
                    double dist_f_d = df - obs_d;
                    
                    rt_qp.dcx[k](0) = (obs_r * obs_r) - (dist_f_s * dist_f_s + dist_f_d * dist_f_d);
                    rt_qp.Cx[k](0, 0) = -2.0 * dist_f_s; 
                    rt_qp.Cx[k](0, 1) = -2.0 * dist_f_d; 
                    rt_qp.Cx[k](0, 2) = -2.0 * dist_f_s * (-L_f * std::sin(mu_k)) - 2.0 * dist_f_d * (L_f * std::cos(mu_k)); 

                    // 2. Rear Circle
                    double sr = s_k - L_r * std::cos(mu_k);
                    double dr = d_k - L_r * std::sin(mu_k);
                    double dist_r_s = sr - obs_s;
                    double dist_r_d = dr - obs_d;
                    
                    rt_qp.dcx[k](1) = (obs_r * obs_r) - (dist_r_s * dist_r_s + dist_r_d * dist_r_d);
                    rt_qp.Cx[k](1, 0) = -2.0 * dist_r_s; 
                    rt_qp.Cx[k](1, 1) = -2.0 * dist_r_d; 
                    rt_qp.Cx[k](1, 2) = -2.0 * dist_r_s * (L_r * std::sin(mu_k)) - 2.0 * dist_r_d * (-L_r * std::cos(mu_k)); 
                }
            }

            rt_qp.init_interior_points(); 
            
            SolverStatus status = rt_qp.solve(Q_seq, R_seq, q_seq, r_seq, A_seq, B_seq, d_seq, 20, 1e-4);

            // =========================================================================
            // [Architect's Core Fix] Phase 2.5: 의미론적 매핑 (Semantic Bridging)
            // =========================================================================
            Optimization::solver::SolverKKTState<double> kkt_state;
            if constexpr (Ncx >= 2) {
                kkt_state.mu_f = rt_qp.dual_x[0](0);
                kkt_state.mu_r = rt_qp.dual_x[0](1);
                kkt_state.h_f  = rt_qp.dcx[0](0);
                kkt_state.h_r  = rt_qp.dcx[0](1);
                
                // [핵심 교정] IPM 배리어 슬랙(sx)을 주지 않고, 준상님의 엔진이 이해하는 '진짜 물리적 위반량'으로 매핑
                kkt_state.s_f  = (rt_qp.dcx[0](0) > 0.0) ? rt_qp.dcx[0](0) : 0.0;
                kkt_state.s_r  = (rt_qp.dcx[0](1) > 0.0) ? rt_qp.dcx[0](1) : 0.0;
            } else {
                kkt_state.mu_f = 0.0; kkt_state.mu_r = 0.0;
                kkt_state.h_f  = 0.0; kkt_state.h_r  = 0.0;
                kkt_state.s_f  = 0.0; kkt_state.s_r  = 0.0;
            }

            Optimization::solver::KKTMonitorParams<double> fallback_params;
            fallback_params.slack_penalty_weight = 1e5; 
            fallback_params.slack_threshold      = 0.1;   // 10cm 이상의 물리적 침범이 일어날 때만 비상 제동 트리거
            fallback_params.comp_error_tol       = 1.0;   
            fallback_params.dual_max_limits      = 1000.0; 
            
            auto trigger_state = Optimization::solver::evaluateKKTAndFallback(fallback_params, kkt_state);

            if (status == SolverStatus::MATH_ERROR || trigger_state.is_fallback_required) {
                std::cout << "[KKT MONITOR] Emergency Condition! Activating Fallback Strategy.\n";
                
                Optimization::solver::ControlOutput<double> nmpc_u = {u_guess[0](1), u_guess[0](0)};
                auto safe_u = Optimization::solver::applyFallbackStrategy(nmpc_u, trigger_state, -3.0, 0.0);
                
                matrix::StaticVector<double, Nu> safe_u_vec;
                safe_u_vec(0) = safe_u.delta; 
                safe_u_vec(1) = safe_u.a_cmd; 
                
                shift_trajectory(); 
                return safe_u_vec;
            }

            // -----------------------------------------------------------------
            // Phase 3: 궤적 갱신
            // -----------------------------------------------------------------
            auto calc_merit = [&](const std::array<matrix::StaticVector<double, Nx>, H + 1>& x_traj,
                                  const std::array<matrix::StaticVector<double, Nu>, H>& u_traj) -> double {
                double merit = 0.0;
                constexpr double SIGMA_DYN = 100.0;
                constexpr double SIGMA_OBS = 1e5;

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
                        merit += SIGMA_DYN * std::abs((x_curr(i) + x_dot_sim(i) * dt_) - x_traj[k + 1](i));
                    }

                    if constexpr (Ncx >= 2) {
                        constexpr double L_f = 1.2;
                        constexpr double L_r = 1.6;
                        double s_k = x_curr(0), d_k = x_curr(1), mu_k = x_curr(2);

                        double sf = s_k + L_f * std::cos(mu_k);
                        double df = d_k + L_f * std::sin(mu_k);
                        double viol_f = (obs_r * obs_r) - ((sf - obs_s)*(sf - obs_s) + (df - obs_d)*(df - obs_d));
                        if (viol_f > 0.0) merit += SIGMA_OBS * viol_f * viol_f;

                        double sr = s_k - L_r * std::cos(mu_k);
                        double dr = d_k - L_r * std::sin(mu_k);
                        double viol_r = (obs_r * obs_r) - ((sr - obs_s)*(sr - obs_s) + (dr - obs_d)*(dr - obs_d));
                        if (viol_r > 0.0) merit += SIGMA_OBS * viol_r * viol_r;
                    }
                }
                
                matrix::StaticVector<double, Nx> x_err_H;
                for (size_t i = 0; i < Nx; ++i) x_err_H(i) = x_traj[H](i) - x_ref[H](i);
                matrix::StaticVector<double, Nx> Qx_H;
                linalg::multiply(Q_terminal, x_err_H, Qx_H);
                for (size_t i = 0; i < Nx; ++i) merit += 0.5 * x_err_H(i) * Qx_H(i);

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

            if (max_defect < tol && max_dx < tol) break;
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