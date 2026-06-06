#ifndef OPTIMIZATION_CONTROLLER_SPARSE_NMPC_IPM_HPP_
#define OPTIMIZATION_CONTROLLER_SPARSE_NMPC_IPM_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "Optimization/Control/SafeBuffer.hpp"
#include "Optimization/Matrix/AD/DualVec.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Simulation/Integrator.hpp"
#include "Optimization/Solver/KKTMonitor.hpp"
#include "Optimization/Solver/MeritLineSearch.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"

namespace Optimization {

#ifndef OBSTACLE_FRENET_DEFINED
#define OBSTACLE_FRENET_DEFINED
struct ObstacleFrenet {
    double s = 0.0;
    double d = 0.0;
    double r = 0.5;
    double vs = 0.0;
    double vd = 0.0;
};
#endif

namespace controller {

#ifndef NMPC_RESULT_DEFINED
#define NMPC_RESULT_DEFINED
struct NMPCResult {
    bool success = false;
    bool fallback_triggered = false;
    double max_kkt_error = 0.0;
    int sqp_iterations = 0;
    std::string status_msg = "OK";
};
#endif

#ifndef NMPC_TUNING_CONFIG_DEFINED
#define NMPC_TUNING_CONFIG_DEFINED
struct NMPCTuningConfig {
    double Q_D = 200.0;
    double Q_mu = 50.0;
    double Q_Vx = 50.0;
    double Q_Vy = 1.0;
    double Q_r = 1.0;
    double Q_alpha_f = 10.0;
    double Q_alpha_r = 10.0;

    double R_Steer = 5000.0;
    double R_Accel = 10.0;

    double R_Steer_Rate = 50000.0;
    double R_Accel_Rate = 100.0;

    double Obstacle_Penalty = 20000.0;
    double Obstacle_Margin = 1.5;

    double W_slack = 100000.0;

    double damping_Q = 5.0;
    double damping_R = 500.0;

    double d_max = 3.5;
    double d_min = -3.5;
    double u_min[2] = {-0.6, -10.0};
    double u_max[2] = {0.6, 10.0};

    double kappa = 0.0;
    double target_vx = 10.0;
    double target_d[100] = {0.0};

    int ipm_max_iter = 8;
    double kkt_tolerance = 1e-2;
};
#endif

template <size_t H, typename PlantModel = Dynamics::RealTimeDynamicsModel, size_t Nx = 8, size_t Nu = 2>
class SparseNMPC_IPM {
   public:
    double dt;
    std::array<matrix::StaticVector<double, Nu>, H> U_guess;
    std::array<matrix::StaticVector<double, Nx>, H + 1> X_pred;
    matrix::StaticVector<double, Nu> u_last;

    std::array<ObstacleFrenet, 10> obstacles;
    double mu;

    solver::RiccatiSolver<H, Nx, Nu> riccati;
    SafeBuffer<H, Nx, Nu> safe_buffer;

    SparseNMPC_IPM() : dt(0.05), mu(1.0) {
        u_last.set_zero();
        for (size_t k = 0; k < H; ++k) U_guess[k].set_zero();
        for (auto& obs : obstacles) {
            obs.s = 10000.0;
            obs.d = 10000.0;
            obs.r = 0.1;
            obs.vs = 0.0;
            obs.vd = 0.0;
        }
    }

    inline void shift_sequence() {
        for (size_t k = 0; k < H - 1; ++k) {
            U_guess[k] = U_guess[k + 1];
            X_pred[k] = X_pred[k + 1];
        }
        X_pred[H - 1] = X_pred[H];
        U_guess[H - 1](0) *= 0.5;
        U_guess[H - 1](1) *= 0.5;
        
        PlantModel model;
        X_pred[H] = integrator::step_rk4<Nx, Nu, PlantModel, double>(model, X_pred[H - 1], U_guess[H - 1], dt);
    }

    inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason,
                                       const NMPCTuningConfig& config) {
        res.success = false;
        res.fallback_triggered = true;
        res.status_msg = "IPM Fallback: " + reason;

        matrix::StaticVector<double, Nu> safe_u = safe_buffer.extract_fallback_control(config.u_min[1]);
        if (safe_buffer.has_valid_trajectory) {
            for (size_t k = 0; k < H; ++k) U_guess[k] = safe_buffer.U_safe[k];
            for (size_t k = 0; k <= H; ++k) X_pred[k] = safe_buffer.X_safe[k];
        } else {
            for (size_t k = 0; k < H; ++k) U_guess[k] = safe_u;
        }
        u_last = U_guess[0];
        return res;
    }

    double evaluate_merit(double alpha, const NMPCTuningConfig& config,
                          const matrix::StaticVector<double, Nx>& x_init) {
        double merit = 0.0;
        constexpr double L1_WEIGHT = 1000.0;
        PlantModel plant;
        plant.kappa = config.kappa;

        matrix::StaticVector<double, Nx> x_curr = X_pred[0];
        x_curr.saxpy(alpha, riccati.dx[0]);
        for (size_t i = 0; i < Nx; ++i) merit += L1_WEIGHT * std::abs(x_curr(i) - x_init(i));

        for (size_t k = 0; k < H; ++k) {
            matrix::StaticVector<double, Nu> u_cand = U_guess[k];
            u_cand.saxpy(alpha, riccati.du[k]);
            matrix::StaticVector<double, Nx> x_next = X_pred[k + 1];
            x_next.saxpy(alpha, riccati.dx[k + 1]);

            double err_d = x_curr(1) - config.target_d[k];
            double err_mu = x_curr(2); 
            double err_v = x_curr(3) - config.target_vx;
            
            merit += 0.5 * (config.Q_D * err_d * err_d + config.Q_mu * err_mu * err_mu + config.Q_Vx * err_v * err_v);
            merit += 0.5 * (config.Q_Vy * x_curr(4) * x_curr(4));
            merit += 0.5 * (config.Q_r * x_curr(5) * x_curr(5));
            merit += 0.5 * (config.Q_alpha_f * x_curr(6) * x_curr(6));
            merit += 0.5 * (config.Q_alpha_r * x_curr(7) * x_curr(7));

            merit += 0.5 * (config.R_Steer * u_cand(0) * u_cand(0) +
                            config.R_Accel * u_cand(1) * u_cand(1));

            double violation_max = x_curr(1) - config.d_max;
            double violation_min = config.d_min - x_curr(1);
            
            if (violation_max > 0.0) {
                merit += 0.5 * config.W_slack * violation_max * violation_max;
            }
            if (violation_min > 0.0) {
                merit += 0.5 * config.W_slack * violation_min * violation_min;
            }

            double time_future = k * dt;
            for (size_t i = 0; i < 10; ++i) {
                double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                double ds = x_curr(0) - obs_pred_s;
                double dd = x_curr(1) - obs_pred_d;
                
                // Symmetry breaking to push it laterally early
                double dd_eff = dd;
                if (std::abs(dd_eff) < 0.1) dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                
                double dist_sq = ds * ds + dd_eff * dd_eff;
                double safety_margin = obstacles[i].r + config.Obstacle_Margin;
                double violation = safety_margin * safety_margin - dist_sq;
                
                if (violation > 0.0) {
                    merit += config.Obstacle_Penalty * violation * violation;
                }
            }

            matrix::StaticVector<double, Nx> x_sim =
                integrator::IntegratorEngine<Nx, Nu, PlantModel, double>::compute(
                    plant, x_curr, u_cand, dt);
            for (size_t i = 0; i < Nx; ++i) merit += L1_WEIGHT * std::abs(x_next(i) - x_sim(i));
            x_curr = x_next;
        }
        return merit;
    }

    NMPCResult solve_ipm(const matrix::StaticVector<double, Nx>& x_curr, const NMPCTuningConfig& config) {
        NMPCResult result;
        result.sqp_iterations = 0;

        if (std::isnan(x_curr(1)) || std::isnan(x_curr(3))) {
            return execute_fallback(result, "Primal State NaN Detected", config);
        }

        PlantModel model;
        model.kappa = config.kappa;

        X_pred[0] = x_curr;
        X_pred[0](3) = MathTraits<double>::max(0.1, x_curr(3));
        
        // Initial Rollout to ensure dynamically feasible initial guess
        for (size_t k = 0; k < H; ++k) {
            X_pred[k + 1] = integrator::step_rk4<Nx, Nu, PlantModel, double>(model, X_pred[k], U_guess[k], dt);
        }

        mu = 1.0;

        for (int ipm_step = 0; ipm_step < config.ipm_max_iter; ++ipm_step) {
            result.sqp_iterations++;

            bool update_jacobian = (ipm_step % 2 == 0);

            if (update_jacobian) {
                using ADVar = matrix::DualVec<double, Nx + Nu>;
                for (size_t k = 0; k < H; ++k) {
                    matrix::StaticVector<ADVar, Nx> x_dual;
                    matrix::StaticVector<ADVar, Nu> u_dual;
                    for (size_t i = 0; i < Nx; ++i)
                        x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                    for (size_t i = 0; i < Nu; ++i)
                        u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);

                    matrix::StaticVector<ADVar, Nx> x_next_dual =
                        integrator::step_rk4<Nx, Nu, PlantModel, ADVar>(model, x_dual, u_dual, dt);

                    for (size_t j = 0; j < Nx; ++j) {
                        for (size_t i = 0; i < Nx; ++i) {
                            riccati.A[k](i, j) = x_next_dual(i).g[j];
                        }
                    }
                    for (size_t j = 0; j < Nu; ++j) {
                        for (size_t i = 0; i < Nx; ++i) {
                            riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                        }
                    }
                    for (size_t i = 0; i < Nx; ++i) {
                        riccati.d[k](i) = x_next_dual(i).v - X_pred[k + 1](i);
                    }
                }
            }
            for (size_t k = 0; k < H; ++k) {
                riccati.Q[k].set_zero();
                riccati.R[k].set_zero();
                riccati.q[k].set_zero();
                riccati.r[k].set_zero();

                double d_val = X_pred[k](1);
                double err_d = d_val - config.target_d[k];
                double err_mu = X_pred[k](2);
                double err_v = X_pred[k](3) - config.target_vx;

                riccati.Q[k](1, 1) = config.Q_D;
                riccati.q[k](1) = config.Q_D * err_d;
                riccati.Q[k](2, 2) = config.Q_mu;
                riccati.q[k](2) = config.Q_mu * err_mu;
                riccati.Q[k](3, 3) = config.Q_Vx;
                riccati.q[k](3) = config.Q_Vx * err_v;

                riccati.Q[k](4, 4) = config.Q_Vy;
                riccati.q[k](4) = config.Q_Vy * X_pred[k](4);
                riccati.Q[k](5, 5) = config.Q_r;
                riccati.q[k](5) = config.Q_r * X_pred[k](5);
                riccati.Q[k](6, 6) = config.Q_alpha_f;
                riccati.q[k](6) = config.Q_alpha_f * X_pred[k](6);
                riccati.Q[k](7, 7) = config.Q_alpha_r;
                riccati.q[k](7) = config.Q_alpha_r * X_pred[k](7);

                riccati.R[k](0, 0) = config.R_Steer;
                riccati.r[k](0) = config.R_Steer * U_guess[k](0);
                riccati.R[k](1, 1) = config.R_Accel;
                riccati.r[k](1) = config.R_Accel * U_guess[k](1);

                double violation_max = d_val - config.d_max;
                double violation_min = config.d_min - d_val;

                if (violation_max > 0.0) {
                    riccati.q[k](1) += config.W_slack * violation_max;
                    riccati.Q[k](1, 1) += config.W_slack;
                }
                
                if (violation_min > 0.0) {
                    riccati.q[k](1) -= config.W_slack * violation_min;
                    riccati.Q[k](1, 1) += config.W_slack;
                }

                double time_future = k * dt;
                for (size_t i = 0; i < 10; ++i) {
                    double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                    double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                    double ds = X_pred[k](0) - obs_pred_s;
                    double dd = d_val - obs_pred_d;
                    
                    // Symmetry breaking to prevent local minima (U-turns) when directly in front
                    double dd_eff = dd;
                    if (std::abs(dd_eff) < 0.1) dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                    
                    // Elliptical obstacle influence: stretch longitudinally by a factor of 2.0
                    double ds_scaled = ds * 0.5; 
                    double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                    double safety_margin = obstacles[i].r + config.Obstacle_Margin;
                    double violation = safety_margin * safety_margin - dist_sq;

                    if (violation > 0.0) {
                        double J_s = -2.0 * ds_scaled * 0.5; // Chain rule for ds_scaled^2 w.r.t s
                        double J_d = -2.0 * dd_eff;          // Gradient uses dd_eff

                        riccati.q[k](0) += config.Obstacle_Penalty * violation * J_s;
                        riccati.q[k](1) += config.Obstacle_Penalty * violation * J_d;

                        riccati.Q[k](0, 0) += config.Obstacle_Penalty * J_s * J_s;
                        riccati.Q[k](0, 1) += config.Obstacle_Penalty * J_s * J_d;
                        riccati.Q[k](1, 0) += config.Obstacle_Penalty * J_d * J_s;
                        riccati.Q[k](1, 1) += config.Obstacle_Penalty * J_d * J_d;
                    }
                }

                for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
                for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
            }

            riccati.Q[H].set_zero();
            riccati.q[H].set_zero();
            riccati.Q[H](1, 1) = config.Q_D * 5.0;
            riccati.q[H](1) = config.Q_D * 5.0 * (X_pred[H](1) - config.target_d[H]);
            riccati.Q[H](2, 2) = config.Q_mu * 5.0;
            riccati.q[H](2) = config.Q_mu * 5.0 * X_pred[H](2);
            riccati.Q[H](3, 3) = config.Q_Vx * 5.0;
            riccati.q[H](3) = config.Q_Vx * 5.0 * (X_pred[H](3) - config.target_vx);

            if (riccati.solve() != SolverStatus::SUCCESS) {
                return execute_fallback(result, "Riccati Factorization Failed", config);
            }

            double current_merit = evaluate_merit(0.0, config, x_curr);
            auto merit_evaluator = [&](double alpha) {
                return evaluate_merit(alpha, config, x_curr);
            };
            double optimal_alpha = solver::MeritLineSearch::run(merit_evaluator, current_merit);

            matrix::StaticVector<double, H * Nu> du_vec;
            for (size_t k = 0; k < H; ++k) {
                X_pred[k].saxpy(optimal_alpha, riccati.dx[k]);
                for (size_t i = 0; i < Nu; ++i) {
                    double step_u = optimal_alpha * riccati.du[k](i);
                    U_guess[k](i) =
                        std::clamp(U_guess[k](i) + step_u, config.u_min[i], config.u_max[i]);
                    du_vec(k * Nu + i) = step_u;
                }
            }
            X_pred[H].saxpy(optimal_alpha, riccati.dx[H]);

            double kkt_residual = Solver::KKTMonitor<H * Nu, 1>::fast_infinity_norm(du_vec);
            result.max_kkt_error = kkt_residual;

            if (std::isnan(kkt_residual) || kkt_residual > 20.0) {
                return execute_fallback(result, "KKT Divergence Detected", config);
            }

            if (kkt_residual < config.kkt_tolerance * 2.0) {
                mu = MathTraits<double>::max(1e-4, mu * 0.1);
            } else if (kkt_residual < config.kkt_tolerance * 10.0) {
                mu = MathTraits<double>::max(1e-3, mu * 0.2);
            } else {
                mu = MathTraits<double>::max(1e-2, mu * 0.5);
            }

            if (kkt_residual < config.kkt_tolerance && mu <= 1e-3) {
                result.status_msg = "IPM Converged (Fast Exit)";
                safe_buffer.commit(X_pred, U_guess);
                break;
            }
        }

        u_last = U_guess[0];
        result.success = true;
        if (result.status_msg == "OK") result.status_msg = "IPM Solved";
        safe_buffer.commit(X_pred, U_guess);
        return result;
    }
};

}  // namespace controller
}  // namespace Optimization

#endif  // OPTIMIZATION_CONTROLLER_SPARSE_NMPC_IPM_HPP_
