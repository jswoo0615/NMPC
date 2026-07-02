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
#include "Optimization/Utils/EnvironmentEvaluator.hpp" 

namespace Optimization {

    namespace controller {
        
        /**
         * @brief Structure to hold the result of the NMPC solver.
         */
        #ifndef NMPC_RESULT_DEFINED
        #define NMPC_RESULT_DEFINED
        struct NMPCResult {
            bool success = false;                ///< True if the solver converged successfully
            bool fallback_triggered = false;     ///< True if a fallback safe trajectory was triggered due to failure
            double max_kkt_error = 0.0;          ///< Infinity norm of the KKT residual
            double log10_final_merit = 0.0;      ///< Log10 of the final merit function value
            int sqp_iterations = 0;              ///< Number of IPM/SQP iterations performed
            std::string status_msg = "OK";       ///< Status message describing convergence or failure reason
        };
        #endif // NMPC_RESULT_DEFINED

        /**
         * @brief Configuration parameters and tuning weights for the NMPC solver.
         */
        #ifndef NMPC_TUNING_CONFIG_DEFINED
        #define NMPC_TUNING_CONFIG_DEFINED
        struct NMPCTuningConfig {
            // State weighting terms
            double Q_D = 200.0;         ///< Penalty for lateral deviation
            double Q_mu = 50.0;         ///< Penalty for heading error
            double Q_Vx = 50.0;         ///< Penalty for longitudinal velocity error
            double Q_Vy = 1.0;          ///< Penalty for lateral velocity
            double Q_r = 1.0;           ///< Penalty for yaw rate
            double Q_alpha_f = 10.0;    ///< Penalty for front slip angle
            double Q_alpha_r = 10.0;    ///< Penalty for rear slip angle

            // Input weighting terms
            double R_Steer = 5000.0;    ///< Penalty for steering magnitude
            double R_Accel = 10.0;      ///< Penalty for acceleration magnitude

            // Input rate weighting terms (Slew Rate/Jerk)
            double R_SteerRate = 1500.0; ///< Penalty for steering rate (jerk)
            double R_AccelRate = 100.0;  ///< Penalty for acceleration rate (jerk)

            // Vehicle & Tracking parameters
            double Q_kappa_track = 1500.0; ///< Penalty for deviating from feedforward steering
            double Q_ay = 10.0;            ///< Penalty for lateral acceleration (comfort)
            double L_wb = 3.0;             ///< Wheelbase of the vehicle

            // Obstacle & Slack penalties
            double Obstacle_Penalty = 20000.0;  ///< Legacy penalty weight for obstacles
            double Obstacle_Margin = 1.5;       ///< Safety margin radius added around obstacles
            double W_slack = 100000.0;          ///< Legacy slack weight

            // Hessian Regularization
            double damping_Q = 5.0;     ///< Damping added to state Hessian (Q) for SPD properties
            double damping_R = 500.0;   ///< Damping added to input Hessian (R) for SPD properties

            // Constraints limits
            double d_max = 3.5;         ///< Maximum lateral boundary
            double d_min = -3.5;        ///< Minimum lateral boundary
            double u_min[2] = {-0.6, -10.0}; ///< Minimum input limits: {Steer, Accel}
            double u_max[2] = {0.6, 10.0};   ///< Maximum input limits: {Steer, Accel}

            // Reference trajectory arrays
            double kappa_ref[200] = {0.0};   ///< Reference curvature along the horizon
            
            double target_vx = 10.0;         ///< Target longitudinal velocity
            double target_d[200] = {0.0};    ///< Reference lateral deviation along the horizon

            // Solver parameters
            int ipm_max_iter = 8;          ///< Maximum number of interior-point method iterations
            double kkt_tolerance = 1e-2;   ///< Threshold for KKT residual convergence
        };
        #endif // NMPC_TUNING_CONFIG_DEFINED

        /**
         * @brief Sparse Non-Linear Model Predictive Control solver using Primal-Dual Interior-Point Method.
         * 
         * @tparam H Prediction horizon length
         * @tparam PlantModel The dynamics model used for prediction and differentiation
         * @tparam Nx_mem Total state dimension (allocated memory)
         * @tparam Nx_active Active state dimension used in optimization
         * @tparam Nu Input dimension
         */
        template <std::size_t H, typename PlantModel = Dynamics::RealTimeDynamicsModel, std::size_t Nx_mem = 8, std::size_t Nx_active = 6, std::size_t Nu = 2>
        class SparseNMPC_IPM {
            public:
                /**
                 * @brief Represents the state of a single constraint in the IPM.
                 */
                struct ConstraintState {
                    double s = 1.0;     ///< Slack variable (must be > 0)
                    double lam = 1.0;   ///< Dual variable (multiplier)
                    double ds = 0.0;    ///< Newton step for slack
                    double dlam = 0.0;  ///< Newton step for dual
                };
                
                /**
                 * @brief Groups all constraint states for a single time step.
                 */
                struct IPMDuals {
                    ConstraintState d_max;      ///< Max lateral deviation constraint
                    ConstraintState d_min;      ///< Min lateral deviation constraint
                    ConstraintState u_max[2];   ///< Max input constraints
                    ConstraintState u_min[2];   ///< Min input constraints
                    ConstraintState obs[10];    ///< Obstacle avoidance constraints (up to 10)
                };

                double dt; ///< Time step for prediction [s]
                std::array<matrix::StaticVector<double, Nu>, H> U_guess;        ///< Control trajectory guess
                std::array<matrix::StaticVector<double, Nx_mem>, H + 1> X_pred; ///< Predicted state trajectory
                std::array<IPMDuals, H> duals;                                  ///< Dual variables and slacks along the horizon
                matrix::StaticVector<double, Nu> u_last;                        ///< The last applied control (used for rate constraints)

                double mu; ///< Barrier parameter for Interior-Point Method

                Evaluator::EnvironmentEvaluator<Nx_active, 10> env_evaluator; ///< Evaluates obstacle collisions
                solver::RiccatiSolver<H, Nx_active, Nu> riccati;              ///< Solves the KKT linear system
                SafeTrajectoryBuffer<H, Nx_mem, Nu> safe_buffer;              ///< Buffers safe trajectories for fallbacks

                /**
                 * @brief Constructor. Initializes trajectories and duals.
                 */
                SparseNMPC_IPM() : dt(0.05), mu(1.0) {
                    u_last.set_zero();
                    for (std::size_t k = 0; k < H; ++k) {
                        U_guess[k].set_zero();
                        duals[k] = IPMDuals();
                    }
                }

                /**
                 * @brief Shifts the prediction sequence forward by one time step.
                 * Smooths the transition and predicts the new terminal state via RK4.
                 */
                inline void shift_sequence() {
                    for (std::size_t k = 0; k < H - 1; ++k) {
                        U_guess[k] = U_guess[k + 1];
                        X_pred[k] = X_pred[k + 1];
                        duals[k] = duals[k + 1];
                    }
                    X_pred[H - 1] = X_pred[H];
                    U_guess[H - 1](0) *= 0.5;
                    U_guess[H - 1](1) *= 0.5;

                    PlantModel model;
                    model.kappa = 0.0; 
                    X_pred[H] = integrator::step_rk4<Nx_mem, Nu, PlantModel, double>(model, X_pred[H - 1], U_guess[H - 1], dt);
                }

                /**
                 * @brief Safely falls back to a previous valid trajectory if the IPM fails.
                 */
                inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason, const NMPCTuningConfig& config) {
                    res.success = false;
                    res.fallback_triggered = true;
                    res.status_msg = "IPM Fallback : " + reason;

                    matrix::StaticVector<double, Nu> safe_u = safe_buffer.generate_fallback_control(config.u_min[1]);
                    if (safe_buffer.has_valid_trajectory) {
                        for (std::size_t k = 0; k < H; ++k) {
                            U_guess[k] = safe_buffer.U_safe[k];
                        }
                        for (std::size_t k = 0; k <= H; ++k) {
                            X_pred[k] = safe_buffer.X_safe[k];
                        }
                    } else {
                        for (std::size_t k = 0; k < H; ++k) {
                            U_guess[k] = safe_u;
                        }
                    }
                    u_last = U_guess[0];
                    return res;
                }

                /**
                 * @brief Evaluates the Merit function used for the backtracking line search.
                 * Computes tracking costs, barrier penalties, and state consistency (L1 Penalty).
                 */
                double evaluate_merit(double alpha, const NMPCTuningConfig& config,
                                      const matrix::StaticVector<double, Nx_mem>& x_init, double current_mu) {
                    
                    const double max_stage_weight = std::max({
                        config.Q_D, config.Q_mu, config.Q_Vx, config.Q_Vy, config.Q_r,
                        config.Q_alpha_f, config.Q_alpha_r, config.R_Steer, config.R_Accel, config.R_SteerRate
                    });
                    const double L1_WEIGHT = std::max(1000.0, 10.0 * max_stage_weight);

                    double merit = 0.0;
                    
                    matrix::StaticVector<double, Nx_mem> x_curr = X_pred[0];
                    
                    for (std::size_t i = 0; i < Nx_active; ++i) {
                        x_curr(i) += alpha * riccati.dx[0](i);
                        merit += L1_WEIGHT * std::abs(x_curr(i) - x_init(i));
                    }

                    for (std::size_t k = 0; k < H; ++k) {
                        matrix::StaticVector<double, Nu> u_cand = U_guess[k];
                        u_cand.saxpy(alpha, riccati.du[k]);
                        
                        matrix::StaticVector<double, Nx_mem> x_next = X_pred[k + 1];
                        for (std::size_t i = 0; i < Nx_active; ++i) {
                            x_next(i) += alpha * riccati.dx[k + 1](i);
                        }

                        double err_d = x_curr(1) - config.target_d[k];
                        double err_mu = x_curr(2);
                        double err_v = x_curr(3) - config.target_vx;

                        merit += 0.5 * (config.Q_D * err_d * err_d + config.Q_mu * err_mu * err_mu + config.Q_Vx * err_v * err_v);
                        merit += 0.5 * (config.Q_Vy * x_curr(4) * x_curr(4));
                        merit += 0.5 * (config.Q_r * x_curr(5) * x_curr(5));
                        merit += 0.5 * (config.Q_alpha_f * x_curr(6) * x_curr(6));
                        merit += 0.5 * (config.Q_alpha_r * x_curr(7) * x_curr(7));

                        merit += 0.5 * (config.R_Steer * u_cand(0) * u_cand(0) + config.R_Accel * u_cand(1) * u_cand(1));
                        
                        // [아키텍트의 수술: 전체 지평에 대한 Jerk 및 Steer Rate Merit 평가 복원]
                        matrix::StaticVector<double, Nu> u_prev_cand;
                        if (k == 0) {
                            u_prev_cand = u_last;
                        } else {
                            u_prev_cand = U_guess[k-1];
                            u_prev_cand.saxpy(alpha, riccati.du[k-1]);
                        }
                        double delta_steer = u_cand(0) - u_prev_cand(0);
                        double delta_accel = u_cand(1) - u_prev_cand(1);
                        merit += 0.5 * (config.R_SteerRate * delta_steer * delta_steer + config.R_AccelRate * delta_accel * delta_accel);

                        double delta_ff = config.L_wb * config.kappa_ref[k];
                        double err_steer_track = u_cand(0) - delta_ff;
                        merit += 0.5 * (config.Q_kappa_track * err_steer_track * err_steer_track);

                        double a_y = x_curr(3) * x_curr(5);
                        merit += 0.5 * (config.Q_ay * a_y * a_y);

                        // Evaluates the log-barrier and the Exact L1 penalty
                        auto eval_barrier = [&](double c, double s, double ds) {
                            double s_cand = s + alpha * ds;
                            if (s_cand <= 1e-8) {
                                s_cand = 1e-8;
                            }
                            return -current_mu * std::log(s_cand) + L1_WEIGHT * std::abs(c + s_cand);
                        };

                        double c_d_max = x_curr(1) - config.d_max;
                        merit += eval_barrier(c_d_max, duals[k].d_max.s, duals[k].d_max.ds);

                        double c_d_min = config.d_min - x_curr(1);
                        merit += eval_barrier(c_d_min, duals[k].d_min.s, duals[k].d_min.ds);

                        for (std::size_t i = 0; i < 2; ++i) {
                            double c_u_max = u_cand(i) - config.u_max[i];
                            merit += eval_barrier(c_u_max, duals[k].u_max[i].s, duals[k].u_max[i].ds);
                            double c_u_min = config.u_min[i] - u_cand(i);
                            merit += eval_barrier(c_u_min, duals[k].u_min[i].s, duals[k].u_min[i].ds);
                        }

                        double time_future = k * dt;
                        auto obs_grads = env_evaluator.evaluate_obstacles(x_curr(0), x_curr(1), time_future);
                        
                        for (std::size_t i = 0; i < 10; ++i) {
                            if (!obs_grads[i].is_active) continue;
                            merit += eval_barrier(obs_grads[i].c_val, duals[k].obs[i].s, duals[k].obs[i].ds);
                        }

                        for (std::size_t i = 0; i < Nx_active; ++i) {
                            merit += L1_WEIGHT * std::abs((1.0 - alpha) * riccati.d[k](i));
                        }
                        x_curr = x_next;
                    }
                    return merit;
                }

                /**
                 * @brief Main entry point to solve the Sparse NMPC problem using Primal-Dual IPM.
                 */
                NMPCResult solve_ipm(const matrix::StaticVector<double, Nx_mem>& x_curr, const NMPCTuningConfig& config) {
                    NMPCResult result;
                    result.sqp_iterations = 0;

                    // 1. NaN checks
                    if (std::isnan(x_curr(1)) || std::isnan(x_curr(3))) {
                        return execute_fallback(result, "Primal State NaN Detected", config);
                    }
                    
                    env_evaluator.obstacle_margin = config.Obstacle_Margin;

                    // 2. Initialize prediction states
                    X_pred[0] = x_curr;
                    X_pred[0](3) = MathTraits<double>::max(0.1, x_curr(3));

                    mu = 1.0; // Reset barrier parameter

                    PlantModel model_forward;
                    // Forward pass to initialize slacks and dual variables
                    for (std::size_t k = 0; k < H; ++k) {
                        model_forward.kappa = config.kappa_ref[k]; 
                        
                        X_pred[k + 1] = integrator::step_rk4<Nx_mem, Nu, PlantModel, double>(model_forward, X_pred[k], U_guess[k], dt);
                        double d_val = X_pred[k](1);
                        auto init_dual = [&](double c, ConstraintState& cs) {
                            cs.s = MathTraits<double>::max(cs.s, MathTraits<double>::max(0.1, -c));
                            cs.lam = mu / cs.s;
                        };

                        init_dual(d_val - config.d_max, duals[k].d_max);
                        init_dual(config.d_min - d_val, duals[k].d_min);

                        for (std::size_t i = 0; i < 2; ++i) {
                            init_dual(U_guess[k](i) - config.u_max[i], duals[k].u_max[i]);
                            init_dual(config.u_min[i] - U_guess[k](i), duals[k].u_min[i]);
                        }
                        
                        double time_future = k * dt;
                        auto obs_grads = env_evaluator.evaluate_obstacles(X_pred[k](0), d_val, time_future);
                        
                        for (std::size_t i = 0; i < 10; ++i) {
                            if (!obs_grads[i].is_active) continue;
                            init_dual(obs_grads[i].c_val, duals[k].obs[i]);
                        }
                    }

                    // 3. Main IPM/SQP Loop
                    for (int ipm_step = 0; ipm_step < config.ipm_max_iter; ++ipm_step) {
                        result.sqp_iterations++;

                        // Update Jacobian only every other iteration via Automatic Differentiation (AD)
                        bool update_jacobian = (ipm_step % 2 == 0);

                        if (update_jacobian) {
                            using ADVar = matrix::DualVec<double, Nx_active + Nu>;
                            PlantModel model_ad;
                            for (std::size_t k = 0; k < H; ++k) {
                                model_ad.kappa = config.kappa_ref[k];

                                matrix::StaticVector<ADVar, Nx_mem> x_dual;
                                matrix::StaticVector<ADVar, Nu> u_dual;

                                // Inject Dual variables for AD gradient tracking
                                for (std::size_t i = 0; i < Nx_active; ++i) {
                                    x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                                }
                                for (std::size_t i = Nx_active; i < Nx_mem; ++i) {
                                    x_dual(i) = ADVar();
                                    x_dual(i).v = X_pred[k](i);
                                }
                                for (std::size_t i = 0; i < Nu; ++i) {
                                    u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx_active + i);    
                                }

                                matrix::StaticVector<ADVar, Nx_mem> x_next_dual = 
                                    integrator::step_rk4<Nx_mem, Nu, PlantModel, ADVar>(model_ad, x_dual, u_dual, dt);

                                // Extract Jacobian A and B matrices from Dual variable gradients
                                for (std::size_t j = 0; j < Nx_active; ++j) {
                                    for (std::size_t i = 0; i < Nx_active; ++i) {
                                        riccati.A[k](i, j) = x_next_dual(i).g[j];
                                    }
                                }
                                for (std::size_t j = 0; j < Nu; ++j) {
                                    for (std::size_t i = 0; i < Nx_active; ++i) {
                                        riccati.B[k](i, j) = x_next_dual(i).g[Nx_active + j];
                                    }
                                }

                                for (std::size_t i = 0; i < Nx_active; ++i) {
                                    riccati.d[k](i) = x_next_dual(i).v - X_pred[k + 1](i);
                                }
                            }
                        }

                        double gap_sum = 0.0;
                        int num_constraints = 0;

                        // Build Q, R, q, r matrices for Riccati solver
                        for (std::size_t k = 0; k < H; ++k) {
                            riccati.Q[k].set_zero();
                            riccati.R[k].set_zero();
                            riccati.q[k].set_zero();
                            riccati.r[k].set_zero();

                            double d_val = X_pred[k](1);
                            double err_d = d_val - config.target_d[k];
                            double err_mu = X_pred[k](2);
                            double err_v = X_pred[k](3) - config.target_vx;

                            // Stage costs
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
                            
                            riccati.R[k](0, 0) = config.R_Steer;
                            riccati.r[k](0) = config.R_Steer * U_guess[k](0);
                            riccati.R[k](1, 1) = config.R_Accel;
                            riccati.r[k](1) = config.R_Accel * U_guess[k](1);

                            // [아키텍트의 수술: SQP Gradient Injection을 통한 전체 지평 Jerk 제어]
                            // 오프 대각선(Off-diagonal) 결합을 피하면서, SQP의 1차 미분(Gradient)을 통해 
                            // 100스텝 전체의 변화율(Slew Rate/Jerk) 비용을 완벽하게 소급시킵니다.
                            matrix::StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k-1];
                            double diff_steer = U_guess[k](0) - u_prev(0);
                            double diff_accel = U_guess[k](1) - u_prev(1);

                            riccati.R[k](0, 0) += config.R_SteerRate;
                            riccati.R[k](1, 1) += config.R_AccelRate;
                            riccati.r[k](0) += config.R_SteerRate * diff_steer;
                            riccati.r[k](1) += config.R_AccelRate * diff_accel;

                            if (k > 0) {
                                // 현재 스텝의 변화율 페널티에 대한 이전 스텝(k-1)의 기여도를 체인 룰(Chain Rule)로 분배
                                riccati.R[k-1](0, 0) += config.R_SteerRate;
                                riccati.R[k-1](1, 1) += config.R_AccelRate;
                                riccati.r[k-1](0) -= config.R_SteerRate * diff_steer;
                                riccati.r[k-1](1) -= config.R_AccelRate * diff_accel;
                            }

                            // Curvature Tracking & Lateral Acceleration
                            double delta_ff = config.L_wb * config.kappa_ref[k];
                            double err_steer_track = U_guess[k](0) - delta_ff;
                            riccati.R[k](0, 0) += config.Q_kappa_track;
                            riccati.r[k](0) += config.Q_kappa_track * err_steer_track;

                            double v_x = X_pred[k](3);
                            double r_rate = X_pred[k](5);
                            double a_y = v_x * r_rate;

                            riccati.q[k](3) += config.Q_ay * a_y * r_rate;
                            riccati.q[k](5) += config.Q_ay * a_y * v_x;

                            riccati.Q[k](3, 3) += config.Q_ay * r_rate * r_rate;
                            riccati.Q[k](5, 5) += config.Q_ay * v_x * v_x;
                            riccati.Q[k](3, 5) += config.Q_ay * v_x * r_rate;
                            riccati.Q[k](5, 3) += config.Q_ay * v_x * r_rate;

                            // Applies the condensed log-barrier constraints into the KKT system
                            auto apply_condensed = [&](double c, double J_x0, double J_x1, double J_u0, double J_u1, ConstraintState& cs) {
                                double J_term = (mu + cs.lam * c) / cs.s;
                                double H_term = cs.lam / cs.s;

                                if (J_x0 != 0.0) {
                                    riccati.q[k](0) += J_x0 * J_term;
                                    riccati.Q[k](0, 0) += J_x0 * J_x0 * H_term;
                                }
                                if (J_x1 != 0.0) {
                                    riccati.q[k](1) += J_x1 * J_term;
                                    riccati.Q[k](1, 1) += J_x1 * J_x1 * H_term;
                                }
                                if (J_x0 != 0.0 && J_x1 != 0.0) {
                                    riccati.Q[k](0, 1) += J_x0 * J_x1 * H_term;
                                    riccati.Q[k](1, 0) += J_x1 * J_x0 * H_term;
                                }
                                if (J_u0 != 0.0) {
                                    riccati.r[k](0) += J_u0 * J_term;
                                    riccati.R[k](0, 0) += J_u0 * J_u0 * H_term;
                                }
                                if (J_u1 != 0.0) {
                                    riccati.r[k](1) += J_u1 * J_term;
                                    riccati.R[k](1, 1) += J_u1 * J_u1 * H_term;
                                }

                                gap_sum += cs.s * cs.lam;
                                num_constraints++;
                            };

                            apply_condensed(d_val - config.d_max, 0.0, 1.0, 0.0, 0.0, duals[k].d_max);
                            apply_condensed(config.d_min - d_val, 0.0, -1.0, 0.0, 0.0, duals[k].d_min);

                            for (std::size_t i = 0; i < 2; ++i) {
                                apply_condensed(U_guess[k](i) - config.u_max[i], 0.0, 0.0, i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, duals[k].u_max[i]);
                                apply_condensed(config.u_min[i] - U_guess[k](i), 0.0, 0.0, i == 0 ? -1.0 : 0.0, i == 1 ? -1.0 : 0.0, duals[k].u_min[i]);
                            }

                            double time_future = k * dt;
                            auto obs_grads = env_evaluator.evaluate_obstacles(X_pred[k](0), d_val, time_future);
                            
                            for (std::size_t i = 0; i < 10; ++i) {
                                if (!obs_grads[i].is_active) continue;
                                apply_condensed(
                                    obs_grads[i].c_val, 
                                    obs_grads[i].J_x(0), 
                                    obs_grads[i].J_x(1), 
                                    0.0, 0.0, 
                                    duals[k].obs[i]
                                );
                            }

                            // Hessian Regularization to ensure SPD property
                            for (std::size_t i = 0; i < Nx_active; ++i) {
                                riccati.Q[k](i, i) += config.damping_Q;
                            }
                            for (std::size_t i = 0; i < Nu; ++i) {
                                riccati.R[k](i, i) += config.damping_R;
                            }
                        }

                        // Terminal Cost Configuration
                        riccati.Q[H].set_zero();
                        riccati.q[H].set_zero();
                        riccati.Q[H](1, 1) = config.Q_D * 5.0;
                        riccati.q[H](1) = config.Q_D * 5.0 * (X_pred[H](1) - config.target_d[H]);
                        riccati.Q[H](2, 2) = config.Q_mu * 5.0;
                        riccati.q[H](2) = config.Q_mu * 5.0 * X_pred[H](2);
                        riccati.Q[H](3, 3) = config.Q_Vx * 5.0;
                        riccati.q[H](3) = config.Q_Vx * 5.0 * (X_pred[H](3) - config.target_vx);

                        // 4. Solve the KKT System
                        if (riccati.solve() != SolverStatus::SUCCESS) {
                            return execute_fallback(result, "Riccati Factorization Failed", config);
                        }

                        double alpha_max = 1.0;
                        constexpr double tau = 0.995;

                        matrix::StaticVector<double, H * Nu> du_vec;

                        // 5. Extract Dual Steps and Compute max line search step (alpha_max)
                        for (std::size_t k = 0; k < H; ++k) {
                            auto extract_dual = [&](double c, double J_x0, double J_x1, double J_u0, double J_u1, ConstraintState& cs) {
                                double dz = J_x0 * riccati.dx[k](0) + J_x1 * riccati.dx[k](1) + J_u0 * riccati.du[k](0) + J_u1 * riccati.du[k](1);

                                cs.ds = -c -cs.s -dz;
                                cs.dlam = (mu - cs.lam * cs.s - cs.lam * cs.ds) / cs.s;

                                if (cs.ds < 0.0) {
                                    alpha_max = std::min(alpha_max, -tau * cs.s / cs.ds);
                                }
                                if (cs.dlam < 0.0) {
                                    alpha_max = std::min(alpha_max, -tau * cs.lam / cs.dlam);
                                }
                            };

                            double d_val = X_pred[k](1);
                            
                            extract_dual(d_val - config.d_max, 0.0, 1.0, 0.0, 0.0, duals[k].d_max);
                            extract_dual(config.d_min - d_val, 0.0, -1.0, 0.0, 0.0, duals[k].d_min);

                            for (std::size_t i = 0; i < 2; ++i) {
                                extract_dual(U_guess[k](i) - config.u_max[i], 0.0, 0.0, i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, duals[k].u_max[i]);
                                extract_dual(config.u_min[i] - U_guess[k](i), 0.0, 0.0, i == 0 ? -1.0 : 0.0, i == 1 ? -1.0 : 0.0, duals[k].u_min[i]);
                            }

                            double time_future = k * dt;
                            auto obs_grads = env_evaluator.evaluate_obstacles(X_pred[k](0), d_val, time_future);
                            
                            for (std::size_t i = 0; i < 10; ++i) {
                                if (!obs_grads[i].is_active) continue;
                                extract_dual(
                                    obs_grads[i].c_val, 
                                    obs_grads[i].J_x(0), 
                                    obs_grads[i].J_x(1), 
                                    0.0, 0.0, 
                                    duals[k].obs[i]
                                );
                            }

                            for (std::size_t i = 0; i < Nu; ++i) {
                                du_vec(k * Nu + i) = riccati.du[k](i);
                            }
                        }

                        // 6. Backtracking Line Search using Merit Function
                        double current_merit = evaluate_merit(0.0, config, x_curr, mu);
                        
                        if (std::isnan(current_merit) || std::isinf(current_merit)) {
                            return execute_fallback(result, "Current Merit NaN/Inf Detected", config);
                        }

                        auto merit_evaluator = [&](double alpha) {
                            return evaluate_merit(alpha, config, x_curr, mu);
                        };

                        double optimal_alpha = alpha_max;
                        for (int ls = 0; ls < 10; ++ls) {
                            double trial_merit = merit_evaluator(optimal_alpha);
                            
                            if (std::isnan(trial_merit) || std::isinf(trial_merit)) {
                                optimal_alpha *= 0.5;
                                continue;
                            }

                            // Armijo Condition
                            double armijo_threshold = current_merit + 1e-4 * std::max(1.0, std::abs(current_merit));
                            if (trial_merit <= armijo_threshold) {
                                break;
                            }
                            optimal_alpha *= 0.5;
                        }

                        // 7. Update Primal and Dual trajectories
                        for (std::size_t k = 0; k < H; ++k) {
                            for (std::size_t i = 0; i < Nx_active; ++i) {
                                X_pred[k](i) += optimal_alpha * riccati.dx[k](i);
                            }
                            for (std::size_t i = 0; i < Nu; ++i) {
                                U_guess[k](i) += optimal_alpha * riccati.du[k](i);
                            }

                            auto update_dual = [&](ConstraintState& cs) {
                                cs.s += optimal_alpha * cs.ds;
                                cs.lam += optimal_alpha * cs.dlam;
                            };

                            update_dual(duals[k].d_max);
                            update_dual(duals[k].d_min);
                            for (int i = 0; i < 2; ++i) {
                                update_dual(duals[k].u_max[i]);
                                update_dual(duals[k].u_min[i]);
                            }
                            for (int i = 0; i < 10; ++i) {
                                update_dual(duals[k].obs[i]);
                            }
                        }
                        for (std::size_t i = 0; i < Nx_active; ++i) {
                            X_pred[H](i) += optimal_alpha * riccati.dx[H](i);
                        }

                        // 8. Convergence Check
                        double kkt_residual = Solver::KKTMonitor<H * Nu, 1>::fast_infinity_norm(du_vec);
                        result.max_kkt_error = kkt_residual;
                        result.log10_final_merit = std::log10(std::max(1e-9, current_merit));

                        if (std::isnan(kkt_residual) || kkt_residual > 20.0) {
                            return execute_fallback(result, "KKT Divergence Detected", config);
                        }
                        double average_gap = gap_sum / num_constraints;
                        mu = MathTraits<double>::max(1e-4, 0.2 * average_gap);

                        // Early exit if converged
                        if (kkt_residual < config.kkt_tolerance && average_gap <= 1e-3) {
                            result.status_msg = "IPM Converged (Fast Exit)";
                            safe_buffer.commit(X_pred, U_guess);
                            break;
                        }
                    }

                    // 9. Finalize
                    u_last = U_guess[0];
                    result.success = true;
                    if (result.status_msg == "OK") {
                        result.status_msg = "IPM Solved";
                    }
                    safe_buffer.commit(X_pred, U_guess);
                    return result;
                }
        }; 

    } // namespace controller
} // namespace Optimization

#endif // OPTIMIZATION_CONTROLLER_SPARSE_NMPC_IPM_HPP_