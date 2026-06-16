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
    #endif // OBSTACLE_FRENET_DEFINED

    namespace controller {
        #ifndef NMPC_RESULT_DEFINED
        #define NMPC_RESULT_DEFINED
        struct NMPCResult {
            bool success = false;
            bool fallback_triggered = false;
            double max_kkt_error = 0.0;
            double log10_final_merit = 0.0; // [C] 로그 스케일 디버깅 모니터링 지표 추가
            int sqp_iterations = 0;
            std::string status_msg = "OK";
        };
        #endif // NMPC_RESULT_DEFINED

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

            double Obstacle_Penalty = 20000.0;  // Not used in Hard PD-IPM, but kept for interface compatibility
            double Obstacle_Margin = 1.5;

            double W_slack = 100000.0;          // Not used in Hard PD-IPM

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
        #endif // NMPC_TUNING_CONFIG_DEFINED

        template <std::size_t H, typename PlantModel = Dynamics::RealTimeDynamicsModel, std::size_t Nx = 8, std::size_t Nu = 2>
        class SparseNMPC_IPM {
            public:
                struct ConstraintState {
                    double s = 1.0;
                    double lam = 1.0;
                    double ds = 0.0;
                    double dlam = 0.0;
                };
                struct IPMDuals {
                    ConstraintState d_max;
                    ConstraintState d_min;
                    ConstraintState u_max[2];
                    ConstraintState u_min[2];
                    ConstraintState obs[10];
                };

                double dt;
                std::array<matrix::StaticVector<double, Nu>, H> U_guess;
                std::array<matrix::StaticVector<double, Nx>, H + 1> X_pred;
                std::array<IPMDuals, H> duals;
                matrix::StaticVector<double, Nu> u_last;

                std::array<ObstacleFrenet, 10> obstacles;
                double mu;

                solver::RiccatiSolver<H, Nx, Nu> riccati;
                SafeBuffer<H, Nx, Nu> safe_buffer;

                SparseNMPC_IPM() : dt(0.05), mu(1.0) {
                    u_last.set_zero();
                    for (std::size_t k = 0; k < H; ++k) {
                        U_guess[k].set_zero();
                        duals[k] = IPMDuals();
                    }
                    for (auto& obs : obstacles) {
                        obs.s = 10000.0;
                        obs.d = 10000.0;
                        obs.r = 0.1;
                        obs.vs = 0.0;
                        obs.vd = 0.0;
                    }
                }

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
                    X_pred[H] = integrator::step_rk4<Nx, Nu, PlantModel, double>(model, X_pred[H - 1], U_guess[H - 1], dt);
                }

                inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason, const NMPCTuningConfig& config) {
                    res.success = false;
                    res.fallback_triggered = true;
                    res.status_msg = "IPM Fallback : " + reason;

                    matrix::StaticVector<double, Nu> safe_u = safe_buffer.extract_fallback_control(config.u_min[1]);
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

                double evaluate_merit(double alpha, const NMPCTuningConfig& config,
                                      const matrix::StaticVector<double, Nx>& x_init, double current_mu) {
                    
                    // [A] 가중치 황금률 자동 강제 체계 구축
                    // 하드코딩된 1000.0 대신 스테이지 코스트 최댓값의 최소 10배 이상을 동적으로 보장
                    const double max_stage_weight = std::max({
                        config.Q_D, config.Q_mu, config.Q_Vx, config.Q_Vy, config.Q_r,
                        config.Q_alpha_f, config.Q_alpha_r, config.R_Steer, config.R_Accel
                    });
                    const double L1_WEIGHT = std::max(1000.0, 10.0 * max_stage_weight);

                    double merit = 0.0;
                    
                    // PlantModel 관련 파현 도려냄 (Dead Code 제거 완료)

                    matrix::StaticVector<double, Nx> x_curr = X_pred[0];
                    x_curr.saxpy(alpha, riccati.dx[0]);
                    for (std::size_t i = 0; i < Nx; ++i) {
                        merit += L1_WEIGHT * std::abs(x_curr(i) - x_init(i));
                    }

                    for (std::size_t k = 0; k < H; ++k) {
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

                        merit += 0.5 * (config.R_Steer * u_cand(0) * u_cand(0) + config.R_Accel * u_cand(1) * u_cand(1));
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
                        for (std::size_t i = 0; i < 10; ++i) {
                            double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                            double ds = x_curr(0) - obs_pred_s;

                            if (std::abs(ds) > 20.0) {
                                continue;
                            }
                            double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                            double dd = x_curr(1) - obs_pred_d;

                            double dd_eff = dd;
                            if (std::abs(dd_eff) < 0.1) {
                                dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                            }
                            double ds_scaled = ds * 0.5;
                            double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                            double safety_margin = obstacles[i].r + config.Obstacle_Margin;
                            double c_obs = safety_margin * safety_margin - dist_sq;

                            merit += eval_barrier(c_obs, duals[k].obs[i].s, duals[k].obs[i].ds);
                        }

                        for (std::size_t i = 0; i < Nx; ++i) {
                            merit += L1_WEIGHT * std::abs((1.0 - alpha) * riccati.d[k](i));
                        }
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

                    mu = 1.0;

                    for (std::size_t k = 0; k < H; ++k) {
                        X_pred[k + 1] = integrator::step_rk4<Nx, Nu, PlantModel, double>(model, X_pred[k], U_guess[k], dt);
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
                        for (std::size_t i = 0; i < 10; ++i) {
                            double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                            double ds = X_pred[k](0) - obs_pred_s;

                            if (std::abs(ds) > 20.0) {
                                continue;
                            }

                            double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                            double dd = d_val - obs_pred_d;

                            double dd_eff = dd;
                            if (std::abs(dd_eff) < 0.1) {
                                dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                            }
                            double ds_scaled = ds * 0.5;
                            double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                            double safety_margin = obstacles[i].r + config.Obstacle_Margin;
                            double c_obs = safety_margin * safety_margin - dist_sq;

                            init_dual(c_obs, duals[k].obs[i]);
                        }
                    }

                    for (int ipm_step = 0; ipm_step < config.ipm_max_iter; ++ipm_step) {
                        result.sqp_iterations++;

                        bool update_jacobian = (ipm_step % 2 == 0);

                        if (update_jacobian) {
                            using ADVar = matrix::DualVec<double, Nx + Nu>;
                            for (std::size_t k = 0; k < H; ++k) {
                                matrix::StaticVector<ADVar, Nx> x_dual;
                                matrix::StaticVector<ADVar, Nu> u_dual;

                                for (std::size_t i = 0; i < Nx; ++i) {
                                    x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                                }
                                for (std::size_t i = 0; i < Nu; ++i) {
                                    u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);    
                                }

                                matrix::StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu, PlantModel, ADVar>(model, x_dual, u_dual, dt);

                                for (std::size_t j = 0; j < Nx; ++j) {
                                    for (std::size_t i = 0; i < Nx; ++i) {
                                        riccati.A[k](i, j) = x_next_dual(i).g[j];
                                    }
                                }
                                for (std::size_t j = 0; j < Nu; ++j) {
                                    for (std::size_t i = 0; i < Nx; ++i) {
                                        riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                                    }
                                }

                                for (std::size_t i = 0; i < Nx; ++i) {
                                    riccati.d[k](i) = x_next_dual(i).v - X_pred[k + 1](i);
                                }
                            }
                        }

                        double gap_sum = 0.0;
                        int num_constraints = 0;

                        for (std::size_t k = 0; k < H; ++k) {
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
                            for (std::size_t i = 0; i < 10; ++i) {
                                double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                                double ds = X_pred[k](0) - obs_pred_s;

                                if (std::abs(ds) > 20.0) {
                                    continue;
                                }
                                double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                                double dd = d_val - obs_pred_d;

                                double dd_eff = dd;
                                if (std::abs(dd_eff) < 0.1) {
                                    dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                                }
                                double ds_scaled = ds * 0.5;
                                double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                                double safety_margin = obstacles[i].r + config.Obstacle_Margin;

                                apply_condensed(safety_margin * safety_margin - dist_sq, -2.0 * ds_scaled * 0.5, -2.0 * dd_eff, 0.0, 0.0, duals[k].obs[i]);
                            }

                            for (std::size_t i = 0; i < Nx; ++i) {
                                riccati.Q[k](i, i) += config.damping_Q;
                            }
                            for (std::size_t i = 0; i < Nu; ++i) {
                                riccati.R[k](i, i) += config.damping_R;
                            }
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

                        double alpha_max = 1.0;
                        constexpr double tau = 0.995;

                        matrix::StaticVector<double, H * Nu> du_vec;

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
                            for (std::size_t i = 0; i < 10; ++i) {
                                double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                                double ds = X_pred[k](0) - obs_pred_s;

                                if (std::abs(ds) > 20.0) {
                                    continue;
                                }

                                double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                                double dd = d_val - obs_pred_d;

                                double dd_eff = dd;
                                if (std::abs(dd_eff) < 0.1) {
                                    dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                                }

                                double ds_scaled = ds * 0.5;
                                double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                                double safety_margin = obstacles[i].r + config.Obstacle_Margin;

                                extract_dual(safety_margin * safety_margin - dist_sq, -2.0 * ds_scaled * 0.5, -2.0 * dd_eff, 0.0, 0.0, duals[k].obs[i]);
                            }

                            for (std::size_t i = 0; i < Nu; ++i) {
                                du_vec(k * Nu + i) = riccati.du[k](i);
                            }
                        }

                        double current_merit = evaluate_merit(0.0, config, x_curr, mu);
                        
                        // [B] 수치적 예외 처리 브레이크 배치
                        if (std::isnan(current_merit) || std::isinf(current_merit)) {
                            return execute_fallback(result, "Current Merit NaN/Inf Detected", config);
                        }

                        auto merit_evaluator = [&](double alpha) {
                            return evaluate_merit(alpha, config, x_curr, mu);
                        };

                        // Limit line search to alpha_max
                        double optimal_alpha = alpha_max;
                        for (int ls = 0; ls < 10; ++ls) {
                            double trial_merit = merit_evaluator(optimal_alpha);
                            
                            if (std::isnan(trial_merit) || std::isinf(trial_merit)) {
                                optimal_alpha *= 0.5;
                                continue;
                            }

                            // [B] 수치 정밀도 저하(Round-off error) 방지를 위한 적응형 Armijo 임계값 도입
                            double armijo_threshold = current_merit + 1e-4 * std::max(1.0, std::abs(current_merit));
                            if (trial_merit <= armijo_threshold) {
                                break;
                            }
                            optimal_alpha *= 0.5;
                        }

                        for (std::size_t k = 0; k < H; ++k) {
                            X_pred[k].saxpy(optimal_alpha, riccati.dx[k]);
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
                        X_pred[H].saxpy(optimal_alpha, riccati.dx[H]);

                        double kkt_residual = Solver::KKTMonitor<H * Nu, 1>::fast_infinity_norm(du_vec);
                        result.max_kkt_error = kkt_residual;
                        
                        // [C] 로그 스케일 모니터링 값 업데이트 (수치 진동 및 발산 추적 유틸리티용)
                        result.log10_final_merit = std::log10(std::max(1e-9, current_merit));

                        if (std::isnan(kkt_residual) || kkt_residual > 20.0) {
                            return execute_fallback(result, "KKT Divergence Detected", config);
                        }
                        double average_gap = gap_sum / num_constraints;
                        mu = MathTraits<double>::max(1e-4, 0.2 * average_gap);

                        if (kkt_residual < config.kkt_tolerance && average_gap <= 1e-3) {
                            result.status_msg = "IPM Converged (Fast Exit)";
                            safe_buffer.commit(X_pred, U_guess);
                            break;
                        }
                    }

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