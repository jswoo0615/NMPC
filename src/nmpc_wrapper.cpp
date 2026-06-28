#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include "Optimization/Control/SparseNMPC_IPM.hpp"
#include "Optimization/Evaluator/StaticCubicSpline.hpp"

namespace py = pybind11;
using namespace Optimization;

class SparseNMPCWrapper {
    private:
        constexpr static size_t H = 100;
        constexpr static size_t Nx_mem = 8;
        constexpr static size_t Nx_active = 6;
        constexpr static size_t Nu = 2;
        constexpr static size_t MaxWp = 200;

        controller::SparseNMPC_IPM<H, Dynamics::RealTimeDynamicsModel, Nx_mem, Nx_active, Nu> nmpc;
        controller::NMPCTuningConfig config;
        Evaluator::StaticCubicSpline2D<MaxWp> spline;

        double* ego_state_ptr;
        double* wp_x_ptr;
        double* wp_y_ptr;
        double* obstacle_ptr;

        double last_s_projection = 0.0;
        bool spline_valid = false;

        constexpr static double lf = 1.4;
        constexpr static double lr = 1.6;

    public:
        SparseNMPCWrapper(py::array_t<double> ego_state_arr,
                          py::array_t<double> wp_x_arr,
                          py::array_t<double> wp_y_arr,
                          py::array_t<double> obstacle_arr) {
            py::buffer_info ego_buf = ego_state_arr.request();
            py::buffer_info wp_x_buf = wp_x_arr.request();
            py::buffer_info wp_y_buf = wp_y_arr.request();
            py::buffer_info obs_buf = obstacle_arr.request();

            ego_state_ptr = static_cast<double*>(ego_buf.ptr);
            wp_x_ptr = static_cast<double*>(wp_x_buf.ptr);
            wp_y_ptr = static_cast<double*>(wp_y_buf.ptr);
            obstacle_ptr = static_cast<double*>(obs_buf.ptr);

            // [아키텍트의 수술: 90km/h 주행을 위한 예측/물리 밸런스 완벽 동기화]
            config.target_vx = 25.0;  // 모델 내부의 마찰원 계산 기준을 현실 속도(90km/h)와 일치시킴
            
            config.Q_D = 100.0;       
            config.Q_mu = 50.0;       
            config.Q_Vx = 5.0;        
            config.Q_Vy = 20.0;       
            config.Q_r = 20.0;        

            // 슬립각에 대한 페널티를 자연스러운 물리 현상 수준으로 하향
            config.Q_alpha_f = 200.0;
            config.Q_alpha_r = 200.0;

            config.R_Steer = 50.0;    
            config.R_Accel = 1.0;     

            // [Phase 2] 조향 발작 원천 차단: Steer Rate Penalty 강력 주입
            // 10ms 단위로 핸들을 급격히 꺾는 것에 엄청난 벌금을 매겨, 조향의 기계적 관성을 수학적으로 강제함
            config.R_SteerRate = 5000.0; 

            config.Obstacle_Margin = 0.0;

            // 기계적 한계는 완전히 열어둠. 제약(Constraint)이 아니라 비용(Cost)으로 제어하는 것이 진짜 NMPC.
            config.u_min[0] = -0.6; 
            config.u_max[0] = 0.6;
            config.u_min[1] = -3.0; 
            config.u_max[1] = 3.0;
        }

        void set_target_speed(double speed) {
            config.target_vx = speed;
        }

        void update_config(py::dict opt_dict) {
            if (opt_dict.contains("Q_D")) config.Q_D = opt_dict["Q_D"].cast<double>();
            if (opt_dict.contains("Q_mu")) config.Q_mu = opt_dict["Q_mu"].cast<double>();
            if (opt_dict.contains("Q_Vx")) config.Q_Vx = opt_dict["Q_Vx"].cast<double>();
            if (opt_dict.contains("Q_Vy")) config.Q_Vy = opt_dict["Q_Vy"].cast<double>();
            if (opt_dict.contains("R_Steer")) config.R_Steer = opt_dict["R_Steer"].cast<double>();
            if (opt_dict.contains("R_Accel")) config.R_Accel = opt_dict["R_Accel"].cast<double>();
            if (opt_dict.contains("R_SteerRate")) config.R_SteerRate = opt_dict["R_SteerRate"].cast<double>();
        }

        struct FrenetState {
            double s, d, mu;
        };

        FrenetState project_vehicle_state(double x, double y, double yaw) {
            if (!spline_valid) return {0.0, 0.0, 0.0};

            double s_center = last_s_projection;
            double best_s = s_center;
            double best_dist2 = std::numeric_limits<double>::max();

            const double search_window = 15.0;
            const double coarse_step = 0.5;

            for (double ds = -search_window; ds <= search_window; ds += coarse_step) {
                double s = std::clamp(s_center + ds, 0.0, spline.get_max_s());
                double dx = spline.calc_x(s) - x;
                double dy = spline.calc_y(s) - y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_s = s;
                }
            }

            double s_opt = best_s;
            for (int iter = 0; iter < 5; ++iter) {
                double px = spline.calc_x(s_opt);
                double py = spline.calc_y(s_opt);
                double dx_ds = spline.sx.calc_d1(s_opt);
                double dy_ds = spline.sy.calc_d1(s_opt);
                double ddx_ds = spline.sx.calc_d2(s_opt);
                double ddy_ds = spline.sy.calc_d2(s_opt);
                double ex = px - x;
                double ey = py - y;

                double f = ex * dx_ds + ey * dy_ds; 
                double df = dx_ds * dx_ds + dy_ds * dy_ds + ex * ddx_ds + ey * ddy_ds; 

                if (std::abs(df) < 1e-8) break;

                s_opt -= f / df;
                s_opt = std::clamp(s_opt, 0.0, spline.get_max_s());
            }

            double ref_yaw = spline.calc_yaw(s_opt);
            double d = -(x - spline.calc_x(s_opt)) * std::sin(ref_yaw) + (y - spline.calc_y(s_opt)) * std::cos(ref_yaw);
            double mu = yaw - ref_yaw;

            while (mu > M_PI) mu -= 2.0 * M_PI;
            while (mu < -M_PI) mu += 2.0 * M_PI;

            last_s_projection = s_opt;
            return {s_opt, d, mu};
        }

        py::tuple solve(int num_wp) {
            std::array<double, MaxWp> wp_x_arr, wp_y_arr;
            std::size_t n = 0;

            for (int i = 0; i < num_wp && n < MaxWp; ++i) {
                if (n > 0) {
                    if (std::hypot(wp_x_ptr[i] - wp_x_arr[n - 1], wp_y_ptr[i] - wp_y_arr[n - 1]) < 1e-3) continue;
                }
                wp_x_arr[n] = wp_x_ptr[i];
                wp_y_arr[n] = wp_y_ptr[i];
                n++;
            }

            if (n < 5) {
                spline_valid = false;
                return py::make_tuple("Insufficient Waypoints", py::make_tuple(0.0, -1.0));
            }

            spline.build(wp_x_arr, wp_y_arr, n);
            spline_valid = true;

            double vx = ego_state_ptr[3];
            double vy = ego_state_ptr[4];
            double r_rate = ego_state_ptr[5];

            auto frenet = project_vehicle_state(ego_state_ptr[0], ego_state_ptr[1], ego_state_ptr[2]);

            matrix::StaticVector<double, Nx_mem> x_curr;
            x_curr.set_zero();
            x_curr(0) = frenet.s;
            x_curr(1) = frenet.d;
            x_curr(2) = frenet.mu;
            x_curr(3) = std::max(0.1, vx);
            x_curr(4) = vy;
            x_curr(5) = r_rate;

            double delta = nmpc.u_last(0);
            x_curr(6) = std::atan2(vy + lf * r_rate, std::max(0.5, vx)) - delta;
            x_curr(7) = std::atan2(vy - lr * r_rate, std::max(0.5, vx));

            double dt = nmpc.dt;

            // [Phase 1 아키텍트의 수술: 맹인의 시력 회복 (Predictive Curvature Horizon)]
            // 100 스텝 앞까지의 모든 곡률을 배열에 저장하여 NMPC에 미래 시야를 제공합니다.
            // [Phase 1 아키텍트의 수술: 맹인의 시력 회복 (Predictive Curvature Horizon)]
            for (std::size_t k = 0; k < H; ++k) {
                // [수술 부위: 시야 붕괴 방지]
                // 차량이 정지해 있더라도, 최소 5.0m/s(약 18km/h)로 달릴 때의 거리를 내다보도록 강제합니다.
                // 제자리에서 핸들을 비비는 환각을 원천 차단합니다.
                double v_horizon = std::max(5.0, config.target_vx);
                double s_pred = frenet.s + v_horizon * k * dt;
                
                s_pred = std::clamp(s_pred, 0.0, spline.get_max_s());

                double raw_kappa = spline.calc_curvature(s_pred);
                config.kappa_ref[k] = std::clamp(raw_kappa, -0.2, 0.2); 
                config.target_d[k] = 0.0; 
            }

            for (std::size_t i = 0; i < 10; ++i) {
                nmpc.env_evaluator.obstacles[i].s = obstacle_ptr[i * 5 + 0];
                nmpc.env_evaluator.obstacles[i].d = obstacle_ptr[i * 5 + 1];
                nmpc.env_evaluator.obstacles[i].r = obstacle_ptr[i * 5 + 2];
                nmpc.env_evaluator.obstacles[i].vs = obstacle_ptr[i * 5 + 3];
                nmpc.env_evaluator.obstacles[i].vd = obstacle_ptr[i * 5 + 4];
            }

            auto res = nmpc.solve_ipm(x_curr, config);

            matrix::StaticVector<double, Nu> u_opt;
            if (res.success || res.fallback_triggered) {
                u_opt = nmpc.u_last;
            } else {
                u_opt.set_zero();
            }

            nmpc.shift_sequence();
            return py::make_tuple(res.status_msg, py::make_tuple(u_opt(0), u_opt(1)));
        }
};

PYBIND11_MODULE(nmpc_core, m) {
    py::class_<SparseNMPCWrapper>(m, "SparseNMPCWrapper")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("set_target_speed", &SparseNMPCWrapper::set_target_speed)
        .def("update_config", &SparseNMPCWrapper::update_config)
        .def("solve", &SparseNMPCWrapper::solve);
}