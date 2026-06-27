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

            config.target_vx = 10.0; // 90km/h에 맞춰 타겟을 상향 조정합니다.
            
            // [아키텍트의 수술: 2. 비용 함수 밸런스 재조정]
            config.Q_D = 200.0;       // 궤적을 좀 더 타이트하게 추종하되
            config.Q_mu = 50.0;       // 헤딩 안정성 유지
            config.Q_Vx = 5.0;        // 속도 오차에 대한 강박을 줄임 (속도는 물리 엔진에 맡김)
            config.Q_Vy = 5.0;        
            config.Q_r = 5.0;         

            // [아키텍트의 수술: 3. 슬립각 페널티 하향]
            // 2000.0은 너무 가혹합니다. 200.0으로 낮추어 타이어의 자연스러운 슬립을 허용하십시오.
            config.Q_alpha_f = 2000.0;
            config.Q_alpha_r = 2000.0;

            config.R_Steer = 150.0;    // 조향을 좀 더 부드럽게 가져갑니다.
            config.R_Accel = 1.0;     // 가속 명령을 해방합니다.
            
            config.Obstacle_Margin = 0.0;
            
            // 조향/가속 제한은 물리적 한계로 넉넉하게 풉니다.
            config.u_min[0] = -0.6; config.u_max[0] = 0.6;
            config.u_min[1] = -10.0; config.u_max[1] = 10.0;
        }

        void set_target_speed(double speed) {
            config.target_vx = speed;
        }

        void update_config(py::dict opt_dict) {
            if (opt_dict.contains("Q_D")) {
                config.Q_D = opt_dict["Q_D"].cast<double>();
            }
            if (opt_dict.contains("Q_mu")) {
                config.Q_mu = opt_dict["Q_mu"].cast<double>();
            }
            if (opt_dict.contains("Q_Vx")) {
                config.Q_Vx = opt_dict["Q_Vx"].cast<double>();
            }
            if (opt_dict.contains("Q_Vy")) {
                config.Q_Vy = opt_dict["Q_Vy"].cast<double>();
            }
            if (opt_dict.contains("R_Steer")) {
                config.R_Steer = opt_dict["R_Steer"].cast<double>();
            }
            if (opt_dict.contains("R_Accel")) {
                config.R_Accel = opt_dict["R_Accel"].cast<double>();
            }
        }

        struct FrenetState {
            double s, d, mu;
        };

        FrenetState project_vehicle_state(double x, double y, double yaw) {
            if (!spline_valid) {
                return {0.0, 0.0, 0.0};
            }

            double s_center = last_s_projection;
            double best_s = s_center;
            double best_dist2 = std::numeric_limits<double>::max();

            const double search_window = 15.0;
            const double coarse_step = 0.5;

            // 1. Coarse Search
            for (double ds = -search_window; ds <= search_window; ds += coarse_step) {
                double s = std::clamp(s_center + ds, 0.0, spline.get_max_s());
                double px = spline.calc_x(s);
                double py = spline.calc_y(s);

                double dx = px - x;
                double dy = py - y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_s = s;
                }
            }

            // 2. Newton Refinement
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

                double f = ex * dx_ds + ey * dy_ds; // 수정됨 (dy_Ds -> dy_ds)
                double df = dx_ds * dx_ds + dy_ds * dy_ds + ex * ddx_ds + ey * ddy_ds; // 수정됨 (ddy_df -> ddy_ds)

                if (std::abs(df) < 1e-8) {
                    break;
                }

                s_opt -= f / df;
                s_opt = std::clamp(s_opt, 0.0, spline.get_max_s());
            }

            // 3. Frenet State
            double ref_x = spline.calc_x(s_opt);
            double ref_y = spline.calc_y(s_opt);
            double ref_yaw = spline.calc_yaw(s_opt);

            double dx = x - ref_x;
            double dy = y - ref_y;

            double d = -dx * std::sin(ref_yaw) + dy * std::cos(ref_yaw);
            double mu = yaw - ref_yaw;

            while (mu > M_PI) {
                mu -= 2.0 * M_PI;
            }
            while (mu < -M_PI) {
                mu += 2.0 * M_PI;
            }

            last_s_projection = s_opt;

            return {s_opt, d, mu};
        }

        py::tuple solve(int num_wp) {
            std::array<double, MaxWp> wp_x_arr, wp_y_arr;
            std::size_t n = 0;

            for (int i = 0; i < num_wp && n < MaxWp; ++i) {
                if (n > 0) {
                    double dx = wp_x_ptr[i] - wp_x_arr[n - 1];
                    double dy = wp_y_ptr[i] - wp_y_arr[n - 1];
                    if (std::hypot(dx, dy) < 1e-3) {
                        continue;
                    }
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

            double x_veh = ego_state_ptr[0];
            double y_veh = ego_state_ptr[1];
            double yaw_veh = ego_state_ptr[2];
            double vx = ego_state_ptr[3];
            double vy = ego_state_ptr[4];
            double r_rate = ego_state_ptr[5];

            auto frenet = project_vehicle_state(x_veh, y_veh, yaw_veh);

            matrix::StaticVector<double, Nx_mem> x_curr;
            x_curr.set_zero();
            x_curr(0) = frenet.s;
            x_curr(1) = frenet.d;
            x_curr(2) = frenet.mu;
            x_curr(3) = std::max(0.1, vx);
            x_curr(4) = vy;
            x_curr(5) = r_rate;

            // 조향 입력 지연 보상 및 타이어 동역학 반영
            double delta = nmpc.u_last(0);
            double alpha_f = std::atan2(vy + lf * r_rate, std::max(0.5, vx)) - delta;
            double alpha_r = std::atan2(vy - lr * r_rate, std::max(0.5, vx));

            x_curr(6) = alpha_f;
            x_curr(7) = alpha_r;

            double dt = nmpc.dt;

            for (std::size_t k = 0; k < H; ++k) {
                double s_pred = frenet.s + config.target_vx * k * dt;
                
                if (s_pred > spline.get_max_s()) {
                    s_pred = spline.get_max_s();
                }

                if (k == 0) {
                    double raw_kappa = spline.calc_curvature(s_pred);
                    if (raw_kappa > 0.2) {
                        raw_kappa = 0.2;
                    }
                    if (raw_kappa < -0.2) {
                        raw_kappa = -0.2;
                    }
                    config.kappa = raw_kappa;
                }
                config.target_d[k] = 0.0; // 오타 수정됨 (config.target_d[k] - 0.0; -> = 0.0;)
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
    py::class_<SparseNMPCWrapper>(m, "SparseNMPCWrapper") // 오타 수정됨 (SparseNMPCWrappter)
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("set_target_speed", &SparseNMPCWrapper::set_target_speed)
        .def("update_config", &SparseNMPCWrapper::update_config)
        .def("solve", &SparseNMPCWrapper::solve);
}