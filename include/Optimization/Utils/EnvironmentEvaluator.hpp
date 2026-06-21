#ifndef OPTIMIZATION_EVALUATOR_ENVIRONMENT_EVALUATOR_HPP_
#define OPTIMIZATION_EVALUATOR_ENVIRONMENT_EVALUATOR_HPP_

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include <array>
#include <cmath>

namespace Optimization {
namespace Evaluator {

    struct ObstacleFrenet {
        double s = 0.0;
        double d = 0.0;
        double r = 0.5;
        double vs = 0.0;
        double vd = 0.0;
    };

    // 솔버에 주입될 추상화된 제약 조건 평가 데이터 (c, Jacobian)
    template <std::size_t Nx_active>
    struct ConstraintGradient {
        double c_val = 0.0;
        matrix::StaticVector<double, Nx_active> J_x;
        bool is_active = false;
    };

    template <std::size_t Nx_active, std::size_t MaxObs = 10>
    class EnvironmentEvaluator {
    public:
        std::array<ObstacleFrenet, MaxObs> obstacles;
        double obstacle_margin = 1.5;

        EnvironmentEvaluator() {
            for (auto& obs : obstacles) {
                obs.s = 10000.0; obs.d = 10000.0; obs.r = 0.1;
                obs.vs = 0.0; obs.vd = 0.0;
            }
        }

        // 특정 시점(time_future)의 상태(s, d)를 받아 제약조건 평가
        std::array<ConstraintGradient<Nx_active>, MaxObs> evaluate_obstacles(
            double current_s, double current_d, double time_future) const 
        {
            std::array<ConstraintGradient<Nx_active>, MaxObs> grads;
            
            for (std::size_t i = 0; i < MaxObs; ++i) {
                grads[i].J_x.set_zero();
                
                double obs_pred_s = obstacles[i].s + obstacles[i].vs * time_future;
                double ds = current_s - obs_pred_s;

                // 관심 영역(ROI)을 벗어난 장애물은 연산에서 제외
                if (std::abs(ds) > 20.0) {
                    grads[i].is_active = false;
                    continue;
                }

                double obs_pred_d = obstacles[i].d + obstacles[i].vd * time_future;
                double dd = current_d - obs_pred_d;
                double dd_eff = dd;
                if (std::abs(dd_eff) < 0.1) dd_eff = (dd_eff >= 0 ? 0.1 : -0.1);
                
                double ds_scaled = ds * 0.5;
                double dist_sq = ds_scaled * ds_scaled + dd_eff * dd_eff;
                double safety_margin = obstacles[i].r + obstacle_margin;
                
                // c(x) 도출
                grads[i].c_val = safety_margin * safety_margin - dist_sq;
                
                // Jacobian [s, d, mu, vx, vy, r, ...] 
                grads[i].J_x(0) = -2.0 * ds_scaled * 0.5; // d(c)/ds
                grads[i].J_x(1) = -2.0 * dd_eff;          // d(c)/dd
                grads[i].is_active = true;
            }
            return grads;
        }
    };

} // namespace Evaluator
} // namespace Optimization

#endif // OPTIMIZATION_EVALUATOR_ENVIRONMENT_EVALUATOR_HPP_