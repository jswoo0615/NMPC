#ifndef OPTIMIZATION_DYNAMICS_REAL_TIME_MODEL_HPP_
#define OPTIMIZATION_DYNAMICS_REAL_TIME_MODEL_HPP_

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Dynamics/VehiclePhysicsCore.hpp"

namespace Optimization {
namespace Dynamics {

/**
 * @brief 실시간 NMPC 솔버 연동을 위한 8차원 Frenet 자전거 모델 (기능 1: 실시간용)
 * @details 연산 속도 극대화를 위해 서스펜션 기하학(롤/피치)을 제외하고, 
 *          종방향 하중 이동 및 기본 Magic Formula 연산만을 포함합니다.
 */
class RealTimeDynamicsModel {
public:
    VehicleDynamicsParams<double> veh_params_base;
    TireLoadParams<double> tire_params_base;
    double kappa = 0.0;

    CUDA_CALLABLE RealTimeDynamicsModel() {
        // 기본 파라미터 초기화
        veh_params_base = {1500.0, 9.81, 1.2, 1.6, 0.5, 1.6, 1.6, 0.6, 100.0, 3000.0};
        tire_params_base = {1.0, 0.0001, 4000.0, 4000.0, 15.0, 1.4, 0.1, 0.15, 0.15};
    }

    template <typename T>
    CUDA_CALLABLE matrix::StaticVector<T, 8> operator()(const matrix::StaticVector<T, 8>& x, const matrix::StaticVector<T, 2>& u) const {
        // 상태 언패킹
        T s = x(0);
        T d = x(1);
        T mu = x(2);
        T vx = x(3);
        T vy = x(4);
        T r = x(5);
        T alpha_f_curr = x(6);
        T alpha_r_curr = x(7);

        // 제어 입력 언패킹
        T delta = u(0);
        T a = u(1);

        // 파라미터 캐스팅
        VehicleDynamicsParams<T> v_p;
        v_p.m = T(veh_params_base.m);
        v_p.g = T(veh_params_base.g);
        v_p.l_f = T(veh_params_base.l_f);
        v_p.l_r = T(veh_params_base.l_r);
        v_p.h_cg = T(veh_params_base.h_cg);
        v_p.t_f = T(veh_params_base.t_f);
        v_p.t_r = T(veh_params_base.t_r);
        v_p.K_df = T(veh_params_base.K_df);
        v_p.fz_min = T(veh_params_base.fz_min);
        v_p.I_z = T(veh_params_base.I_z);

        TireLoadParams<T> t_p;
        t_p.mu_0 = T(tire_params_base.mu_0);
        t_p.d_mu = T(tire_params_base.d_mu);
        t_p.fz_nom = T(tire_params_base.fz_nom);
        t_p.c1 = T(tire_params_base.c1);
        t_p.c2 = T(tire_params_base.c2);
        t_p.C_shape = T(tire_params_base.C_shape);
        t_p.eps_vx = T(tire_params_base.eps_vx);
        t_p.sigma_f = T(tire_params_base.sigma_f);
        t_p.sigma_r = T(tire_params_base.sigma_r);

        // 수치 안정성을 위해 vx 클리핑
        double vx_val = Optimization::MathTraits<T>::get_value(vx);
        T vx_safe = (vx_val >= 0.1) ? vx : ((vx_val <= -0.1) ? vx : T(0.1));

        // 1. 하중 이동 (Quasi-Static)
        FourWheelLoads<T> loads = computeQuasiStaticLoadTransfer(v_p, vx_safe, r, a);

        // 2. 정상 상태 슬립각 (Steady-State Slip Angle)
        T alpha_ss_f = delta - FastMath::math_atan2(vy + v_p.l_f * r, vx_safe);
        T alpha_ss_r = -FastMath::math_atan2(vy - v_p.l_r * r, vx_safe);

        // 3. 지연길이 (Relaxation Length) ODE
        T vx_abs = (vx_val >= 0.0) ? vx : -vx;
        T vx_relax = safe_max(vx_abs, 0.5); 

        T rate_f = vx_relax / t_p.sigma_f;
        T rate_r = vx_relax / t_p.sigma_r;
        rate_f = safe_min(rate_f, 25.0);
        rate_r = safe_min(rate_r, 25.0);

        T d_alpha_f = rate_f * (alpha_ss_f - alpha_f_curr);
        T d_alpha_r = rate_r * (alpha_ss_r - alpha_r_curr);

        // 4. Magic Formula 비선형 횡력 계산
        BicycleLateralForces<T> forces = computeBicycleLateralForces(t_p, loads, alpha_f_curr, alpha_r_curr);

        // 5. Frenet Kinematics 및 가속도
        T denom = T(1.0) - d * T(kappa);
        double denom_val = Optimization::MathTraits<T>::get_value(denom);
        T denom_safe = (denom_val >= 0.05) ? denom : T(0.05);

        T s_dot = (vx * FastMath::math_cos(mu) - vy * FastMath::math_sin(mu)) / denom_safe;

        matrix::StaticVector<T, 8> x_dot;
        x_dot(0) = s_dot;
        x_dot(1) = vx * FastMath::math_sin(mu) + vy * FastMath::math_cos(mu);
        x_dot(2) = r - T(kappa) * s_dot;
        x_dot(3) = a + r * vy;
        x_dot(4) = (forces.fy_f * FastMath::math_cos(delta) + forces.fy_r) / v_p.m - r * vx;
        x_dot(5) = (v_p.l_f * forces.fy_f * FastMath::math_cos(delta) - v_p.l_r * forces.fy_r) / v_p.I_z;
        x_dot(6) = d_alpha_f;
        x_dot(7) = d_alpha_r;

        return x_dot;
    }
};

} // namespace Dynamics
} // namespace Optimization

#endif // OPTIMIZATION_DYNAMICS_REAL_TIME_MODEL_HPP_
