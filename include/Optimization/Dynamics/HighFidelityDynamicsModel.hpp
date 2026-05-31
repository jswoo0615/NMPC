#ifndef OPTIMIZATION_DYNAMICS_HIGH_FIDELITY_MODEL_HPP_
#define OPTIMIZATION_DYNAMICS_HIGH_FIDELITY_MODEL_HPP_

#include <array>
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Dynamics/VehiclePhysicsCore.hpp"

// Optimization::Dual을 전역 네임스페이스에 노출하여 기존 코드의 Dual 명칭 그대로 사용 지원
using Optimization::Dual;

namespace Optimization {
namespace Dynamics {

/**
 * @brief 정밀 시뮬레이션 및 검증용 8차원 Frenet 자전거 모델 (기능 2: 세부 정밀용)
 * @details 4륜 독립 하중 이동, 서스펜션 롤/피치 지오메트리, 동적 캠버 변화에 따른 추력(Camber Thrust) 등을 
 *          고려한 높은 정확도의 동역학을 제공하며, 연속 상태공간 Jacobian 추출 기능을 포함합니다.
 */
class HighFidelityDynamicsModel {
public:
    VehicleDynamicsParams<double> veh_params_base;
    SuspensionParams<double> susp_params_base;
    TireLoadParams<double> tire_params_base;
    double kappa = 0.0;

    CUDA_CALLABLE HighFidelityDynamicsModel() {
        // 기본 파라미터 초기화
        veh_params_base = {1500.0, 9.81, 1.2, 1.6, 0.5, 1.6, 1.6, 0.6, 100.0, 3000.0};
        susp_params_base = {50000.0, 80000.0, 48000.0, 0.3, 0.35, 0.05, 20000.0, -0.01, -0.01};
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
        T a_cmd = u(1);

        // 파라미터 캐스팅 (double -> T)
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

        SuspensionParams<T> s_p;
        s_p.K_pitch = T(susp_params_base.K_pitch);
        s_p.K_roll = T(susp_params_base.K_roll);
        s_p.K_roll_f = T(susp_params_base.K_roll_f);
        s_p.h_rc_f = T(susp_params_base.h_rc_f);
        s_p.h_rc_r = T(susp_params_base.h_rc_r);
        s_p.K_camber_gain = T(susp_params_base.K_camber_gain);
        s_p.C_gamma = T(susp_params_base.C_gamma);
        s_p.gamma_0_f = T(susp_params_base.gamma_0_f);
        s_p.gamma_0_r = T(susp_params_base.gamma_0_r);

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

        // 1. 정상 상태 슬립각 및 지연길이 (Relaxation Length) ODE
        T alpha_ss_f = delta - FastMath::math_atan2(vy + v_p.l_f * r, vx_safe);
        T alpha_ss_r = -FastMath::math_atan2(vy - v_p.l_r * r, vx_safe);

        T vx_abs = (vx_val >= 0.0) ? vx : -vx;
        T vx_relax = safe_max(vx_abs, 0.5);

        T rate_f = vx_relax / t_p.sigma_f;
        T rate_r = vx_relax / t_p.sigma_r;
        rate_f = safe_min(rate_f, 25.0);
        rate_r = safe_min(rate_r, 25.0);

        T d_alpha_f = rate_f * (alpha_ss_f - alpha_f_curr);
        T d_alpha_r = rate_r * (alpha_ss_r - alpha_r_curr);

        // 2. High-Fidelity 하중 이동 및 캠버 계산 (Suspension Geometrics)
        ChassisAttitude<T> attitude;
        FourWheelLoads<T> loads = computeSuspensionLoadTransfer(v_p, s_p, vx_safe, r, a_cmd, attitude);

        // 3. 캠버 추력을 포함한 비선형 타이어 횡력 계산
        BicycleLateralForces<T> forces = computeLateralForcesWithCamber(t_p, s_p, loads, attitude, alpha_f_curr, alpha_r_curr);

        // 환경 저항 (Aerodynamic Drag)
        T C_drag = static_cast<T>(0.3);
        T F_drag = C_drag * vx * Optimization::MathTraits<T>::abs(vx);
        T effective_a = a_cmd - (F_drag / v_p.m);

        // 4. Frenet Kinematics
        T denom = T(1.0) - d * T(kappa);
        double denom_val = Optimization::MathTraits<T>::get_value(denom);
        T denom_safe = (denom_val >= 0.05) ? denom : T(0.05);

        T s_dot = (vx * FastMath::math_cos(mu) - vy * FastMath::math_sin(mu)) / denom_safe;

        matrix::StaticVector<T, 8> x_dot;
        x_dot(0) = s_dot;
        x_dot(1) = vx * FastMath::math_sin(mu) + vy * FastMath::math_cos(mu);
        x_dot(2) = r - T(kappa) * s_dot;
        x_dot(3) = effective_a + r * vy;
        x_dot(4) = (forces.fy_f * FastMath::math_cos(delta) + forces.fy_r) / v_p.m - r * vx;
        x_dot(5) = (v_p.l_f * forces.fy_f * FastMath::math_cos(delta) - v_p.l_r * forces.fy_r) / v_p.I_z;
        x_dot(6) = d_alpha_f;
        x_dot(7) = d_alpha_r;

        return x_dot;
    }

    // Matrix Engine: 8-State Jacobian (A, B) 추출
    CUDA_CALLABLE inline void extractJacobians(const matrix::StaticVector<double, 8>& x0, 
                                               const matrix::StaticVector<double, 2>& u0,
                                               std::array<std::array<double, 8>, 8>& A,
                                               std::array<std::array<double, 2>, 8>& B) const {
        // A 행렬 추출 (상태 변수에 대해 미분)
        for (int i = 0; i < 8; ++i) {
            matrix::StaticVector<Dual<double>, 8> x_dual;
            for (int k = 0; k < 8; ++k) x_dual(k) = Dual<double>(x0(k), (i == k) ? 1.0 : 0.0);

            matrix::StaticVector<Dual<double>, 2> u_dual;
            u_dual(0) = Dual<double>(u0(0), 0.0);
            u_dual(1) = Dual<double>(u0(1), 0.0);

            matrix::StaticVector<Dual<double>, 8> x_dot_dual = (*this)(x_dual, u_dual);
            
            for (int k = 0; k < 8; ++k) A[k][i] = x_dot_dual(k).d;
        }

        // B 행렬 추출 (제어 입력에 대해 미분)
        for (int j = 0; j < 2; ++j) {
            matrix::StaticVector<Dual<double>, 8> x_dual;
            for (int k = 0; k < 8; ++k) x_dual(k) = Dual<double>(x0(k), 0.0);

            matrix::StaticVector<Dual<double>, 2> u_dual;
            u_dual(0) = Dual<double>(u0(0), (j == 0) ? 1.0 : 0.0);
            u_dual(1) = Dual<double>(u0(1), (j == 1) ? 1.0 : 0.0);

            matrix::StaticVector<Dual<double>, 8> x_dot_dual = (*this)(x_dual, u_dual);
            
            for (int k = 0; k < 8; ++k) B[k][j] = x_dot_dual(k).d;
        }
    }
};

} // namespace Dynamics
} // namespace Optimization

#endif // OPTIMIZATION_DYNAMICS_HIGH_FIDELITY_MODEL_HPP_
