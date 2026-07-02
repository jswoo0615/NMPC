#ifndef OPTIMIZATION_DYNAMICS_VEHICLE_PHYSICS_CORE_HPP_
#define OPTIMIZATION_DYNAMICS_VEHICLE_PHYSICS_CORE_HPP_

#include <algorithm>
#include <cmath>
#include "Optimization/Matrix/Core/MathTraits.hpp"

#include "Optimization/Matrix/AD/Dual.hpp"

#include "Optimization/Utils/CUDAMacros.hpp"

#include "Optimization/Dynamics/FastMath.hpp"
#include "Optimization/Dynamics/VehicleParams.hpp"

namespace Optimization {

namespace Dynamics {

// =========================================================================
// 2. Load Transfer Physics
// =========================================================================

template <typename T>
CUDA_CALLABLE inline T safe_max(T val, double min_val) {
    double v = Optimization::MathTraits<T>::get_value(val);
    return (v >= min_val) ? val : static_cast<T>(min_val);
}

template <typename T>
CUDA_CALLABLE inline T safe_min(T val, double max_val) {
    double v = Optimization::MathTraits<T>::get_value(val);
    return (v <= max_val) ? val : static_cast<T>(max_val);
}

/**
 * @brief Computes Quasi-Static Load Transfer (Used for Real-Time Model)
 * @details Calculates the vertical load (fz) on all 4 wheels based on longitudinal 
 *          acceleration and a steady-state approximation of lateral acceleration.
 */
template <typename T>
CUDA_CALLABLE inline FourWheelLoads<T> computeQuasiStaticLoadTransfer(
    const VehicleDynamicsParams<T>& params,
    const T v_x,
    const T yaw_rate,
    const T a_cmd
) {
    const T L = params.l_f + params.l_r;

    const T fz_f_stat = params.m * params.g * (params.l_r / L);
    const T fz_r_stat = params.m * params.g * (params.l_f / L);

    const T delta_fz_long = params.m * a_cmd * (params.h_cg / L);

    const T a_y_approx = v_x * yaw_rate;
    const T delta_fz_lat_f = params.m * a_y_approx * (params.h_cg / params.t_f) * params.K_df;
    const T delta_fz_lat_r = params.m * a_y_approx * (params.h_cg / params.t_r) * (static_cast<T>(1.0) - params.K_df);

    FourWheelLoads<T> loads;
    loads.fz_fl = safe_max((fz_f_stat * static_cast<T>(0.5)) - (delta_fz_long * static_cast<T>(0.5)) - delta_fz_lat_f, Optimization::MathTraits<T>::get_value(params.fz_min));
    loads.fz_fr = safe_max((fz_f_stat * static_cast<T>(0.5)) - (delta_fz_long * static_cast<T>(0.5)) + delta_fz_lat_f, Optimization::MathTraits<T>::get_value(params.fz_min));
    loads.fz_rl = safe_max((fz_r_stat * static_cast<T>(0.5)) + (delta_fz_long * static_cast<T>(0.5)) - delta_fz_lat_r, Optimization::MathTraits<T>::get_value(params.fz_min));
    loads.fz_rr = safe_max((fz_r_stat * static_cast<T>(0.5)) + (delta_fz_long * static_cast<T>(0.5)) + delta_fz_lat_r, Optimization::MathTraits<T>::get_value(params.fz_min));

    return loads;
}

/**
 * @brief Computes Suspension Geometric Load Transfer (Used for High-Fidelity Model)
 * @details Calculates the vertical load (fz) considering suspension roll/pitch stiffness, 
 *          roll center heights, and updates the chassis attitude.
 */
template <typename T>
CUDA_CALLABLE inline FourWheelLoads<T> computeSuspensionLoadTransfer(
    const VehicleDynamicsParams<T>& veh_params,
    const SuspensionParams<T>& susp_params,
    const T v_x,
    const T yaw_rate,
    const T a_cmd,
    ChassisAttitude<T>& out_attitude
) {
    const T L = veh_params.l_f + veh_params.l_r;

    const T h_rc_avg = (susp_params.h_rc_f * veh_params.l_r + susp_params.h_rc_r * veh_params.l_f) / L;
    const T h_roll_arm = veh_params.h_cg - h_rc_avg;
    const T a_y_approx = v_x * yaw_rate;

    out_attitude.pitch_theta = (veh_params.m * a_cmd * veh_params.h_cg) / susp_params.K_pitch;

    T roll_den = susp_params.K_roll - (veh_params.m * veh_params.g * h_roll_arm);
    roll_den = safe_max(roll_den, 100.0);
    out_attitude.roll_phi = (veh_params.m * a_y_approx * h_roll_arm) / roll_den;

    const T delta_fz_long = (veh_params.m * a_cmd * veh_params.h_cg + veh_params.m * veh_params.g * veh_params.h_cg * FastMath::fast_sin(out_attitude.pitch_theta)) / L;

    const T elastic_roll_moment = veh_params.m * a_y_approx * h_roll_arm + veh_params.m * veh_params.g * h_roll_arm * FastMath::fast_sin(out_attitude.roll_phi);
    const T delta_fz_lat_f = (susp_params.K_roll_f / susp_params.K_roll) * (elastic_roll_moment / veh_params.t_f) + (veh_params.m * a_y_approx * susp_params.h_rc_f / veh_params.t_f);
    const T delta_fz_lat_r = ((susp_params.K_roll - susp_params.K_roll_f) / susp_params.K_roll) * (elastic_roll_moment / veh_params.t_r) + (veh_params.m * a_y_approx * susp_params.h_rc_r / veh_params.t_r);
    
    const T fz_f_stat = veh_params.m * veh_params.g * (veh_params.l_r / L);
    const T fz_r_stat = veh_params.m * veh_params.g * (veh_params.l_f / L);

    FourWheelLoads<T> loads;
    loads.fz_fl = safe_max((fz_f_stat * static_cast<T>(0.5)) - (delta_fz_long * static_cast<T>(0.5)) - delta_fz_lat_f, Optimization::MathTraits<T>::get_value(veh_params.fz_min));
    loads.fz_fr = safe_max((fz_f_stat * static_cast<T>(0.5)) - (delta_fz_long * static_cast<T>(0.5)) + delta_fz_lat_f, Optimization::MathTraits<T>::get_value(veh_params.fz_min));
    loads.fz_rl = safe_max((fz_r_stat * static_cast<T>(0.5)) + (delta_fz_long * static_cast<T>(0.5)) - delta_fz_lat_r, Optimization::MathTraits<T>::get_value(veh_params.fz_min));
    loads.fz_rr = safe_max((fz_r_stat * static_cast<T>(0.5)) + (delta_fz_long * static_cast<T>(0.5)) + delta_fz_lat_r, Optimization::MathTraits<T>::get_value(veh_params.fz_min));
    
    return loads;
}

// =========================================================================
// 3. Tire Magic Formula Physics
// =========================================================================

/**
 * @brief Computes the non-linear lateral force for a single tire using the Magic Formula
 */
template <typename T>
CUDA_CALLABLE inline T computeMagicFormulaTireForce(
    const TireLoadParams<T>& tire_params,
    const T fz,
    const T alpha
) {
    T mu = tire_params.mu_0 - tire_params.d_mu * (fz - tire_params.fz_nom);
    mu = safe_max(mu, 0.1);

    T c_alpha = tire_params.c1 * fz - tire_params.c2 * fz * fz;
    c_alpha = safe_max(c_alpha, 1.0);

    T D = mu * fz;
    T B = c_alpha / (tire_params.C_shape * D);

    return D * FastMath::fast_sin(tire_params.C_shape * FastMath::fast_atan(B * alpha));
}

/**
 * @brief Integrates lateral forces across all 4 wheels into a bicycle model equivalent
 */
template <typename T>
CUDA_CALLABLE inline BicycleLateralForces<T> computeBicycleLateralForces(
    const TireLoadParams<T>& tire_params,
    const FourWheelLoads<T>& loads,
    const T alpha_f,
    const T alpha_r
) {
    T fy_fl = computeMagicFormulaTireForce(tire_params, loads.fz_fl, alpha_f);
    T fy_fr = computeMagicFormulaTireForce(tire_params, loads.fz_fr, alpha_f);
    T fy_rl = computeMagicFormulaTireForce(tire_params, loads.fz_rl, alpha_r);
    T fy_rr = computeMagicFormulaTireForce(tire_params, loads.fz_rr, alpha_r);

    BicycleLateralForces<T> forces;
    forces.fy_f = fy_fl + fy_fr;
    forces.fy_r = fy_rl + fy_rr;

    return forces;
}

/**
 * @brief Computes lateral forces including Camber Thrust (Used for High-Fidelity Model)
 */
template <typename T>
CUDA_CALLABLE inline BicycleLateralForces<T> computeLateralForcesWithCamber(
    const TireLoadParams<T>& tire_params,
    const SuspensionParams<T>& susp_params,
    const FourWheelLoads<T>& loads,
    const ChassisAttitude<T>& attitude,
    const T alpha_f,
    const T alpha_r
) {
    BicycleLateralForces<T> base_forces = computeBicycleLateralForces(tire_params, loads, alpha_f, alpha_r);

    T gamma_f_dyn = susp_params.gamma_0_f + susp_params.K_camber_gain * attitude.roll_phi;
    T gamma_r_dyn = susp_params.gamma_0_r + susp_params.K_camber_gain * attitude.roll_phi;

    T camber_thrust_f = susp_params.C_gamma * gamma_f_dyn;
    T camber_thrust_r = susp_params.C_gamma * gamma_r_dyn;

    BicycleLateralForces<T> final_forces;
    final_forces.fy_f = base_forces.fy_f + camber_thrust_f;
    final_forces.fy_r = base_forces.fy_r + camber_thrust_r;

    return final_forces;
}

} // namespace Dynamics
} // namespace Optimization

#endif // OPTIMIZATION_DYNAMICS_VEHICLE_PHYSICS_CORE_HPP_
