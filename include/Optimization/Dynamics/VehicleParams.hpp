#ifndef OPTIMIZATION_DYNAMICS_VEHICLE_PARAMS_HPP_
#define OPTIMIZATION_DYNAMICS_VEHICLE_PARAMS_HPP_

namespace Optimization {
    namespace Dynamics {
        // 1. Parameters & Data Structures
        template <typename T>
        struct VehicleDynamicsParams {
            T m;
            T g;
            T l_f;
            T l_r;
            T h_cg;
            T t_f;
            T t_r;
            T K_df;
            T fz_min;
            T I_z;
        };

        template <typename T>
        struct TireLoadParams {
            T mu_0;
            T d_mu;
            T fz_nom;
            T c1;
            T c2;
            T C_shape;
            T eps_vx;
            T sigma_f;
            T sigma_r;
        };

        template <typename T>
        struct SuspensionParams {
            T K_pitch;
            T K_roll;
            T K_roll_f;
            T h_rc_f;
            T h_rc_r;
            T K_camber_gain;
            T C_gamma;
            T gamma_0_f;
            T gamma_0_r;
        };

        template <typename T>
        struct FourWheelLoads {
            T fz_fl;
            T fz_fr;
            T fz_rl;
            T fz_rr;
        };

        template <typename T>
        struct BicycleLateralForces {
            T fy_f;
            T fy_r;
        };

        template <typename T>
        struct ChassisAttitude {
            T pitch_theta;
            T roll_phi;
        };

        template <typename T>
        struct State8 {
            T s;
            T d;
            T mu;
            T vx;
            T vy;
            T r;
            T alpha_f;
            T alpha_r;
        };

        template <typename T>
        struct Control2 {
            T delta;
            T a_cmd;
        };
    } // namespace Dynamics
} // namespace Optimization

#endif // OPTIMIZATION_DYNAMICS_VEHICLE_PARAMS_HPP_