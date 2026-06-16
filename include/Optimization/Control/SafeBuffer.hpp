/**
 * @file SafeTrajectoryBuffer.hpp
 * @brief Stores the last valid trajectory for fail-safe control operation.
 * 
 * This class is designed for MPC/NMPC applications where the optimizer 
 * may occasionally fail to converge within the control cycle.
 * 
 * The buffer preserves the most recently feasible state and control trajectory 
 * so that the controller can : 
 * 
 * 1. Reuse the last valid trajectory as a warm-start directly via public members.
 * 2. Maintain safe operation during temporary optimization failures.
 * 3. Generate a deterministic emergency control command.
 */

#ifndef OPTIMIZATION_CONTROL_SAFE_TRAJECTORY_BUFFER_HPP_
#define OPTIMIZATION_CONTROL_SAFE_TRAJECTORY_BUFFER_HPP_
#include <array>
#include <cassert>
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

namespace Optimization {
    namespace controller {
        /**
         * @brief Safe trajectory storage for MPC/NMPC fail-safe operation.
         * 
         * @tparam H Prediction horizon
         * @tparam Nx Number of states.
         * @tparam Nu Number of control inputs.
         * 
         * This buffer stores the most recently feasible optimization result.
         * If the optimizer fails during a control cycle, the stored trajectory 
         * can be reused and a predefined emergency control action can be generated.
         */
        template <std::size_t H, std::size_t Nx, std::size_t Nu>
        class SafeTrajectoryBuffer {
            public:
                /**
                 * @brief Stored feasible state trajectory.
                 * Size = H + 1
                 * 
                 * X_safe[k] corresponds to the predicted state at stage k.
                 * Accessible directly for warm-start initialization
                 */
                std::array<matrix::StaticVector<double, Nx>, H + 1> X_safe;

                /**
                 * @brief Stored feasible control trajectory.
                 * Size = H
                 * 
                 * U_safe[k] corresponds to the control input applied
                 * between stage k and k+1
                 * 
                 * U_safe[k] corresponds to the control input applied
                 * between stage k and k+1
                 * 
                 * Accessible directly for warm-start initialization
                 */
                std::array<matrix::StaticVector<double, Nu>, H> U_safe;

                /**
                 * @brief Indicates whether a valid trajectory has been stored
                 * 
                 * false : 
                 *  No feasible solution has been committed yet.
                 * 
                 * true : 
                 *  X_safe and U_safe contain a valid trajectory
                 */
                bool has_valid_trajectory = false;

                /**
                 * @brief Construct an empty safe trajectory buffer
                 * 
                 * Initializes all state and control vectors to zero and marks
                 * the buffer as invalid
                 */
                SafeTrajectoryBuffer() {
                    for (std::size_t k = 0; k <= H; ++k) {
                        X_safe[k].set_zero();
                    }
                    for (std::size_t k = 0; k < H; ++k) {
                        U_safe[k].set_zero();
                    }
                }

                /**
                 * @brief Store a feasible trajectory
                 * 
                 * This function should be called whenever the optimizer
                 * successfully computes a feasible solution
                 * 
                 * @param X Predicted state trajectory
                 * @param U Predicted control trajectory
                 */
                void commit(
                    const std::array<matrix::StaticVector<double, Nx>, H + 1>& X,
                    const std::array<matrix::StaticVector<double, Nu>, H>& U
                ) {
                    X_safe = X;
                    U_safe = U;

                    has_valid_trajectory = true;
                }

                /**
                 * @brief Generate a deterministic emergency control command
                 * 
                 * This method is intended to be called when an external supervisory
                 * component (KKT Monitor, solver supervisor, timeout monitor, etc..)
                 * has already determined that optimization has failed
                 * 
                 * Control vector convention:
                 *  u(0) : Steering command
                 *  u(1) : Longitudinal acceleration command
                 * 
                 * Any alternative control ordering will produce incorrect behavior
                 * 
                 * @param braking_acceleration Emergency braking command. Must be less than or equal to zero
                 * @return Emergency fallback control input
                 */
                matrix::StaticVector<double, Nu> generate_fallback_control(
                    double braking_acceleration) const {
                    static_assert(Nu >= 2, "Fallback control logic requires at least two control inputs: steering and acceleration");
                    assert(braking_acceleration <= 0.0 && "Emergency braking acceleration must be <= 0");
                    matrix::StaticVector<double, Nu> safe_u;

                    safe_u.set_zero();

                    // Steering straight ahead
                    safe_u(0) = 0.0;
                    // Maximum braking command
                    safe_u(1) = braking_acceleration;

                    return safe_u;

                    // TODO
                    /**
                     * u(0), u(1) 하드코딩 제거를 위해
                     * enum ControlIndex {
                     *      STEERING = 0,
                     *      ACCELERATION = 1
                     * };
                     * 
                     * ---
                     * safe_u(ControlIndex::STEERING);
                     * safe_u(ControlIndex::ACCELERATION);
                     */
                }
        };
    } // namespace controller
} // namespace Optimization
#endif // OPTIMIZATION_CONTROL_SAFE_TRAJECTORY_BUFFER_HPP_