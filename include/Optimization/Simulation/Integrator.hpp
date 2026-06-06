#ifndef OPTIMIZATION_SIMULATION_INTEGRATOR_HPP_
#define OPTIMIZATION_SIMULATION_INTEGRATOR_HPP_

#include "Optimization/Matrix/Core/StaticMatrix.hpp"

namespace Optimization {
namespace integrator {

template <size_t Nx, size_t Nu, typename Model, typename T>
inline matrix::StaticVector<T, Nx> step_rk4(const Model& model, 
                                            const matrix::StaticVector<T, Nx>& x, 
                                            const matrix::StaticVector<T, Nu>& u, 
                                            double dt_double) {
    T dt = static_cast<T>(dt_double);
    matrix::StaticVector<T, Nx> k1 = model(x, u);
    
    matrix::StaticVector<T, Nx> x_k2;
    for (size_t i = 0; i < Nx; ++i) x_k2(i) = x(i) + k1(i) * (dt * T(0.5));
    matrix::StaticVector<T, Nx> k2 = model(x_k2, u);
    
    matrix::StaticVector<T, Nx> x_k3;
    for (size_t i = 0; i < Nx; ++i) x_k3(i) = x(i) + k2(i) * (dt * T(0.5));
    matrix::StaticVector<T, Nx> k3 = model(x_k3, u);
    
    matrix::StaticVector<T, Nx> x_k4;
    for (size_t i = 0; i < Nx; ++i) x_k4(i) = x(i) + k3(i) * dt;
    matrix::StaticVector<T, Nx> k4 = model(x_k4, u);
    
    matrix::StaticVector<T, Nx> x_next;
    for (size_t i = 0; i < Nx; ++i) {
        x_next(i) = x(i) + (k1(i) + k2(i) * T(2.0) + k3(i) * T(2.0) + k4(i)) * (dt / T(6.0));
    }
    return x_next;
}

template <size_t Nx, size_t Nu, typename Model, typename T>
struct IntegratorEngine {
    static inline matrix::StaticVector<T, Nx> compute(const Model& model, 
                                                      const matrix::StaticVector<T, Nx>& x, 
                                                      const matrix::StaticVector<T, Nu>& u, 
                                                      double dt) {
        return step_rk4<Nx, Nu, Model, T>(model, x, u, dt);
    }
};

} // namespace integrator
} // namespace Optimization

#endif // OPTIMIZATION_SIMULATION_INTEGRATOR_HPP_
