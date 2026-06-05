#ifndef OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_
#define OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_

#include <array>
#include <cmath>
#include <algorithm>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"

namespace Optimization {
namespace solver {

template <size_t H, size_t Nx, size_t Nu, size_t Ncx, size_t Ncu>
class RT_QPSolver {
   public:
    static constexpr double SIGMA = 0.5;
    static constexpr double TAU = 0.995;
    static constexpr double MIN_SLACK = 1e-8;

    static constexpr double W_SLACK_X = 1e5; 
    static constexpr double W_SLACK_U = 1e5;

    RiccatiSolver<H, Nx, Nu> riccati;

    std::array<matrix::StaticMatrix<double, Ncx, Nx>, H + 1> Cx;
    std::array<matrix::StaticVector<double, Ncx>, H + 1> dcx;
    
    std::array<matrix::StaticMatrix<double, Ncu, Nu>, H> Cu;
    std::array<matrix::StaticVector<double, Ncu>, H> dcu;

    std::array<matrix::StaticVector<double, Ncx>, H + 1> sx, zx; 
    std::array<matrix::StaticVector<double, Ncu>, H> su, zu;     

    std::array<matrix::StaticVector<double, Ncx>, H + 1>& dual_x = zx;

    std::array<matrix::StaticVector<double, Nx>, H + 1> dx_opt;
    std::array<matrix::StaticVector<double, Nu>, H> du_opt;

    void init_interior_points() {
        for (size_t k = 0; k <= H; ++k) {
            for (size_t i = 0; i < Nx; ++i) dx_opt[k](i) = 0.0; 
            for (size_t i = 0; i < Ncx; ++i) { sx[k](i) = 1.0; zx[k](i) = 1.0; }
            if (k < H) {
                for (size_t i = 0; i < Nu; ++i) du_opt[k](i) = 0.0;
                for (size_t i = 0; i < Ncu; ++i) { su[k](i) = 1.0; zu[k](i) = 1.0; }
            }
        }
    }

    SolverStatus solve(
        const std::array<matrix::StaticMatrix<double, Nx, Nx>, H + 1>& Q,
        const std::array<matrix::StaticMatrix<double, Nu, Nu>, H>& R,
        const std::array<matrix::StaticVector<double, Nx>, H + 1>& q,
        const std::array<matrix::StaticVector<double, Nu>, H>& r,
        const std::array<matrix::StaticMatrix<double, Nx, Nx>, H>& A,
        const std::array<matrix::StaticMatrix<double, Nx, Nu>, H>& B,
        const std::array<matrix::StaticVector<double, Nx>, H>& d,
        int max_iter = 20, double tol = 1e-4) {

        for (int iter = 0; iter < max_iter; ++iter) {
            double mu = 0.0;
            size_t total_constraints = (H + 1) * Ncx + H * Ncu;
            
            std::array<matrix::StaticVector<double, Ncx>, H + 1> r_cx, r_dx;
            std::array<matrix::StaticVector<double, Ncu>, H> r_cu, r_du;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cx_dx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) cx_dx += Cx[k](i, j) * dx_opt[k](j);
                    mu += sx[k](i) * zx[k](i);
                    r_dx[k](i) = cx_dx + dcx[k](i) + sx[k](i); 
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cu_du = 0.0;
                        for (size_t j = 0; j < Nu; ++j) cu_du += Cu[k](i, j) * du_opt[k](j);
                        mu += su[k](i) * zu[k](i);
                        r_du[k](i) = cu_du + dcu[k](i) + su[k](i);
                    }
                }
            }
            mu /= static_cast<double>(total_constraints);

            double max_res = 0.0;
            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    r_cx[k](i) = sx[k](i) * zx[k](i) - SIGMA * mu;
                    if (std::abs(r_dx[k](i)) > max_res) max_res = std::abs(r_dx[k](i));
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        r_cu[k](i) = su[k](i) * zu[k](i) - SIGMA * mu;
                        if (std::abs(r_du[k](i)) > max_res) max_res = std::abs(r_du[k](i));
                    }
                }
            }

            if (max_res < tol && mu < tol) {
                return SolverStatus::SUCCESS;
            }

            for (size_t k = 0; k <= H; ++k) {
                riccati.Q[k] = Q[k];
                
                for (size_t i = 0; i < Nx; ++i) {
                    double q_dx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) q_dx += Q[k](i, j) * dx_opt[k](j);
                    riccati.q[k](i) = q[k](i) + q_dx;
                }

                for (size_t i = 0; i < Ncx; ++i) {
                    double inv_s = 1.0 / MathTraits<double>::max(sx[k](i), MIN_SLACK);
                    double w_x_hard = MathTraits<double>::max(zx[k](i) * inv_s, MIN_SLACK);
                    
                    double w_x = 1.0 / ((1.0 / w_x_hard) + (1.0 / W_SLACK_X));
                    
                    // [Architect's Core Fix] Gradient-Hessian Consistency
                    // 선형화 벡터 보정 항(t_x)에도 소프트 페널티 가중치 w_x를 엄밀하게 적용하여 수치 붕괴 방지
                    double t_x = w_x * r_dx[k](i) - r_cx[k](i) * inv_s;

                    for (size_t r_idx = 0; r_idx < Nx; ++r_idx) {
                        riccati.q[k](r_idx) += Cx[k](i, r_idx) * (zx[k](i) + t_x);
                        for (size_t c_idx = 0; c_idx < Nx; ++c_idx) {
                            riccati.Q[k](r_idx, c_idx) += Cx[k](i, r_idx) * w_x * Cx[k](i, c_idx);
                        }
                    }
                }

                if (k < H) {
                    riccati.R[k] = R[k];
                    
                    for (size_t i = 0; i < Nu; ++i) {
                        double r_du = 0.0;
                        for (size_t j = 0; j < Nu; ++j) r_du += R[k](i, j) * du_opt[k](j);
                        riccati.r[k](i) = r[k](i) + r_du;
                    }

                    for (size_t i = 0; i < Nx; ++i) {
                        double a_dx = 0.0, b_du = 0.0;
                        for (size_t j = 0; j < Nx; ++j) a_dx += A[k](i, j) * dx_opt[k](j);
                        for (size_t j = 0; j < Nu; ++j) b_du += B[k](i, j) * du_opt[k](j);
                        riccati.d[k](i) = a_dx + b_du + d[k](i) - dx_opt[k+1](i);
                    }
                    
                    riccati.A[k] = A[k];
                    riccati.B[k] = B[k];

                    for (size_t i = 0; i < Ncu; ++i) {
                        double inv_s = 1.0 / MathTraits<double>::max(su[k](i), MIN_SLACK);
                        double w_u_hard = MathTraits<double>::max(zu[k](i) * inv_s, MIN_SLACK);
                        
                        double w_u = 1.0 / ((1.0 / w_u_hard) + (1.0 / W_SLACK_U));
                        
                        // [Architect's Core Fix] Input Constraint Consistency
                        double t_u = w_u * r_du[k](i) - r_cu[k](i) * inv_s;

                        for (size_t r_idx = 0; r_idx < Nu; ++r_idx) {
                            riccati.r[k](r_idx) += Cu[k](i, r_idx) * (zu[k](i) + t_u);
                            for (size_t c_idx = 0; c_idx < Nu; ++c_idx) {
                                riccati.R[k](r_idx, c_idx) += Cu[k](i, r_idx) * w_u * Cu[k](i, c_idx);
                            }
                        }
                    }
                }
            }

            if (riccati.solve(1e-6, 1e-6) != SolverStatus::SUCCESS) {
                return SolverStatus::MATH_ERROR;
            }

            std::array<matrix::StaticVector<double, Ncx>, H + 1> dsx, dzx;
            std::array<matrix::StaticVector<double, Ncu>, H> dsu, dzu;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cx_ddx = 0.0;
                    for (size_t j = 0; j < Nx; ++j) cx_ddx += Cx[k](i, j) * riccati.dx[k](j);

                    dsx[k](i) = -r_dx[k](i) - cx_ddx;
                    double inv_s = 1.0 / MathTraits<double>::max(sx[k](i), MIN_SLACK);
                    dzx[k](i) = -(r_cx[k](i) + zx[k](i) * dsx[k](i)) * inv_s;
                }

                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cu_ddu = 0.0;
                        for (size_t j = 0; j < Nu; ++j) cu_ddu += Cu[k](i, j) * riccati.du[k](j);

                        dsu[k](i) = -r_du[k](i) - cu_ddu;
                        double inv_s = 1.0 / MathTraits<double>::max(su[k](i), MIN_SLACK);
                        dzu[k](i) = -(r_cu[k](i) + zu[k](i) * dsu[k](i)) * inv_s;
                    }
                }
            }

            double alpha_p = 1.0, alpha_d = 1.0;

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Ncx; ++i) {
                    double cp = (dsx[k](i) < 0.0) ? (-TAU * sx[k](i) / dsx[k](i)) : 1.0;
                    alpha_p = MathTraits<double>::min(alpha_p, cp);
                    double cd = (dzx[k](i) < 0.0) ? (-TAU * zx[k](i) / dzx[k](i)) : 1.0;
                    alpha_d = MathTraits<double>::min(alpha_d, cd);
                }
                if (k < H) {
                    for (size_t i = 0; i < Ncu; ++i) {
                        double cp = (dsu[k](i) < 0.0) ? (-TAU * su[k](i) / dsu[k](i)) : 1.0;
                        alpha_p = MathTraits<double>::min(alpha_p, cp);
                        double cd = (dzu[k](i) < 0.0) ? (-TAU * zu[k](i) / dzu[k](i)) : 1.0;
                        alpha_d = MathTraits<double>::min(alpha_d, cd);
                    }
                }
            }

            for (size_t k = 0; k <= H; ++k) {
                for (size_t i = 0; i < Nx; ++i) dx_opt[k](i) += alpha_p * riccati.dx[k](i); 
                for (size_t i = 0; i < Ncx; ++i) {
                    sx[k](i) += alpha_p * dsx[k](i);
                    zx[k](i) += alpha_d * dzx[k](i);
                }
                if (k < H) {
                    for (size_t i = 0; i < Nu; ++i) du_opt[k](i) += alpha_p * riccati.du[k](i);
                    for (size_t i = 0; i < Ncu; ++i) {
                        su[k](i) += alpha_p * dsu[k](i);
                        zu[k](i) += alpha_d * dzu[k](i);
                    }
                }
            }
        }
        return SolverStatus::MAX_ITERATION_REACHED;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_RT_QP_SOLVER_HPP_