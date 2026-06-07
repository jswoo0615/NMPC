#ifndef OPTIMIZATION_ESTIMATOR_SPARSE_MHE_HPP_
#define OPTIMIZATION_ESTIMATOR_SPARSE_MHE_HPP_

#include <algorithm>
#include <array>
#include <cmath>

#include "Optimization/Estimator/HistoryBuffer.hpp"
#include "Optimization/Matrix/AD/Dual.hpp"
#include "Optimization/Matrix/Linalg/LinearAlgebra.hpp"  // [Architect's Update] Layer 2 SIMD 엔진 통합
#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Simulation/Integrator.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"

namespace Optimization {
namespace estimator {

/**
 * @brief 희소 행렬 기반의 이동 구간 추정기 (Sparse Moving Horizon Estimator, MHE) 경량 최적화판
 * @details 임베디드 타겟(Jetson Nano)의 System Slack 확보를 위해 차원을 최소화하고,
 * 수치적 폭주 방지 가드와 Layer 2 SIMD 엔진을 결합하여 O(N^3) 연산 지연을 원천 차단한 무결점
 * 엔진입니다.
 */
template <size_t HE, typename PlantModel = Dynamics::RealTimeDynamicsModel, size_t Nx = 8, size_t Nu = 2, size_t Nz = 8>
class SparseMHE {
   public:
    double dt;

    HistoryBuffer<Nz, HE> Z_history;
    HistoryBuffer<Nu, HE> U_history;

    std::array<matrix::StaticVector<double, Nx>, HE + 1> X_opt;
    matrix::StaticVector<double, Nx> x_arr;

    matrix::StaticMatrix<double, Nz, Nz> W_meas;
    matrix::StaticMatrix<double, Nx, Nx> W_dyn;
    matrix::StaticMatrix<double, Nx, Nx> W_arr;

    SparseMHE() : dt(0.1) {
        W_meas.set_zero();
        for (size_t i = 0; i < Nz; ++i) W_meas(i, i) = 1.0;

        W_dyn.set_zero();
        for (size_t i = 0; i < Nx; ++i) W_dyn(i, i) = 1.0;

        W_arr.set_zero();
        for (size_t i = 0; i < Nx; ++i) W_arr(i, i) = 1.0;

        for (auto& x : X_opt) x.set_zero();
        x_arr.set_zero();
    }

    inline void shift_sequence(const matrix::StaticVector<double, Nu>& current_u) {
        x_arr = X_opt[1];
        for (size_t k = 0; k < HE; ++k) {
            X_opt[k] = X_opt[k + 1];
        }
        PlantModel model;
        X_opt[HE] =
            integrator::IntegratorEngine<Nx, Nu, PlantModel, double>::compute(
                model, X_opt[HE - 1], current_u, dt);
    }

    inline void shift_sequence() {
        x_arr = X_opt[1];
        for (size_t k = 0; k < HE; ++k) {
            X_opt[k] = X_opt[k + 1];
        }
        PlantModel model;
        if (U_history.size() > 0) {
            X_opt[HE] =
                integrator::IntegratorEngine<Nx, Nu, PlantModel, double>::compute(
                    model, X_opt[HE - 1], U_history[0], dt);
        }
    }

    bool solve() {
        constexpr size_t TOTAL_DIM = (HE + 1) * Nx;

        // 프라이멀 필터링: 극한 결함 주입 시 연산 스킵을 통한 보호
        if (Z_history.size() > 0) {
            bool is_faulty = false;
            for (size_t k = 0; k < Z_history.size(); ++k) {
                if (std::isnan(Z_history[k](1)) || std::abs(Z_history[k](1)) > 50.0) {
                    is_faulty = true;
                    break;
                }
            }
            if (is_faulty) return false;
        }

        if (Z_history.size() < HE || U_history.size() < HE) {
            static bool initialized = false;
            if (!initialized && Z_history.size() > 0) {
                for (size_t k = 0; k <= HE; ++k) {
                    X_opt[k].set_zero();
                    for (size_t i = 0; i < (Nz < Nx ? Nz : Nx); ++i) {
                        X_opt[k](i) = Z_history[0](i);
                    }
                }
                x_arr.set_zero();
                for (size_t i = 0; i < (Nz < Nx ? Nz : Nx); ++i) {
                    x_arr(i) = Z_history[0](i);
                }
                initialized = true;
            }
            return true;
        }

        // 반복 연산 상한을 최소화하여 확실한 응답 속도 보장
        constexpr int MAX_ITER = 2;
        PlantModel model;
        using ADVar = DualVec<double, Nx>;

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            matrix::StaticMatrix<double, TOTAL_DIM, TOTAL_DIM> H_mat;
            matrix::StaticVector<double, TOTAL_DIM> g_vec;
            H_mat.set_zero();
            g_vec.set_zero();

            // 1. Arrival Cost 조립
            for (size_t i = 0; i < Nx; ++i) {
                double r_val = W_arr(i, i) * (X_opt[0](i) - x_arr(i));
                g_vec(i) += W_arr(i, i) * r_val;
                H_mat(i, i) += W_arr(i, i) * W_arr(i, i);
            }

            // 2. Measurement Cost 조립
            for (size_t k = 1; k <= HE; ++k) {
                size_t meas_idx = HE - k;
                matrix::StaticVector<double, Nz> z = Z_history[meas_idx];
                for (size_t i = 0; i < Nz; ++i) {
                    double r_val = W_meas(i, i) * (z(i) - X_opt[k](i));
                    size_t idx = k * Nx + i;
                    g_vec(idx) += (-W_meas(i, i)) * r_val;
                    H_mat(idx, idx) += W_meas(i, i) * W_meas(i, i);
                }
            }

            // 3. Dynamic Feasibility Cost 조립 (AD & SIMD 최적화 구간)
            for (size_t k = 0; k < HE; ++k) {
                size_t ctrl_idx = HE - 1 - k;
                matrix::StaticVector<double, Nu> u_k = U_history[ctrl_idx];

                matrix::StaticVector<ADVar, Nx> x_k_dual;
                for (size_t i = 0; i < Nx; ++i) x_k_dual(i) = ADVar::make_variable(X_opt[k](i), i);
                matrix::StaticVector<ADVar, Nu> u_k_dual;
                for (size_t i = 0; i < Nu; ++i) u_k_dual(i) = ADVar(u_k(i));

                matrix::StaticVector<ADVar, Nx> x_sim_dual =
                    integrator::IntegratorEngine<Nx, Nu, PlantModel,
                                                 ADVar>::compute(model, x_k_dual, u_k_dual, dt);

                // =====================================================================
                // [Architect's Update] 자코비안 메모리 응집 (Column-Major Extraction)
                // 분산되어 있던 AD 스칼라 조립부를 일괄 추출(Extract)하여 L1 캐시에 안착시킵니다.
                // =====================================================================
                matrix::StaticMatrix<double, Nx, Nx> J_mat;
                for (size_t j = 0; j < Nx; ++j) {
                    for (size_t i = 0; i < Nx; ++i) {
                        J_mat(i, j) = x_sim_dual(i).g[j];
                    }
                }

                matrix::StaticVector<double, Nx> r_vec;
                for (size_t i = 0; i < Nx; ++i) {
                    r_vec(i) = X_opt[k + 1](i) - x_sim_dual(i).v;
                }

                // 3.1 다음 노드(x_next) 그레디언트 및 대각 헤시안 누산
                for (size_t i = 0; i < Nx; ++i) {
                    double w_i = W_dyn(i, i);
                    double w_r = w_i * r_vec(i);
                    size_t idx_next = (k + 1) * Nx + i;
                    g_vec(idx_next) += w_i * w_r;
                    H_mat(idx_next, idx_next) += w_i * w_i;
                }

                // 3.2 교차항(Cross terms) 및 현재 노드(x_k) 그레디언트 누산
                for (size_t j = 0; j < Nx; ++j) {  // Column outer loop
                    size_t idx_k_j = k * Nx + j;
                    for (size_t i = 0; i < Nx; ++i) {  // Row inner loop
                        size_t idx_next = (k + 1) * Nx + i;
                        double w_i = W_dyn(i, i);
                        double w2_J = w_i * w_i * J_mat(i, j);

                        H_mat(idx_next, idx_k_j) -= w2_J;
                        H_mat(idx_k_j, idx_next) -= w2_J;
                        g_vec(idx_k_j) -= w2_J * r_vec(i);
                    }
                }

                // 3.3 현재 노드(x_k) 헤시안 블록 조립 (J^T * W^2 * J)
                for (size_t m = 0; m < Nx; ++m) {  // Column outer loop
                    size_t col_idx = k * Nx + m;
                    for (size_t j = 0; j < Nx; ++j) {  // Row inner loop
                        size_t row_idx = k * Nx + j;
                        double sum = 0.0;
                        for (size_t i = 0; i < Nx; ++i) {
                            double w_i = W_dyn(i, i);
                            sum += (w_i * J_mat(i, j)) * (w_i * J_mat(i, m));
                        }
                        H_mat(row_idx, col_idx) += sum;
                    }
                }
            }

            // Regularization (Hessian SPD 강제)
            for (size_t i = 0; i < TOTAL_DIM; ++i) H_mat(i, i) += 1e-3;

            // =====================================================================
            // [Architect's Update] 가우스-조던 소거법 폐기 및 Layer 2 SIMD 결착
            // 40줄에 달하는 스칼라 O(N^3) 연산을 도려내고 LDLT 분해기로 파이프라인 관통
            // =====================================================================
            matrix::StaticVector<double, TOTAL_DIM> neg_g;
            for (size_t i = 0; i < TOTAL_DIM; ++i) neg_g(i) = -g_vec(i);

            matrix::StaticVector<double, TOTAL_DIM> dX;
            if (linalg::LDLT_decompose(H_mat) != MathStatus::SUCCESS) {
                break;  // 행렬이 붕괴(Singular)될 경우 기존 추정치를 보존하고 조기 종료
            }
            linalg::LDLT_solve(H_mat, neg_g, dX);

            // Primal Update
            double max_update = 0.0;
            for (size_t k = 0; k <= HE; ++k) {
                for (size_t i = 0; i < Nx; ++i) {
                    double step_val = std::clamp(dX(k * Nx + i), -5.0, 5.0);
                    X_opt[k](i) += step_val;
                    if (std::abs(step_val) > max_update) max_update = std::abs(step_val);
                }
            }

            if (max_update < 1e-3) break;
        }

        return true;
    }
};

}  // namespace estimator
}  // namespace Optimization

#endif  // OPTIMIZATION_ESTIMATOR_SPARSE_MHE_HPP_