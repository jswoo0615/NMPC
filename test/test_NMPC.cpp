#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>
#include <random>

#include "Optimization/Control/SparseNMPC_IPM.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Estimator/EKF.hpp"
#include "Optimization/Estimator/SparseMHE.hpp"

using namespace Optimization;

int main() {
    constexpr size_t Nx = 8;
    constexpr size_t Nu = 2;
    constexpr size_t H = 100;
    constexpr size_t MHE_H = 10;
    double dt = 0.05;

    Optimization::controller::SparseNMPC_IPM<H> nmpc;
    nmpc.dt = dt;

    Optimization::controller::NMPCTuningConfig config;
    config.Q_D = 20.0;        // d: 차선 중앙 복귀 의지를 매우 강하게
    config.Q_mu = 10.0;       // mu: 헤딩 변경 페널티 (정상적인 회피 조향 허용)
    config.Q_Vx = 500.0;      // vx: 속도 유지 강하게
    config.Q_Vy = 1.0;        // vy
    config.Q_r = 1.0;         // r
    config.Q_alpha_f = 1000.0; // alpha_f: 슬립 방지
    config.Q_alpha_r = 1000.0; // alpha_r: 슬립 방지
    config.R_Steer = 1.0;     // R_Steer: 스티어링 페널티를 낮춰서 기민하게 복귀하도록 설정
    config.R_Accel = 10.0;    // R_Accel: 가감속 페널티 하향

    config.u_min[0] = -0.6;  config.u_min[1] = -10.0;
    config.u_max[0] = 0.6;   config.u_max[1] = 10.0;

    config.Obstacle_Margin = 0.5; // 도로 경계(-3.5 ~ 3.5) 내에서 회피 가능하도록 마진 축소

    // 장애물 설정
    nmpc.obstacles[0].s = 25.0;
    nmpc.obstacles[0].d = 0.005;
    nmpc.obstacles[0].r = 2.5;

    // 초기 상태 설정
    matrix::StaticVector<double, Nx> x0;
    x0.set_zero();
    x0(0) = 0.0;    // s
    x0(1) = 0.0;    // d
    x0(2) = 0.087;  // mu (약 5도 틀어짐)
    x0(3) = 10.0;   // vx

    // EKF 설정
    Optimization::estimator::EKF<Nx, Nu> ekf;
    ekf.x_est = x0;

    // SparseMHE 설정
    Optimization::estimator::SparseMHE<MHE_H, Dynamics::RealTimeDynamicsModel, Nx, Nu, Nx> mhe;
    mhe.dt = dt;

    Dynamics::RealTimeDynamicsModel dynamics;

    std::ofstream log_file("nmpc_obstacle_avoidance.csv");
    log_file << "step,s_true,d_true,mu_true,vx_true,s_ekf,d_ekf,mu_ekf,vx_ekf,s_mhe,d_mhe,mu_mhe,vx_mhe,delta,a\n";

    std::mt19937 gen(42);
    std::normal_distribution<double> noise_s(0.0, 0.05);
    std::normal_distribution<double> noise_d(0.0, 0.02);
    std::normal_distribution<double> noise_mu(0.0, 0.01);
    std::normal_distribution<double> noise_v(0.0, 0.1);
    std::normal_distribution<double> noise_gen(0.0, 0.05);

    std::cout << "Starting Simulation with EKF/MHE Integration...\n";
    for (int step = 0; step < 100; ++step) {
        // 1. 센서 노이즈 생성 (가우시안 노이즈 추가)
        matrix::StaticVector<double, Nx> z = x0;
        z(0) += noise_s(gen);
        z(1) += noise_d(gen);
        z(2) += noise_mu(gen);
        z(3) += noise_v(gen);
        z(4) += noise_gen(gen);
        z(5) += noise_gen(gen);
        z(6) += noise_gen(gen);
        z(7) += noise_gen(gen);

        // 2. 상태 추정 업데이트
        // EKF 업데이트
        ekf.update(z);
        
        // MHE 업데이트
        mhe.Z_history.push(z);
        mhe.solve();
        matrix::StaticVector<double, Nx> x_mhe = mhe.X_opt[MHE_H];

        // 3. NMPC 제어 입력 계산 (EKF 추정치 기반)
        auto res = nmpc.solve_ipm(ekf.x_est, config);
        
        matrix::StaticVector<double, Nu> u_opt;
        if (res.success || res.fallback_triggered) {
            u_opt = nmpc.u_last;
        } else {
            u_opt.set_zero();
        }

        std::cout << std::setw(4) << step << " | "
                  << "True: " << std::fixed << std::setprecision(2) << std::setw(6) << x0(0) << "," << std::setw(5) << x0(1) << " | "
                  << "EKF: " << std::setw(6) << ekf.x_est(0) << "," << std::setw(5) << ekf.x_est(1) << " | "
                  << "MHE: " << std::setw(6) << x_mhe(0) << "," << std::setw(5) << x_mhe(1) << " || "
                  << "u: " << std::setw(6) << u_opt(0) << "," << std::setw(6) << u_opt(1) << " | "
                  << res.status_msg << "\n";

        log_file << step << ","
                 << x0(0) << "," << x0(1) << "," << x0(2) << "," << x0(3) << ","
                 << ekf.x_est(0) << "," << ekf.x_est(1) << "," << ekf.x_est(2) << "," << ekf.x_est(3) << ","
                 << x_mhe(0) << "," << x_mhe(1) << "," << x_mhe(2) << "," << x_mhe(3) << ","
                 << u_opt(0) << "," << u_opt(1) << "\n";

        // 4. 실제 차량 물리 모델 상태 업데이트 (Integrator)
        x0 = integrator::IntegratorEngine<Nx, Nu, Dynamics::RealTimeDynamicsModel, double>::compute(dynamics, x0, u_opt, dt);

        // Zero-Velocity Clamp (후진 방지)
        if (x0(3) < 0.0) {
            x0(3) = 0.0; 
        }

        // 5. 다음 스텝을 위한 상태 및 입력 갱신 (Shift & Predict)
        ekf.predict(dynamics, u_opt, dt);
        mhe.U_history.push(u_opt);
        mhe.shift_sequence(u_opt);
        
        nmpc.shift_sequence();
    }

    log_file.close();
    return 0;
}