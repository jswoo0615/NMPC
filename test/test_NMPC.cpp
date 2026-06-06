#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>

#include "Optimization/Control/SparseNMPC_IPM.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

using namespace Optimization;

int main() {
    constexpr size_t Nx = 8;
    constexpr size_t Nu = 2;
    constexpr size_t H = 30;
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
    config.R_Steer = 1.0;     // R_Steer: 스티어링 페널티를 낮춰서 기민하게 복귀하도록 설정 (이전 5.0)
    config.R_Accel = 10.0;    // R_Accel: 가감속 페널티 하향

    config.u_min[0] = -0.6;  config.u_min[1] = -10.0;
    config.u_max[0] = 0.6;   config.u_max[1] = 10.0;

    config.Obstacle_Margin = 0.5; // 도로 경계(-3.5 ~ 3.5) 내에서 회피 가능하도록 마진 축소

    // 장애물 설정
    nmpc.obstacles[0].s = 25.0;
    nmpc.obstacles[0].d = 0.005;
    nmpc.obstacles[0].r = 2.5;

    matrix::StaticVector<double, Nx> x0;
    x0.set_zero();
    x0(0) = 0.0;    // s
    x0(1) = 0.0;    // d
    x0(2) = 0.087;  // mu (약 5도 틀어짐)
    x0(3) = 20.00;  // vx

    Dynamics::RealTimeDynamicsModel dynamics;

    std::ofstream log_file("nmpc_obstacle_avoidance.csv");
    log_file << "step,s,d,mu,vx,delta,a\n";

    std::cout << "Starting Simulation...\n";
    for (int step = 0; step < 1000; ++step) {
        auto res = nmpc.solve_ipm(x0, config);
        
        matrix::StaticVector<double, Nu> u_opt;
        if (res.success || res.fallback_triggered) {
            u_opt = nmpc.u_last;
        } else {
            u_opt.set_zero();
        }

        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << x0(0) << " | " << std::setw(7) << x0(1) << " | "
                  << std::setw(7) << x0(2) << " | " << std::setw(7) << x0(3) << " || "
                  << std::setw(10) << u_opt(0) << " | " << std::setw(10) << u_opt(1) << "\n";

        log_file << step << "," << x0(0) << "," << x0(1) << "," << x0(2) << ","
                 << x0(3) << "," << u_opt(0) << "," << u_opt(1) << "\n";

        x0 = integrator::IntegratorEngine<Nx, Nu, Dynamics::RealTimeDynamicsModel, double>::compute(dynamics, x0, u_opt, dt);

        // ==========================================================
        // [Architect's Fix] 물리적 후진 방지 (Zero-Velocity Clamp)
        // 차량이 브레이크를 밟아 완전히 정차한 경우, 음수 속도(후진)를 차단합니다.
        if (x0(3) < 0.0) {
            x0(3) = 0.0; 
        }
        // ==========================================================

        nmpc.shift_sequence();
    }

    log_file.close();
    return 0;
}