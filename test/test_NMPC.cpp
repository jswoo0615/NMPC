#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>

#include "Optimization/Control/NMPC_Controller.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

using namespace Optimization;
using namespace Optimization::control;
using namespace Optimization::Dynamics;

int main() {
    constexpr size_t H = 30;           
    constexpr size_t Nx = 8;
    constexpr size_t Nu = 2;
    constexpr size_t Ncx = 2; // [Architect's Update] 다중 원형 제약 (Front & Rear)
    constexpr size_t Ncu = 4;
    constexpr double dt = 0.05;

    NMPC_Controller<RealTimeDynamicsModel, H, Nx, Nu, Ncx, Ncu> nmpc;

    matrix::StaticMatrix<double, Nx, Nx> Q;
    Q.set_zero();
    Q(1, 1) = 5.0;    // [수정] d: 차선을 크게 벗어나는 것을 허용 (100.0 -> 5.0)
    Q(2, 2) = 1.0;    // [수정] mu: 차체 헤딩이 크게 틀어지는 것을 허용 (50.0 -> 1.0)
    Q(3, 3) = 500.0;  // [수정] vx: 속도를 잃는 것을 극도로 혐오하게 만듦 (10.0 -> 500.0)
    Q(4, 4) = 1.0;    // vy
    Q(5, 5) = 1.0;    // r
    Q(6, 6) = 1000.0; // alpha_f: 슬립 방지 (유지)
    Q(7, 7) = 1000.0; // alpha_r: 슬립 방지 (유지)

    matrix::StaticMatrix<double, Nx, Nx> Q_term = Q;
    for (size_t i = 0; i < Nx; ++i) Q_term(i, i) *= 5.0; 

    matrix::StaticMatrix<double, Nu, Nu> R;
    R.set_zero();
    R(0, 0) = 5.0;    R(1, 1) = 50.0;   

    matrix::StaticVector<double, Nu> u_min, u_max;
    u_min(0) = -0.6;  u_min(1) = -10.0; 
    u_max(0) = 0.6;   u_max(1) = 10.0;  

    nmpc.init(Q, Q_term, R, u_min, u_max, dt);

    // [Architect's Update] 다중 원형 근사 전용 방벽 세팅
    // 장애물 반경(1.5m) + 차량 부분 반경(0.9m) + 마진(0.1m) = 2.5m
    // 대칭성을 깨기 위해 y축(d)에 0.01 오프셋 부여
    nmpc.set_obstacle(25.0, 0.005, 2.5);

    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;
    for (size_t k = 0; k <= H; ++k) {
        x_ref[k].set_zero();
        x_ref[k](3) = 10.0; 
    }
    nmpc.set_reference_trajectory(x_ref);

    matrix::StaticVector<double, Nx> x0;
    x0.set_zero();
    x0(0) = 0.0;    x0(1) = 0.0;    x0(2) = 0.087;  x0(3) = 20.00;   

    std::ofstream log_file("nmpc_obstacle_avoidance.csv");
    if (!log_file.is_open()) return -1;
    
    log_file << "step,s,d,mu,vx,delta,a\n";

    std::cout << "=== NMPC Dual-Circle Obstacle Avoidance ===" << std::endl;
    for (int step = 0; step < 1000; ++step) {
        auto u_opt = nmpc.compute_control(x0, 5, 1e-4);

        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << x0(0) << " | " << std::setw(7) << x0(1) << " | "
                  << std::setw(7) << x0(2) << " | " << std::setw(7) << x0(3) << " || "
                  << std::setw(10) << u_opt(0) << " | " << std::setw(9) << u_opt(1) << std::endl;

        log_file << step << "," << x0(0) << "," << x0(1) << "," << x0(2) << ","
                 << x0(3) << "," << u_opt(0) << "," << u_opt(1) << "\n";

        matrix::StaticVector<Dual<double>, Nx> x_dual;
        for(size_t i = 0; i < Nx; ++i) x_dual(i) = Dual<double>(x0(i), 0.0);
        matrix::StaticVector<Dual<double>, Nu> u_dual;
        for(size_t i = 0; i < Nu; ++i) u_dual(i) = Dual<double>(u_opt(i), 0.0);

        // [main.cpp 내부 5-2. Plant 시뮬레이션 부분]
        auto x_dot_dual = nmpc.dynamics_model(x_dual, u_dual);

        // Euler Integration: x_{k+1} = x_k + x_dot * dt
        for(size_t i = 0; i < Nx; ++i) {
            x0(i) += x_dot_dual(i).v * dt; 
        }

        // ==========================================================
        // [Architect's Fix] 물리적 후진 방지 (Zero-Velocity Clamp)
        // 차량이 브레이크를 밟아 완전히 정차한 경우, 음수 속도(후진)를 차단합니다.
        if (x0(3) < 0.0) {
            x0(3) = 0.0; 
        }
        // ==========================================================

        nmpc.shift_trajectory();
    }

    log_file.close();
    return 0;
}