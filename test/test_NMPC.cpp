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
    // 1. 아키텍처 환경 설정
    constexpr size_t H = 50;           // 회피 기동을 위해 시야를 2.0초로 확장
    constexpr size_t Nx = 8;
    constexpr size_t Nu = 2;
    constexpr size_t Ncx = 1;          // [핵심] 장애물 상태 제약 1개
    constexpr size_t Ncu = 4;
    constexpr double dt = 0.05;

    // 메인 컨트롤러 인스턴스화
    NMPC_Controller<RealTimeDynamicsModel, H, Nx, Nu, Ncx, Ncu> nmpc;

    // 2. 가중치 제정
    matrix::StaticMatrix<double, Nx, Nx> Q;
    Q.set_zero();
    Q(1, 1) = 100.0;  // d
    Q(2, 2) = 50.0;   // mu
    Q(3, 3) = 10.0;   // vx
    Q(4, 4) = 1.0;    // vy
    Q(5, 5) = 1.0;    // r
    Q(6, 6) = 1000.0; // alpha_f
    Q(7, 7) = 1000.0; // alpha_r

    matrix::StaticMatrix<double, Nx, Nx> Q_term = Q;
    for (size_t i = 0; i < Nx; ++i) Q_term(i, i) *= 5.0; 

    matrix::StaticMatrix<double, Nu, Nu> R;
    R.set_zero();
    R(0, 0) = 5.0;    // 조향 가중치는 적당히 유지
    R(1, 1) = 50.0;   

    matrix::StaticVector<double, Nu> u_min;
    u_min(0) = -0.6;  u_min(1) = -10.0; 
    matrix::StaticVector<double, Nu> u_max;
    u_max(0) = 0.6;   u_max(1) = 10.0;  

    nmpc.init(Q, Q_term, R, u_min, u_max, dt);

    // [Architect's Update] 장애물 선포: 전방 25m 지점, 차선 중앙(d=0), 반경 1.5m (차폭 고려)
    nmpc.set_obstacle(25.0, 0.01, 1.5);

    // 3. 상위 플래너 목표 궤적 (장애물 무시하고 차선 중앙 직진 명령)
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;
    for (size_t k = 0; k <= H; ++k) {
        x_ref[k].set_zero();
        x_ref[k](3) = 10.0; 
    }
    nmpc.set_reference_trajectory(x_ref);

    // 4. 초기 상태 설정
    matrix::StaticVector<double, Nx> x0;
    x0.set_zero();
    x0(0) = 0.0;    // s = 0.0m (출발선)
    x0(1) = 2.0;    // d = +2.0m (차선 이탈 상태)
    x0(2) = 0.087;  // mu = +0.087 rad
    x0(3) = 10.0;   // vx = 10.0 m/s

    std::ofstream log_file("nmpc_obstacle_avoidance.csv");
    if (!log_file.is_open()) return -1;
    
    // 로깅에 s(종방향 위치) 추가
    log_file << "step,s,d,mu,vx,delta,a\n";

    std::cout << "=== NMPC Obstacle Avoidance Commissioning ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step | s (m)   | d (m)   | mu (rad)| vx (m/s)|| delta (rad)| a (m/s^2)" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;

    // 100스텝(5.0초) 주행 테스트: s=0 부터 s=50 까지 주행하며 s=25의 장애물을 피해야 함
    for (int step = 0; step < 100; ++step) {
        auto u_opt = nmpc.compute_control(x0, 5, 1e-4);

        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << x0(0) << " | "
                  << std::setw(7) << x0(1) << " | "
                  << std::setw(7) << x0(2) << " | "
                  << std::setw(7) << x0(3) << " || "
                  << std::setw(10) << u_opt(0) << " | "
                  << std::setw(9) << u_opt(1) << std::endl;

        log_file << step << "," << x0(0) << "," << x0(1) << "," << x0(2) << ","
                 << x0(3) << "," << u_opt(0) << "," << u_opt(1) << "\n";

        matrix::StaticVector<Dual<double>, Nx> x_dual;
        for(size_t i = 0; i < Nx; ++i) x_dual(i) = Dual<double>(x0(i), 0.0);
        
        matrix::StaticVector<Dual<double>, Nu> u_dual;
        for(size_t i = 0; i < Nu; ++i) u_dual(i) = Dual<double>(u_opt(i), 0.0);

        auto x_dot_dual = nmpc.dynamics_model(x_dual, u_dual);

        for(size_t i = 0; i < Nx; ++i) {
            x0(i) += x_dot_dual(i).v * dt; 
        }

        nmpc.shift_trajectory();
    }

    log_file.close();
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "Simulation complete. Data saved to 'nmpc_obstacle_avoidance.csv'." << std::endl;

    return 0;
}