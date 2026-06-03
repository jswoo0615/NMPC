#include <iostream>
#include <iomanip>
#include <array>

// 깎아온 핵심 헤더 인클루드 (경로는 프로젝트 구조에 맞게 수정)
#include "Optimization/Control/NMPC_Controller.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

using namespace Optimization;
using namespace Optimization::control;
using namespace Optimization::Dynamics;

int main() {
    // 1. 아키텍처 환경 설정
    constexpr size_t H = 10;           // 예측 구간 (초기 테스트용 짧은 Horizon)
    constexpr size_t Nx = 8;           // 상태 변수
    constexpr size_t Nu = 2;           // 제어 입력
    constexpr double dt = 0.05;        // 제어 주기 (50ms)

    // 메인 컨트롤러 인스턴스화 (RealTimeDynamicsModel 주입)
    NMPC_Controller<RealTimeDynamicsModel, H, Nx, Nu> nmpc;

    // 2. 입법자(Legislator)의 가중치(Penalty) 제정
    matrix::StaticMatrix<double, Nx, Nx> Q;
    Q.set_zero();
    // 종방향 거리(s)는 추종하지 않음 (가중치 0)
    Q(1, 1) = 100.0;  // d: 횡방향 이탈 오차에 강력한 페널티
    Q(2, 2) = 100.0;  // mu: 헤딩(요각) 오차에 강력한 페널티
    Q(3, 3) = 10.0;   // vx: 목표 속도(10m/s) 유지
    Q(4, 4) = 1.0;    // vy: 횡방향 미끄러짐(슬립) 억제
    Q(5, 5) = 1.0;    // r: 급격한 요레이트 발생 억제
    Q(6, 6) = 0.1;    // 전륜 타이어 슬립각 억제
    Q(7, 7) = 0.1;    // 후륜 타이어 슬립각 억제

    matrix::StaticMatrix<double, Nu, Nu> R;
    R.set_zero();
    R(0, 0) = 50.0;   // delta: 과도한 스티어링 억제
    R(1, 1) = 5.0;    // a: 과도한 가감속 억제

    // 컨트롤러에 법(Cost) 공포
    nmpc.init(Q, Q, R);

    // 3. 상위 플래너(Local Planner)의 목표 궤적 수신
    // 시나리오: 일직선 차선(d=0, mu=0)을 10m/s로 정속 주행
    std::array<matrix::StaticVector<double, Nx>, H + 1> x_ref;
    for (size_t k = 0; k <= H; ++k) {
        x_ref[k].set_zero();
        x_ref[k](3) = 10.0; // vx = 10.0 m/s
    }
    nmpc.set_reference_trajectory(x_ref);

    // 4. 초기 상태(Initial State) 설정 (센서 관측값 가정)
    // 악조건: 차선 중앙에서 2m 벗어났고, 헤딩은 5도(약 0.087rad) 틀어진 상태에서 10m/s로 주행 중
    matrix::StaticVector<double, Nx> x0;
    x0.set_zero();
    x0(1) = 2.0;    // d = +2.0m
    x0(2) = 0.087;  // mu = +0.087 rad
    x0(3) = 10.0;   // vx = +10.0 m/s

    // 5. Closed-Loop 시뮬레이션 가동
    std::cout << "=== NMPC Closed-Loop Commissioning ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Step | d (m)   | mu (rad)| vx (m/s)|| delta (rad)| a (m/s^2)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // 30스텝(1.5초) 주행 테스트
    for (int step = 0; step < 30; ++step) {
        // 5-1. NMPC 엔진 구동 (최대 3회 Iteration, 허용 오차 1e-3)
        auto u_opt = nmpc.compute_control(x0, 3, 1e-3);

        // 현재 상태 및 제어 입력 모니터링 출력
        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << x0(1) << " | "
                  << std::setw(7) << x0(2) << " | "
                  << std::setw(7) << x0(3) << " || "
                  << std::setw(10) << u_opt(0) << " | "
                  << std::setw(9) << u_opt(1) << std::endl;

        // 5-2. Plant 시뮬레이션 (Layer 3: 기초적인 Forward Euler 적분기)
        // 실제 차량이 제어 입력을 받아 다음 상태로 넘어가는 물리적 현상을 시뮬레이션
        matrix::StaticVector<Dual<double>, Nx> x_dual;
        for(size_t i = 0; i < Nx; ++i) x_dual(i) = Dual<double>(x0(i), 0.0);
        
        matrix::StaticVector<Dual<double>, Nu> u_dual;
        for(size_t i = 0; i < Nu; ++i) u_dual(i) = Dual<double>(u_opt(i), 0.0);

        // 8차원 동역학 모델 평가
        auto x_dot_dual = nmpc.dynamics_model(x_dual, u_dual);

        // Euler Integration: x_{k+1} = x_k + x_dot * dt
        for(size_t i = 0; i < Nx; ++i) {
            x0(i) += x_dot_dual(i).v * dt; 
        }

        // 5-3. Warm-start 메커니즘 가동 (궤적 1칸 전진)
        nmpc.shift_trajectory();
    }

    return 0;
}