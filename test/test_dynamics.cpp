#include <cmath>
#include <gtest/gtest.h>

#include "Optimization/Dynamics/FastMath.hpp"
#include "Optimization/Dynamics/VehicleParams.hpp"
#include "Optimization/Dynamics/VehiclePhysicsCore.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"
#include "Optimization/Dynamics/HighFidelityDynamicsModel.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

using namespace Optimization;
using namespace Optimization::Dynamics;

// 1. FastMath 테스트
TEST(DynamicsTest, FastMathPrecision) {
    double theta = 0.5; // radian
    EXPECT_NEAR(FastMath::math_sin(theta), std::sin(theta), 1e-3);
    EXPECT_NEAR(FastMath::math_cos(theta), std::cos(theta), 1e-3);
    EXPECT_NEAR(FastMath::math_atan2(0.5, 1.0), std::atan2(0.5, 1.0), 1e-3);
}

// 2. VehicleParams 테스트
TEST(DynamicsTest, VehicleParamsInit) {
    VehicleDynamicsParams<double> v_p = {1500.0, 9.81, 1.2, 1.6, 0.5, 1.6, 1.6, 0.6, 100.0, 3000.0};
    EXPECT_DOUBLE_EQ(v_p.m, 1500.0);
    EXPECT_DOUBLE_EQ(v_p.g, 9.81);
}

// 3. VehiclePhysicsCore 하중 평형 및 타이어 특성 테스트
TEST(DynamicsTest, VehiclePhysicsCoreLoadAndTire) {
    VehicleDynamicsParams<double> v_p = {1500.0, 9.81, 1.2, 1.6, 0.5, 1.6, 1.6, 0.6, 100.0, 3000.0};
    
    // 정적 평형 상태 (v_x=0, yaw_rate=0, a_cmd=0)
    FourWheelLoads<double> loads = computeQuasiStaticLoadTransfer(v_p, 0.0, 0.0, 0.0);
    double total_load = loads.fz_fl + loads.fz_fr + loads.fz_rl + loads.fz_rr;
    double expected_load = v_p.m * v_p.g;
    EXPECT_NEAR(total_load, expected_load, 1e-3);
    
    // 조향 및 슬립각에 따른 타이어 횡력 방향성 테스트
    TireLoadParams<double> t_p = {1.0, 0.0001, 4000.0, 4000.0, 15.0, 1.4, 0.1, 0.15, 0.15};
    double fz = 3750.0; // 1500kg * 9.81 / 4
    double force_pos = computeMagicFormulaTireForce(t_p, fz, 0.05); // 양의 슬립각
    double force_neg = computeMagicFormulaTireForce(t_p, fz, -0.05); // 음의 슬립각
    EXPECT_GT(force_pos, 0.0);
    EXPECT_LT(force_neg, 0.0);
    EXPECT_NEAR(force_pos, -force_neg, 1e-3); // 대칭성 확인
}

// 4. RealTimeDynamicsModel 테스트
TEST(DynamicsTest, RealTimeModelDerivatives) {
    RealTimeDynamicsModel model;
    model.kappa = 0.02; // 곡률이 있는 도로
    
    matrix::StaticVector<double, 8> x;
    x.set_zero();
    x(1) = 0.5; // d = 0.5
    x(2) = 0.1; // mu = 0.1 rad
    x(3) = 15.0; // vx = 15 m/s
    x(4) = 0.2; // vy = 0.2 m/s
    x(5) = 0.1; // r = 0.1 rad/s
    
    matrix::StaticVector<double, 2> u;
    u(0) = 0.02; // delta = 0.02 rad
    u(1) = 1.0; // a = 1.0 m/s^2
    
    matrix::StaticVector<double, 8> x_dot = model(x, u);
    
    // s_dot 계산 검증: s_dot = (vx * cos(mu) - vy * sin(mu)) / (1.0 - d * kappa)
    double expected_denom = 1.0 - x(1) * model.kappa;
    double expected_s_dot = (x(3) * std::cos(x(2)) - x(4) * std::sin(x(2))) / expected_denom;
    EXPECT_NEAR(x_dot(0), expected_s_dot, 1e-3);
    
    // mu_dot 계산 검증: mu_dot = r - kappa * s_dot
    double expected_mu_dot = x(5) - model.kappa * x_dot(0);
    EXPECT_NEAR(x_dot(2), expected_mu_dot, 1e-3);
}

// 5. HighFidelityDynamicsModel 및 Jacobian 추출 테스트
TEST(DynamicsTest, HighFidelityModelJacobian) {
    HighFidelityDynamicsModel model;
    model.kappa = 0.0;
    
    matrix::StaticVector<double, 8> x0;
    x0.set_zero();
    x0(3) = 10.0; // vx = 10 m/s
    
    matrix::StaticVector<double, 2> u0;
    u0.set_zero();
    
    std::array<std::array<double, 8>, 8> A;
    std::array<std::array<double, 2>, 8> B;
    
    model.extractJacobians(x0, u0, A, B);
    
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_FALSE(std::isnan(A[i][j]));
            EXPECT_FALSE(std::isinf(A[i][j]));
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_FALSE(std::isnan(B[i][j]));
            EXPECT_FALSE(std::isinf(B[i][j]));
        }
    }
    
    // mu의 시간에 대한 변화율(index 2)에서 r(index 5)에 대한 편미분은 1이어야 함. (kappa=0이므로)
    EXPECT_NEAR(A[2][5], 1.0, 1e-5);
    
    // d의 시간에 대한 변화율(index 1)에서 vy(index 4)에 대한 편미분은 cos(mu)이어야 함.
    EXPECT_NEAR(A[1][4], 1.0, 1e-5);
}



