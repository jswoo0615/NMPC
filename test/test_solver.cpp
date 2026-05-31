#include <gtest/gtest.h>
#include "Optimization/Matrix/Core/StaticMatrix.hpp"

#include "Optimization/Solver/KKTMonitor.hpp"
#include "Optimization/Solver/FallbackControl.hpp"

using namespace Optimization;
using namespace Optimization::Solver;

TEST(SolverTest, KKTMonitorEvaluateEQP) {
    // 3 variables, 2 constraints
    matrix::StaticMatrix<double, 3, 3> P; P.set_zero();
    P(0, 0) = 2.0; P(1, 1) = 2.0; P(2, 2) = 2.0;
    
    matrix::StaticVector<double, 3> q; q.set_zero();
    
    matrix::StaticMatrix<double, 2, 3> A; A.set_zero();
    A(0, 0) = 1.0; A(1, 1) = 1.0;
    
    matrix::StaticVector<double, 2> b; b.set_zero();
    b(0) = 1.0; b(1) = 2.0;
    
    // Optimal primal
    matrix::StaticVector<double, 3> u_opt; u_opt.set_zero();
    u_opt(0) = 1.0; u_opt(1) = 2.0; u_opt(2) = 0.0;
    
    // Optimal dual
    // grad_L = P*u + q + A^T * lambda = [2, 4, 0]^T + [lambda0, lambda1, 0]^T = 0
    // so lambda0 = -2, lambda1 = -4
    matrix::StaticVector<double, 2> lambda_opt; lambda_opt.set_zero();
    lambda_opt(0) = -2.0; lambda_opt(1) = -4.0;
    
    auto metrics = KKTMonitor<3, 2>::evaluate_EQP(P, q, A, b, u_opt, lambda_opt);
    EXPECT_TRUE(metrics.is_optimal);
    EXPECT_NEAR(metrics.stationarity_error, 0.0, 1e-6);
    EXPECT_NEAR(metrics.primal_feasibility_error, 0.0, 1e-6);
}

TEST(SolverTest, KKTMonitorEvaluateIPM) {
    // Simulate an IPM iteration
    // 3 variables, 2 equality, 4 inequality
    matrix::StaticVector<double, 3> grad_L_res; grad_L_res.set_zero();
    matrix::StaticVector<double, 2> primal_res; primal_res.set_zero();
    
    matrix::StaticVector<double, 4> s_ineq;
    s_ineq(0) = 0.1; s_ineq(1) = 0.2; s_ineq(2) = 0.3; s_ineq(3) = 0.4;
    
    matrix::StaticVector<double, 4> lambda_ineq;
    // target mu = 0.01
    // s * lambda = target_mu => lambda = target_mu / s
    lambda_ineq(0) = 0.1; lambda_ineq(1) = 0.05; lambda_ineq(2) = 0.033333333333; lambda_ineq(3) = 0.025;
    
    double target_mu = 0.01;
    
    auto metrics = KKTMonitor<3, 2>::template evaluate_IPM<4>(grad_L_res, primal_res, s_ineq, lambda_ineq, target_mu);
    EXPECT_TRUE(metrics.is_optimal);
    EXPECT_NEAR(metrics.complementarity_error, 0.0, 1e-6);
    EXPECT_NEAR(metrics.dual_feasibility_error, 0.0, 1e-6); // dual vars > 0, so error is 0
}

TEST(SolverTest, KKTMonitorEvaluateIPMSIMDFloat) {
    // Float type test for SIMD
    matrix::StaticVector<float, 12> s_ineq;
    matrix::StaticVector<float, 12> lambda_ineq;
    for(int i = 0; i < 12; ++i) {
        s_ineq(i) = 0.1f * (i + 1);
        lambda_ineq(i) = 0.01f / s_ineq(i);
    }
    
    float target_mu = 0.01f;
    float comp_err = KKTMonitor<1,1>::fast_complementarity_norm(s_ineq, lambda_ineq, target_mu);
    EXPECT_NEAR(comp_err, 0.0f, 1e-5f);
}

TEST(SolverTest, FallbackControlEvaluate) {
    KKTMonitorParams<double> params;
    params.slack_penalty_weight = 1000.0;
    params.slack_threshold = 0.1;
    params.comp_error_tol = 1e-3;
    params.dual_max_limits = 10.0;
    
    SolverKKTState<double> state;
    state.h_f = 0.05; state.h_r = 0.05;
    state.s_f = 0.02; state.s_r = 0.02;
    // W_slack * s = 1000 * 0.02 = 20.0 (stationarity maintained)
    state.mu_f = 20.0; state.mu_r = 20.0;
    
    auto trigger = evaluateKKTAndFallback(params, state);
    // mu = 20 > dual_max_limits(10), so it should fallback
    EXPECT_TRUE(trigger.is_fallback_required);
    
    // Normal State
    state.mu_f = 5.0; state.mu_r = 5.0;
    state.s_f = 0.005; state.s_r = 0.005;
    state.h_f = 0.005; state.h_r = 0.005;
    
    trigger = evaluateKKTAndFallback(params, state);
    EXPECT_FALSE(trigger.is_fallback_required);
}

TEST(SolverTest, FallbackControlStrategy) {
    FallbackTriggerState<double> trigger;
    ControlOutput<double> opt_u = {1.5, 0.05};
    double decel = -1.5;
    double pure_pursuit_steer = -0.1;
    
    trigger.is_fallback_required = true;
    auto final_u = applyFallbackStrategy(opt_u, trigger, decel, pure_pursuit_steer);
    EXPECT_DOUBLE_EQ(final_u.a_cmd, decel);
    EXPECT_DOUBLE_EQ(final_u.delta, pure_pursuit_steer);
    
    trigger.is_fallback_required = false;
    final_u = applyFallbackStrategy(opt_u, trigger, decel, pure_pursuit_steer);
    EXPECT_DOUBLE_EQ(final_u.a_cmd, opt_u.a_cmd);
    EXPECT_DOUBLE_EQ(final_u.delta, opt_u.delta);
}
