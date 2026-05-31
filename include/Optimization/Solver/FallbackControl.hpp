#ifndef OPTIMIZATION_SOLVER_FALLBACK_CONTROL_HPP_
#define OPTIMIZATION_SOLVER_FALLBACK_CONTROL_HPP_

#include <algorithm>
#include <cmath>

namespace Optimization {
namespace Solver {

// =========================================================================
// Fallback Control & Robustness Monitoring Mechanisms
// =========================================================================

// KKT 모니터링 및 Fallback 파라미터 구조체
template <typename T>
struct KKTMonitorParams {
    T slack_penalty_weight;         // W_slack
    T slack_threshold;              // 물리적 한계 초과 판별 기준치 (S_thresh)
    T comp_error_tol;               // 상보적 여유성 오차 허용치
    T dual_max_limits;              // 쌍대 변수 폭발 기준치
};

// 솔버의 현재 Step KKT 상태 입력용 구조체
template <typename T>
struct SolverKKTState {
    T h_f;                          // 전륜 마찰원 잔차 h_f(x, u)
    T h_r;                          // 후륜 마찰원 잔차 h_r(x, u)
    T s_f;                          // 전륜 여유 변수 (Slack)
    T s_r;                          // 후륜 여유 변수 (Slack)
    T mu_f;                         // 전륜 부등식 제약 쌍대 변수 (Dual multiplier)
    T mu_r;                         // 후륜 부등식 제약 쌍대 변수 (Dual multiplier)
};

// KKT 검증 결과 및 Fallback 트리거 상태
template <typename T>
struct FallbackTriggerState {
    bool is_fallback_required;
    T physical_violation_max;
    T comp_slackness_error_max;
    T dual_variable_max;
};

// KKT Monitor 진단 엔진
template <typename T>
inline FallbackTriggerState<T> evaluateKKTAndFallback(
    const KKTMonitorParams<T>& params,
    const SolverKKTState<T>& state
) {
    FallbackTriggerState<T> result;
    result.is_fallback_required = false;

    // 1. 물리적 마찰 한계 초과 검증 (Slack variable magnitude)
    result.physical_violation_max = std::max(state.s_f, state.s_r);
    if (result.physical_violation_max > params.slack_threshold) {
        result.is_fallback_required = true;
    }

    // 2. 상보적 여유성 오차 (Complementary Slackness Error) 검증
    // E_comp = |mu * (h - s)|
    T comp_err_f = std::abs(state.mu_f * (state.h_f - state.s_f));
    T comp_err_r = std::abs(state.mu_r * (state.h_r - state.s_r));
    result.comp_slackness_error_max = std::max(comp_err_f, comp_err_r);

    if (result.comp_slackness_error_max > params.comp_error_tol) {
        result.is_fallback_required = true;
    }

    // 3. 쌍대 변수 폭발 (Dual Explosion) 및 정류성 오차 검증
    result.dual_variable_max = std::max(std::abs(state.mu_f), std::abs(state.mu_r));

    if (result.dual_variable_max > params.dual_max_limits) {
        result.is_fallback_required = true;
    }

    // 추가 정류성 붕괴 체크
    T stat_err_f = std::abs(state.mu_f - (params.slack_penalty_weight * state.s_f));
    T stat_err_r = std::abs(state.mu_r - (params.slack_penalty_weight * state.s_r));
    if (std::max(stat_err_f, stat_err_r) > params.dual_max_limits * static_cast<T>(0.5)) {
        result.is_fallback_required = true;
    }

    return result;
}

// Fallback Controller 적용기
template <typename T>
struct ControlOutput {
    T a_cmd;            // 종방향 제어 명령 (m/s^2)
    T delta;            // 전륜 조향 제어 명령 (rad)
};

template <typename T>
inline ControlOutput<T> applyFallbackStrategy(
    const ControlOutput<T>& nmpc_optimal_u,
    const FallbackTriggerState<T>& trigger_state,
    const T safe_deceleration_cmd,              // 예 : -1.0 m/s^2
    const T pure_pursuit_delta                  // Fallback용 LQR/Pure Pursuit 조향각
) {
    if (trigger_state.is_fallback_required) {
        ControlOutput<T> fallback_u;
        fallback_u.a_cmd = safe_deceleration_cmd;
        fallback_u.delta = pure_pursuit_delta;
        return fallback_u;
    }
    return nmpc_optimal_u;
}

}  // namespace Solver
}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_FALLBACK_CONTROL_HPP_
