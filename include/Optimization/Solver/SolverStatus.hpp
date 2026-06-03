#ifndef OPTIMIZATION_SOLVER_STATUS_HPP_
#define OPTIMIZATION_SOLVER_STATUS_HPP_

namespace Optimization {
    /**
     * @brief 실시간 최적화 솔버 상태 진단 코드 (Solver Status)
     * @details
     * 수학적 연산의 결과를 자율주행 차량의 상태 머신 (State Machine)롸
     * 비상 제어 (Fallback Strategy) 로직에 직결시키는 절대 규약 (Absolute Rule)입니다
     */
    enum class SolverStatus {
        // ========================================================================
        // 🟢 사용 가능 상태 (Safe to Control)
        // ========================================================================
        SUCCESS = 1,  // [완전 수렴] KKT 허용 오차(Tolerance) 내에 완벽히 도달함.
        SUBOPTIMAL = 2,  // [부분 수렴] 완벽하진 않으나, 제어 입력으로 쓰기에 무리가 없는 해.
        // (RTI 기법이나 제한 시간 초과 시 이전 해를 바탕으로 제어할 때 발생)

        // ========================================================================
        // 🟡 경고 및 성능 저하 (Warning / Degraded - 이전 제어값 유지 권장)
        // ========================================================================
        MAX_ITERATION_REACHED = -1,  // [반복 초과] 최대 반복 횟수 내에 KKT 조건을 만족하지 못함.
        TIME_LIMIT_REACHED = -2,  // [데드라인 초과] 실시간 제약 시간을 초과하여 연산 강제 종료.
        STEP_SIZE_TOO_SMALL = -3,  // [탐색 불가] 라인 서치 등에서 더 이상 해를 개선할 수 없음.

        // ========================================================================
        // 🔴 치명적 오류 (Emergency / Fallback Trigger)
        // ========================================================================
        MATH_ERROR = -10,  // [수치 붕괴] LDLT/LU 등 내부 엔진에서 역행렬 불가(Singular) 및 NaN 발생.
        INFEASIBLE = -11  // [실현 불가] 물리적 제약 조건(장애물, 조향 한계)을 회피할 공간이 없음.
    };
} // namespace Optimization

#endif // OPTIMIZATION_SOLVER_STATUS_HPP_