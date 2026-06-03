#ifndef OPTIMIZATION_SOLVER_MERIT_LINE_SEARCH_HPP_
#define OPTIMIZATION_SOLVER_MERIT_LINE_SEARCH_HPP_

namespace Optimization {
namespace solver {

/**
 * @brief Armijo 백트래킹 라인 서치 (Armijo Backtracking Line Search) 모듈
 *
 * NMPC(비선형 모델 예측 제어)나 기타 비선형 최적화 솔버에서 뉴턴 스텝(Newton Step) 방향으로
 * 얼만큼 나아갈지(Step Size, alpha)를 결정하는 역할을 합니다.
 * 비선형성이 강한 구간에서 100% (alpha = 1.0) 진행했다가 목적 함수가 오히려 악화되는
 * '폭주 현상(Overshoot)'을 막기 위한 필수적인 안전 장치입니다.
 *
 * [Architect's Update]
 * 루프 내부에 존재하던 불변식(Loop Invariant) 분기를 밖으로 끌어올리고(Hoisting),
 * 강등 연산(Strength Reduction)을 통해 곱셈 횟수를 최소화하여 극단적인 마이크로 최적화를
 * 달성했습니다.
 */
class MeritLineSearch {
   public:
    // 백트래킹 시 스텝 사이즈(alpha)를 줄여나가는 비율 (1.0 -> 0.5 -> 0.25 ...)
    static constexpr double BETA = 0.5;

    // Armijo 충분 감소(Sufficient Decrease) 조건의 완화 상수로, 아주 약간의 감소만으로도 스텝을
    // 허용하도록 설정함 보통 1e-4를 기본값으로 사용합니다.
    static constexpr double C1 = 1e-4;

    // 최대 백트래킹 횟수. (최소 alpha = 0.5^6 ≈ 0.015)
    static constexpr int MAX_ITER = 6;

    /**
     * @brief 라인 서치를 수행하여 최적의 스텝 사이즈(alpha)를 반환합니다.
     */
    template <typename Evaluator>
    [[nodiscard]] static inline double run(Evaluator evaluator, double current_merit,
                                           double directional_derivative = 0.0) {
        // [Architect's Update] 루프 외부로의 분기 및 상수 연산 끌어올리기 (Hoisting)
        // 방향 도함수가 양수(방향 설정 오류)인 경우 기대 감소량을 0.0으로 고정.
        // 루프 안에서 매번 검사하던 비효율을 원천 차단합니다.
        double current_expected_decrease =
            (directional_derivative >= 0.0) ? 0.0 : (C1 * directional_derivative);

        double alpha = 1.0;

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            // 후보 스텝 사이즈(alpha)만큼 전진했을 때의 시스템 상태(Merit)를 평가합니다.
            // (이 콜백 함수 내부에서 NMPC의 무거운 물리 시뮬레이션이 돌아갑니다)
            double candidate_merit = evaluator(alpha);

            // Armijo 충분 감소 조건 (Sufficient Decrease Condition)
            if (candidate_merit <= current_merit + current_expected_decrease) {
                return alpha;  // 스텝 승인: 탐색 종료
            }

            // 스텝 거절: 함수 값이 오히려 증가했거나 충분히 감소하지 않음
            // [Architect's Update] 곱셈 연산 강등 (Strength Reduction)
            // C1 * alpha * dir_deriv 를 다시 계산할 필요 없이, 기대 감소량도 스텝과 동일한 비율로
            // 줄입니다.
            alpha *= BETA;
            current_expected_decrease *= BETA;
        }

        // 극한의 비선형성 구간이라 스텝을 못 찾은 경우, 가장 작은 최소 스텝을 강제 반환하여
        // 시스템의 데드락(Stagnation)을 방지합니다.
        return alpha;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_MERIT_LINE_SEARCH_HPP_