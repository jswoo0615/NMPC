#ifndef OPTIMIZATION_ESTIMATOR_HISTORY_BUFFER_HPP_
#define OPTIMIZATION_ESTIMATOR_HISTORY_BUFFER_HPP_

#include <array>
#include <cassert>
#include <cstddef>

#include "Optimization/Matrix/Core/StaticMatrix.hpp"

namespace Optimization {
namespace estimator {

/**
 * @brief 과거의 상태나 센서 데이터 등을 저장하는 원형 큐(Circular Buffer)
 * * [Architect's Update]
 * - 임베디드 타겟(Jetson Nano)의 ALU 병목을 제거하기 위해 모듈러(%) 연산을 전면 폐기하고
 * Branchless(CSEL) 인덱스 계산 기법을 도입했습니다.
 * - std::array와 StaticMatrix의 alignas(64) 특성을 활용하여
 * O(1) SIMD 고속 복사(Zero-Allocation)를 보장합니다.
 *
 * @tparam Dim 저장할 데이터 벡터의 차원 (예: 상태 변수가 6개면 6)
 * @tparam Capacity 버퍼에 저장할 수 있는 최대 데이터 개수 (히스토리 길이)
 */
template <size_t Dim, size_t Capacity>
class HistoryBuffer {
   private:
    // 고정 크기의 정적 벡터 배열 (SIMD 정렬 보장)
    std::array<matrix::StaticVector<double, Dim>, Capacity> buffer_;

    // 가장 최근에 추가된 데이터의 위치(인덱스)를 가리키는 포인터
    size_t head_;

    // 현재 버퍼에 저장된 유효한 데이터의 총 개수
    size_t count_;

   public:
    /**
     * @brief 기본 생성자: 메모리 초기화
     */
    HistoryBuffer() : head_(0), count_(0) {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].set_zero();
        }
    }

    /**
     * @brief 버퍼에 새로운 데이터를 추가합니다 (Push)
     * @details 모듈러 연산(%) 패널티를 제거하고 조건부 덧셈으로 최적화
     * @param data 저장할 새로운 데이터 벡터
     */
    inline void push(const matrix::StaticVector<double, Dim>& data) {
        // [Architect's Update] % 연산 폐기 -> 1-Cycle 연산
        head_ = (head_ + 1 == Capacity) ? 0 : head_ + 1;

        // StaticMatrix의 암시적 복사 대입 연산자 호출 (SIMD Load/Store 가동)
        buffer_[head_] = data;

        if (count_ < Capacity) count_++;
    }

    /**
     * @brief 최신 데이터로부터 과거로 'age' 만큼 떨어진 데이터를 조회합니다 (과거 탐색)
     * @details 모듈러 연산(%) 패널티를 제거하고 조건부 이동 명령어(CSEL) 유도
     * @param age 조회할 데이터의 나이 (0이면 가장 최신 데이터, 1이면 바로 이전 데이터)
     * @return const matrix::StaticVector<double, Dim>& 요청한 과거 시점의 데이터 벡터
     */
    inline const matrix::StaticVector<double, Dim>& operator[](size_t age) const {
        // 논리적 치명 오류 방어선 (Release 빌드에서는 오버헤드 0)
        assert(age < Capacity && "HistoryBuffer: age exceeds buffer capacity.");
        assert(age < count_ && "HistoryBuffer: attempting to read uninitialized memory.");

        // [Architect's Update] % 연산 폐기 -> 분기 예측 실패가 없는 조건부 연산
        size_t idx = (head_ >= age) ? (head_ - age) : (head_ + Capacity - age);

        return buffer_[idx];
    }

    /**
     * @brief 현재 버퍼에 저장된 유효한 데이터의 개수를 반환합니다.
     */
    inline size_t size() const { return count_; }

    /**
     * @brief 버퍼의 최대 용량을 반환합니다.
     */
    inline constexpr size_t capacity() const { return Capacity; }

    /**
     * @brief 버퍼를 초기 상태로 비웁니다 (Clear).
     */
    inline void clear() {
        count_ = 0;
        head_ = 0;
    }
};

}  // namespace estimator
}  // namespace Optimization

#endif  // OPTIMIZATION_ESTIMATOR_HISTORY_BUFFER_HPP_