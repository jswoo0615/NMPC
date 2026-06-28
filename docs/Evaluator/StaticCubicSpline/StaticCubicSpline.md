## StaticCubicSpline.hpp

본 문서는 NMPC 참조 경로 생성 및 평가를 위한 정적 메모리 기반 3차 스플라인 보간기 클래스의 명세입니다. 힙 (Heap) 동적 할당을 배제하고 `std::array`를 활용하여 실시간 최적화 솔버의 제약조건 및 비용 함수 평가 시 요구되는 결정론적 실행 시간과 메모리 안전성을 보장하도록 설계되었습니다.

### 1. 수학적 배경 (Mathematical Formulation)
1차원 3차 스플라인은 각 구간 $[x_{i}, x_{i+1}]$에 대해 다음과 같은 다항식으로 정의됩니다.

$$S(t) = a_{i} + b_{i}(t - x_{i}) + c_{i}(t - x_{i})^{2} + d_{i}(t - x_{i})^{3}$$

2차원 매개변수 스플라인은 누적 거리 (Arc length) $s$를 매개변수로 하여 $x(s)$와 $y(s)$를 각각 독립적인 1D 스플라인으로 모델링합니다. 이를 기반으로 한 곡률 (Curvature) $\kappa$는 다음과 같이 산출됩니다.

$$\kappa = \frac{x'y''-x''y'}{(x'^{2} + y'^{2})^{3/2}}$$

### 2. `Optimization::Evaluator::StaticCubicSpline1D<MaxPoints>`
단일 변수에 대한 3차 스플라인 계수 계산 및 보간을 수행하는 클래스입니다.

#### 주요 멤버 변수
- `std::array<double, MaxPoints> a, b, c, d` : 스플라인 다항식의 각 항 계수
- `std::array<double, MaxPoints> x` : 기준점의 독립 변수 배열
- `std::size_t num_points` : 실제 유효한 데이터 포인트의 수

#### 멤버 함수 명세
|함수명|시그니처|기능 설명|
|:---|:---|:---|
|`build`|`void build(const std::array<double, MaxPoints>& x_in, const std::array<double, MaxPoints>& y_in, std::size_t n)`|입력된 점들을 바탕으로 선형 방정식을 풀어 스플라인 계수를 계산합니다. 유효성 검증을 위해 최소 3개의 점이 요구됩니다.|
|`calc`|`double calc(double t) const`|지정된 지점 $t$에서의 스플라인 보간값 $S(t)$를 반환합니다.|
|`calc_d1`|`double calc_d1(double t) const`|지정된 지점 $t$에서의 1차 미분값 $S'(t)$를 반환합니다.|
|`calc_d2`|`double calc_d2(double t) const`|지정된 지점 $t$에서의 2차 미분값 $S''(t)$를 반환합니다.|
|`search_index`|`std::size_t search_index(double t) const`|(`private`) 입력값 $t$가 속한 다항식 구간의 인덱스를 이진 탐색으로 도출합니다. $O(\log n)$의 탐색 복잡도를 가집니다.|

### 3. `Optimization::Evaluator::StaticCubicSpline2D<MaxPoints>`
누적 거리 $s$를 매개변수로 사용하는 2차원 경로 스플라인 보간기입니다. 차량 동역학 기반 제어 시 궤적 추종을 위한 연속적인 기준 상태 (Reference State) 생성에 사용됩니다.

#### 주요 멤버 변수
- `StaticCubicSpline1D<MaxPoints> sx, sy` : X 좌표와 Y 좌표를 매개변수 $s$에 대해 각각 보간하는 내부 1D 스플라인 객체
- `std::array<double, MaxPoints> s` : 계산된 누적 거리 (Arc Length) 배열
- `std::size_t num_points` : 실제 유효한 데이터 포인트의 수

#### 멤버 함수 명세
|함수명|시그니처|기능 설명|
|:---|:---|:---|
|`build`|`void build(const std::array<double, MaxPoints>& x_in, const std::array<double, MaxPoints>& y_in, std::size_t n)`|인접한 두 점 사이의 유클리드 거리를 누적하여 매개변수 $s$ 배열을 생성하고, 이를 바탕으로 `sx`와 `sy` 스플라인을 초기화합니다|
|`get_max_s`|`double get_max_s() const`|생성된 궤적의 전체 누적 길이 (최대 $s$ 값)를 반환합니다. 종단 조건 판별에 활용됩니다|
|`calc_x`|`double calc_x(double t)`|누적 거리 $t$에서의 보간된 X 좌표를 반환합니다.|
|`calc_y`|`double calc_y(double t)`|누적 거리 $t$에서의 보간된 Y 좌표를 반환합니다.|
|`calc_yaw`|`double calc_yaw(double t)`|누적 거리 $t$에서의 궤적 헤딩 (Yaw)을 $atan2(y', x')$ 연산으로 계산하여 반환합니다.|
|`calc_curvature`|`double calc_curvature(double t) const`|누적 거리 $t$에서의 곡률 $\kappa$를 계산합니다. 연산 안정성을 위해 분모가 `1e-6` 미만일 경우 0.0으로 클램핑합니다.|