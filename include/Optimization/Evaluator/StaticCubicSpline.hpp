#ifndef OPTIMIZATION_EVALUATOR_STATIC_CUBIC_SPLINE_HPP_
#define OPTIMIZATION_EVALUATOR_STATIC_CUBIC_SPLINE_HPP_

#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

namespace Optimization {
    namespace Evaluator {
        template <std::size_t MaxPoints>
        class StaticCubicSpline1D {
            public:
                void build(const std::array<double, MaxPoints>& x_in,
                           const std::array<double, MaxPoints>& y_in,
                           std::size_t n) {
                    num_points = n;
                    if (n < 3) {
                        return;
                    }

                    std::array<double, MaxPoints> h;
                    std::array<double, MaxPoints> alpha;

                    for (std::size_t i = 0; i < n - 1; ++i) {
                        h[i] = x_in[i + 1] - x_in[i];
                        a[i] = y_in[i];
                        x[i] = x_in[i];
                    }

                    a[n - 1] = y_in[n - 1];
                    x[n - 1] = x_in[n - 1];

                    for (std::size_t i = 1; i < n - 1; ++i) {
                        alpha[i] = 3.0 / h[i] * (a[i + 1] - a[i]) - 3.0 / h[i - 1] * (a[i] - a[i - 1]);
                    }

                    std::array<double, MaxPoints> l, mu, z;
                    l[0] = 1.0;
                    mu[0] = 0.0;
                    z[0] = 0.0;

                    for (std::size_t i = 1; i < n - 1; ++i) {
                        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
                        mu[i] = h[i] / l[i];
                        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
                    }

                    l[n - 1] = 1.0;
                    z[n - 1] = 0.0;
                    c[n - 1] = 0.0;

                    for (int i = n - 2; i >= 0; --i) {
                        c[i] = z[i] - mu[i] * c[i + 1];
                        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
                        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
                    }
                }

                double calc(double t) const {
                    if (num_points == 0) {
                        return 0.0;
                    }
                    std::size_t i = search_index(t);
                    double dx = t - x[i];
                    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
                }

                double calc_d1(double t) const {
                    if (num_points == 0) {
                        return 0.0;
                    }
                    std::size_t i = search_index(t);
                    double dx = t - x[i];
                    return b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx * dx;
                }

                double calc_d2(double t) const {
                    if (num_points == 0) {
                        return 0.0;
                    }
                    std::size_t i = search_index(t); // 오타 수정됨 (std::size_T -> std::size_t)
                    double dx = t - x[i];
                    return 2.0 * c[i] + 6.0 * d[i] * dx;
                }
            private:
                std::array<double, MaxPoints> a, b, c, d, x;
                std::size_t num_points = 0;
                
                std::size_t search_index(double t) const {
                    if (t <= x[0]) {
                        return 0;
                    }
                    if (t >= x[num_points - 1]) {
                        return num_points - 2;
                    }
                    int low = 0, high = num_points - 2;
                    while (low <= high) {
                        int mid = low + (high - low) / 2;
                        if (x[mid] <= t && t < x[mid + 1]) {
                            return mid;
                        }
                        if (t < x[mid]) {
                            high = mid - 1; // 오타 수정됨 (hig -> high)
                        } else {
                            low = mid + 1;
                        }
                    }
                    return 0;
                }
        };

        template <std::size_t MaxPoints>
        class StaticCubicSpline2D {
            public:
                StaticCubicSpline1D<MaxPoints> sx, sy;
                std::array<double, MaxPoints> s;
                std::size_t num_points = 0;

                void build(const std::array<double, MaxPoints>& x_in, const std::array<double, MaxPoints>& y_in, std::size_t n) {
                    num_points = n;
                    if (n == 0) {
                        return;
                    }
                    s[0] = 0.0;
                    for (std::size_t i = 1; i < n; ++i) {
                        double dx = x_in[i] - x_in[i - 1];
                        double dy = y_in[i] - y_in[i - 1];
                        s[i] = s[i - 1] + std::hypot(dx, dy);
                    }
                    sx.build(s, x_in, n);
                    sy.build(s, y_in, n);
                }

                double get_max_s() const {
                    return num_points > 0 ? s[num_points - 1] : 0.0;
                }

                double calc_x(double t) const {
                    return sx.calc(t);
                }
                double calc_y(double t) const {
                    return sy.calc(t);
                }
                double calc_yaw(double t) const {
                    return std::atan2(sy.calc_d1(t), sx.calc_d1(t));
                }

                double calc_curvature(double t) const {
                    double dx = sx.calc_d1(t);
                    double ddx = sx.calc_d2(t);
                    double dy = sy.calc_d1(t);
                    double ddy = sy.calc_d2(t);
                    
                    double denom = dx * dx + dy * dy; // 누락된 변수 계산 추가

                    if (denom < 1e-6) {
                        return 0.0;
                    }
                    double k = (ddy * dx - ddx * dy) / std::pow(denom, 1.5);
                    return k;
                }
        };
    } // namespace Evaluator
} // namespace Optimization

#endif // OPTIMIZATION_EVALUATOR_STATIC_CUBIC_SPLINE_HPP_