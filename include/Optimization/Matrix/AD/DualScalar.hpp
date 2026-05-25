#ifndef OPTIMIZATION_MATRIX_DUALSCALAR_HPP_
#define OPTIMIZATION_MATRIX_DUALSCALAR_HPP_

#include <cmath>

namespace Optimization {
    namespace matrix {
        template <typename T>
        struct Dual {
            T v;
            T d;

            inline Dual() noexcept : v(0), d(0) {}
            inline Dual(T val) noexcept : v(val), d(0) {}
            inline Dual(T val, T der) noexcept : v(val), d(der) {}

            inline Dual& operator=(T val) noexcept {
                v = val;
                d = 0;
                return *this;
            }

            inline Dual operator-() const noexcept {
                return Dual(-v, -d);
            }
            inline Dual operator+(const Dual& rhs) const noexcept {
                return Dual(v + rhs.v, d + rhs.d);
            }
            inline Dual operator-(const Dual& rhs) const noexcept {
                return Dual(v - rhs.v, d - rhs.d);
            }
            inline Dual operator*(const Dual& rhs) const noexcept {
                return Dual(v * rhs.v, d * rhs.v + v * rhs.d);
            }
            inline Dual operator/(const Dual& rhs) const noexcept {
                return Dual(v / rhs.v, (d * rhs.v - v * rhs.d) / (rhs.v * rhs.v));
            }

            inline Dual operator+(T s) const noexcept {
                return Dual(v + s, d);
            }
            inline Dual operator-(T s) const noexcept {
                return Dual(v - s, d);
            }
            inline Dual operator*(T s) const noexcept {
                return Dual(v * s, d * s);
            }
            inline Dual operator/(T s) const noexcept {
                return Dual(v / s, d / s);
            }

            inline Dual& operator+=(const Dual& rhs) noexcept {
                *this = *this + rhs;
                return *this;
            }
            inline Dual& operator-=(const Dual& rhs) noexcept {
                *this = *this - rhs;
                return *this;
            }
            inline Dual& operator*=(const Dual& rhs) noexcept {
                *this = *this * rhs;
                return *this;
            }
            inline Dual& operator/=(const Dual& rhs) noexcept {
                *this = *this / rhs;
                return *this;
            }
        };

        template <typename T> 
        inline Dual<T> operator+(T s, const Dual<T>& d) noexcept {
            return Dual<T>(s + d.v, d.d);
        }
        template <typename T>
        inline Dual<T> operator-(T s, const Dual<T>& d) noexcept {
            return Dual<T>(s - d.v, -d.d);
        }
        template <typename T>
        inline Dual<T> operator*(T s, const Dual<T>& d) noexcept {
            return Dual<T>(s * d.v, s * d.d);
        }
        template <typename T>
        inline Dual<T> operator/(T s, const Dual<T>& d) noexcept {
            return Dual<T>(s / d.v, (-s * d.d) / (d.v * d.v));
        }
    } // namespace matrix
} // namespace Optimization

#endif // OPTIMIZATION_MATRIX_DUALSCALAR_HPP_