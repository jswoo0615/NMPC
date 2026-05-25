#ifndef OPTIMIZATION_MATRIX_STATIC_MATRIX_VIEW_HPP_
#define OPTIMIZATION_MATRIX_STATIC_MATRIX_VIEW_HPP_

#include <cassert>
#include <cstddef>
#include <iomanip>

namespace Optimization {
    template <typename T, std::size_t R, std::size_t C>
    class StaticMatrixView {
        private:
            T* data_ptr_;
            std::size_t stride_;

        public:
            using value_type = T;
            static constexpr std::size_t NumRows = R;
            static constexpr std::size_t NumCols = C;

            inline StaticMatrixView() noexcept : data_ptr_(nullptr), stride_(0) {}

            inline StaticMatrixView(T* ptr, std::size_t stride) noexcept : data_ptr_(ptr), stride_(stride) {
                assert(data_ptr_ != nullptr);
                assert(stride_ >= R);
            }

            inline T& operator()(std::size_t r, std::size_t c) noexcept {
                assert(r < R && c < C);
                return data_ptr_[c * stride_ + r];
            }
            inline const T& operator()(std::size_t r, std::size_t c) const noexcept {
                assert(r < R && c < C);
                return data_ptr_[c * stride_ + r];
            }
            inline T& operator()(std::size_t i) noexcept {
                assert(i < R * C);
                std::size_t c = i / R;
                std::size_t r = i % R;
                return data_ptr_[c * stride_ + r];
            }
            inline const T& operator()(std::size_t i) const noexcept {
                assert(i < R * C);
                std::size_t c = i / R;
                std::size_t r = i % R;
                return data_ptr_[c * stride_ + r];
            }

            inline T* data_ptr() noexcept {
                return data_ptr_;
            }
            inline const T* data_ptr() const noexcept {
                return data_ptr_;
            }
            inline std::size_t stride() const noexcept {
                return stride_;
            }
            inline StaticMatrixView& set_zero() noexcept {
                for (std::size_t j = 0; j < C; ++j) {
                    for (std::size_t i = 0; i < R; ++i) {
                        (*this)(i, j) = static_cast<T>(0);
                    }
                }
                return *this;
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_MATRIX_STATIC_MATRIX_VIEW_HPP_