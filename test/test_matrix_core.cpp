#include <gtest/gtest.h>
#include <cmath>

#include "Optimization/Matrix/Core/MathTraits.hpp"
#include "Optimization/Matrix/Core/StaticMatrix.hpp"
#include "Optimization/Matrix/Core/StaticMatrixView.hpp"
#include "Optimization/Utils/CUDAMacros.hpp"

using namespace Optimization;
using namespace Optimization::matrix;

// 1. Test CUDAMacros.hpp
CUDA_CALLABLE int dummy_cuda_callable_func(int a, int b) {
    return a + b;
}

TEST(MatrixCoreTest, CUDAMacrosTest) {
    EXPECT_EQ(dummy_cuda_callable_func(2, 3), 5);
}

// 2. Test MathTraits.hpp
TEST(MatrixCoreTest, MathTraitsTest) {
    // abs
    EXPECT_DOUBLE_EQ(MathTraits<double>::abs(-5.5), 5.5);
    EXPECT_DOUBLE_EQ(MathTraits<double>::abs(5.5), 5.5);
    
    // sqrt
    EXPECT_DOUBLE_EQ(MathTraits<double>::sqrt(16.0), 4.0);
    
    // max / min
    EXPECT_DOUBLE_EQ(MathTraits<double>::max(3.0, 7.0), 7.0);
    EXPECT_DOUBLE_EQ(MathTraits<double>::min(3.0, 7.0), 3.0);
    
    // near_zero
    EXPECT_TRUE(MathTraits<double>::near_zero(1e-17));
    EXPECT_TRUE(MathTraits<double>::near_zero(-1e-17));
    EXPECT_FALSE(MathTraits<double>::near_zero(1e-10)); // Machine epsilon for double is ~2e-16
    
    // get_value
    EXPECT_DOUBLE_EQ(MathTraits<double>::get_value(42.0), 42.0);
}

// 3. Test StaticMatrix.hpp
TEST(MatrixCoreTest, StaticMatrixInitAndAccess) {
    StaticMatrix<double, 3, 2> mat;
    
    // Default constructor should initialize to zero
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(mat(i), 0.0);
    }
    
    // Write access
    mat(0, 1) = 5.0;
    mat(2, 0) = -3.5;
    
    // Read access
    EXPECT_DOUBLE_EQ(mat(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(mat(2, 0), -3.5);
    
    // Linear access (column-major format check)
    // 3x2 matrix: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
    EXPECT_DOUBLE_EQ(mat(2), -3.5); // (2, 0)
    EXPECT_DOUBLE_EQ(mat(3), 5.0);  // (0, 1)
    
    // Copy construction
    StaticMatrix<double, 3, 2> mat_copy(mat);
    EXPECT_DOUBLE_EQ(mat_copy(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(mat_copy(2, 0), -3.5);
    
    // Assignment operator
    StaticMatrix<double, 3, 2> mat_assign;
    mat_assign = mat;
    EXPECT_DOUBLE_EQ(mat_assign(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(mat_assign(2, 0), -3.5);
}

TEST(MatrixCoreTest, StaticMatrixOperations) {
    StaticMatrix<double, 2, 2> mat1;
    mat1(0, 0) = 1.0; mat1(1, 0) = 2.0;
    mat1(0, 1) = 3.0; mat1(1, 1) = 4.0;
    
    StaticMatrix<double, 2, 2> mat2;
    mat2(0, 0) = 5.0; mat2(1, 0) = 6.0;
    mat2(0, 1) = 7.0; mat2(1, 1) = 8.0;
    
    // operator+=
    mat1 += mat2;
    EXPECT_DOUBLE_EQ(mat1(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(mat1(1, 1), 12.0);
    
    // operator-=
    mat1 -= mat2;
    EXPECT_DOUBLE_EQ(mat1(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat1(1, 1), 4.0);
    
    // operator*= (scalar)
    mat1 *= 2.0;
    EXPECT_DOUBLE_EQ(mat1(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(mat1(1, 1), 8.0);
    
    // saxpy: mat1 += 0.5 * mat2
    mat1.saxpy(0.5, mat2);
    EXPECT_DOUBLE_EQ(mat1(0, 0), 2.0 + 2.5);
    EXPECT_DOUBLE_EQ(mat1(1, 1), 8.0 + 4.0);
    
    // Method chaining check
    mat1.set_zero().saxpy(1.0, mat2);
    EXPECT_DOUBLE_EQ(mat1(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mat1(1, 1), 8.0);
}

// 4. Test StaticMatrixView.hpp
TEST(MatrixCoreTest, StaticMatrixViewTest) {
    double raw_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 3x2 matrix in col-major
    
    // Create a 2x2 view, stride = 3 (taking the top 2x2 part of the 3x2 matrix)
    StaticMatrixView<double, 2, 2> view(raw_data, 3);
    
    // Access elements
    EXPECT_DOUBLE_EQ(view(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(view(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(view(1, 1), 5.0);
    
    // Modify via view
    view(0, 0) = 10.0;
    EXPECT_DOUBLE_EQ(raw_data[0], 10.0);
    
    // Set zero
    view.set_zero();
    EXPECT_DOUBLE_EQ(raw_data[0], 0.0);
    EXPECT_DOUBLE_EQ(raw_data[1], 0.0);
    EXPECT_DOUBLE_EQ(raw_data[2], 3.0); // Should remain untouched (out of view bounds)
    EXPECT_DOUBLE_EQ(raw_data[3], 0.0);
    EXPECT_DOUBLE_EQ(raw_data[4], 0.0);
    EXPECT_DOUBLE_EQ(raw_data[5], 6.0); // Should remain untouched
}
