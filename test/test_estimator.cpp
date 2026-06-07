#include <gtest/gtest.h>
#include "Optimization/Estimator/EKF.hpp"
#include "Optimization/Estimator/SparseMHE.hpp"
#include "Optimization/Estimator/HistoryBuffer.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"

using namespace Optimization;
using namespace Optimization::estimator;

TEST(EstimatorTest, Initialization) {
    // Test EKF
    EKF<8, 2> ekf;
    EXPECT_EQ(ekf.P(0, 0), 1.0);

    // Test SparseMHE
    SparseMHE<10, Dynamics::RealTimeDynamicsModel, 8, 2, 8> mhe;
    EXPECT_EQ(mhe.W_dyn(0, 0), 1.0);

    // Test HistoryBuffer
    HistoryBuffer<8, 10> history;
    EXPECT_EQ(history.capacity(), 10);
    EXPECT_EQ(history.size(), 0);
}
