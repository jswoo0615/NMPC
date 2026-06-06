#include <iostream>
#include <cmath>
#include "Optimization/Control/SparseNMPC_IPM.hpp"
#include "Optimization/Dynamics/RealTimeDynamicsModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;
using namespace Optimization::Dynamics;

int main() {
    NMPCTuningConfig config;
    config.target_vx = 10.0;
    for(int i=0; i<100; ++i) config.target_d[i] = 0.0;
    
    SparseNMPC_IPM<30, RealTimeDynamicsModel, 8, 2> nmpc;
    matrix::StaticVector<double, 8> x0;
    x0(0) = 0.0; x0(1) = 0.0; x0(2) = 0.0; x0(3) = 20.0;
    x0(4) = 0.0; x0(5) = 0.0; x0(6) = 0.0; x0(7) = 0.0;
    
    auto result = nmpc.solve_ipm(x0, config);
    std::cout << "Final X_pred[0](3): " << nmpc.X_pred[0](3) << std::endl;
    
    return 0;
}
