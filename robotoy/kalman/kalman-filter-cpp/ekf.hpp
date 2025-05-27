#ifndef KALMAN_FILTER_CPP_EKF_HPP
#define KALMAN_FILTER_CPP_EKF_HPP

#include <Eigen/Dense>
#include <ceres/jet.h>

namespace kalman_filter_cpp::ekf {

template<int n_x, int n_z>
struct Ekf {
public:
    using MatX1 = Eigen::Matrix<double, n_x, 1>;
    using MatXX = Eigen::Matrix<double, n_x, n_x>;
    using MatXZ = Eigen::Matrix<double, n_x, n_z>;
    using MatZ1 = Eigen::Matrix<double, n_z, 1>;
    using MatZX = Eigen::Matrix<double, n_z, n_x>;
    using MatZZ = Eigen::Matrix<double, n_z, n_z>;

    // 可能多种观测量.
    Ekf(MatX1 x, MatXX P): x(x), P(P) {}

private:
    MatX1 x;
    MatXX P;
};

} // namespace kalman_filter_cpp::ekf

#endif
