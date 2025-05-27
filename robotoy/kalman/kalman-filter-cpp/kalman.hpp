//! Don't omit `this->`
#ifndef KALMAN_FILTER_CPP_KALMAN_HPP
#define KALMAN_FILTER_CPP_KALMAN_HPP

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>

namespace kalman_filter_cpp::kalman {

// Pass dimention to template param
template<int n_x, int n_z>
struct Kalman {
public:
    using MatX1 = Eigen::Matrix<double, n_x, 1>;
    using MatXX = Eigen::Matrix<double, n_x, n_x>;
    using MatXZ = Eigen::Matrix<double, n_x, n_z>;
    using MatZ1 = Eigen::Matrix<double, n_z, 1>;
    using MatZX = Eigen::Matrix<double, n_z, n_x>;
    using MatZZ = Eigen::Matrix<double, n_z, n_z>;

    Kalman(MatX1 x, MatXX P, MatZX H) {
        this->x = x;
        this->P = P;
        this->H = H;
    }

    void predict(MatXX F, MatXX Q) {
        this->x = F * this->x;
        this->P = F * this->P * F.transpose() + Q;
    }

    void update(MatZ1 z, MatZZ R) {
        MatZ1 y = z - this->H * this->x;
        MatZZ S = this->H * this->P * this->H.transpose() + R;
        MatXZ K = this->P * this->H.transpose() * S.inverse();
        this->x = this->x + K * y;
        this->P = this->P - K * this->H * this->P;
    }

    MatX1 predicted(MatXX F) {
        return F * this->x;
    }

public:
    MatX1 x;
    MatXX P;

private:
    MatZX H;
};

// State in the form of [x, y, z, vx, vy, vz..]
template<int dim>
struct LinearKalman {
public:
    static constexpr int n_x = dim * 2;
    static constexpr int n_z = dim;
    using MatX1 = Eigen::Matrix<double, n_x, 1>;
    using MatXX = Eigen::Matrix<double, n_x, n_x>;
    using MatZ1 = Eigen::Matrix<double, n_z, 1>;
    using MatZX = Eigen::Matrix<double, n_z, n_x>;
    using MatZZ = Eigen::Matrix<double, n_z, n_z>;

    explicit LinearKalman():
        t(0),
        is_t_set(false),
        kalman(MatX1::Zero(), MatXX::Zero(), []() {
            MatZX H = MatZX::Zero();
            for (int i = 0; i < n_z; ++i) {
                H(i, i) = 1;
            }
            return H;
        }()) {}

    void init(double t, MatZ1 z) {
        this->t = t;
        this->is_t_set = true;
        this->kalman.x = MatX1::Zero();
        this->kalman.P = MatXX::Zero();
        for (int i = 0; i < dim; ++i) {
            this->kalman.x(i) = z(i);
            this->kalman.P(i, i) = 1e6;
        }
        for (int i = dim; i < n_x; ++i) {
            this->kalman.P(i, i) = 1e3;
        }
    }

    void predict(double t, MatXX Q) {
        if (!is_t_set) {
            std::cerr << "@kalman::LinearKalman::predict: Kalman filter not initialized."
                      << std::endl;
            return;
        }
        MatXX F = MatXX::Identity();
        for (int i = 0; i < dim; ++i) {
            F(i, i + dim) = t - this->t;
        }
        this->kalman.predict(F, Q);
        this->t = t;
    }

    void update(MatZ1 z, MatZZ R) {
        if (!is_t_set) {
            std::cerr << "@kalman::LinearKalman::update: Kalman filter not initialized."
                      << std::endl;
            return;
        }
        this->kalman.update(z, R);
    }

public:
    double t;
    bool is_t_set; // for C++14 usage, don't use std::optional
    Kalman<n_x, n_z> kalman;
};

template<int dim>
struct LinearKalmanOneApi {
public:
    static constexpr int n_x = dim * 2;
    static constexpr int n_z = dim;
    using MatX1 = Eigen::Matrix<double, n_x, 1>;
    using MatXX = Eigen::Matrix<double, n_x, n_x>;
    using MatZ1 = Eigen::Matrix<double, n_z, 1>;
    using MatZZ = Eigen::Matrix<double, n_z, n_z>;

    LinearKalmanOneApi(double dead_time, double Qx, double Qv, double Rx):
        linear_kalman(),
        dead_time(dead_time),
        Q(MatXX::Identity() * Qx),
        R(MatZZ::Identity() * Rx) {
        for (int i = 0; i < dim; ++i) {
            this->Q(i + dim, i + dim) = Qv;
        }
    }

    MatZ1 smoothed(double t, MatZ1 z) {
        if (!this->linear_kalman.is_t_set || t - this->linear_kalman.t > this->dead_time) {
            this->linear_kalman.init(t, z);
        }
        // [TODO] else preict
        this->linear_kalman.predict(t, Q);
        this->linear_kalman.update(z, R);
        // wtf is .template?
        return this->linear_kalman.kalman.x.template head<dim>();
    }

public:
    LinearKalman<dim> linear_kalman;
    double dead_time;
    MatXX Q;
    MatZZ R;
};

} // namespace kalman_filter_cpp::kalman

#endif
