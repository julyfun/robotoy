#include "kalman.hpp"

#include <iostream>

int main() {
    using Kalman = kalman::LinearKalmanOneApi<2>;
    // 四个参数:
    // dead_time, 多久不更新后下次会重置内部状态
    // Qx, 位置噪声协方差的每一项，越小越信任预测位置
    // Qv, 速度噪声协方差的每一项，越小越信任预测速度
    // Rx, 观测噪声协方差的每一项，越小越信任观测值，也就导致收敛快、抖动大
    auto kalman = Kalman(1.0, 1.0, 100.0, 2.0);

    using namespace std;
    for (int i = 0; i < 10; ++i) {
        auto t = i * 0.1;
        Kalman::MatZ1 z = Kalman::MatZ1::Zero();
        z(0, 0) = 1.0 + i * 0.1;
        z(1, 0) = 2.0 + i * 0.2;
        // 把当前时间戳和观测值传入，立刻得到当前时刻平滑后的观测值.
        auto x = kalman.smoothed(t, z);
        cout << "t: " << t << "\n\tz: " << z.transpose() << "\n\tx: " << x.transpose() << endl;
    }
}
