#include "kalman.hpp"

#include <iostream>

int main() {
    using Kalman = kalman::LinearKalmanOneApi<2>;
    auto kalman = Kalman(1.0, 1.0, 100.0, 2.0);

    using namespace std;
    for (int i = 0; i < 10; ++i) {
        auto t = i * 0.1;
        Kalman::MatZ1 z = Kalman::MatZ1::Zero();
        z(0, 0) = 1.0 + i * 0.1;
        z(1, 0) = 2.0 + i * 0.2;
        auto x = kalman.smoothed(t, z);
        cout << "t: " << t << "\n\tz: " << z.transpose() << "\n\tx: " << x.transpose() << endl;
    }
}
