import numpy as np
from matplotlib import pyplot as plt
from fast_math.kalman import Kalman

class LinearKalman:
    def __init__(self, dim):
        # kalman.x: x, v, x, v, ...
        # H when dim = 3:
        # 1, 0, 0, 0, 0, 0
        # 0, 1, 0, 0, 0, 0
        # 0, 0, 1, 0, 0, 0
        self.dim = dim
        self.t = None
        self.kalman = Kalman(
            None, None, H=np.hstack([np.eye(dim), np.zeros((dim, dim))])
        )

    # order: time, data, variance
    def init(self, t, z):
        self.t = t
        self.kalman.x = np.hstack([z, np.zeros(self.dim)])
        self.kalman.P = np.zeros((self.dim * 2, self.dim * 2))
        for i in range(self.dim):
            self.kalman.P[i, i] = 1e6

    def predict(self, t, Q):
        F = np.eye(self.dim * 2)
        for i in range(self.dim):
            F[i, i + self.dim] = t - self.t
        self.kalman.predict(F, Q)
        self.t = t

    def update(self, z, R):
        self.kalman.update(z, R)

    # [untested]
    def predicted(self, t):
        F = np.eye(self.dim * 2)
        for i in range(self.dim):
            F[i, i + self.dim] = t - self.t
        return self.kalman.predicted(F)


class LinearKalmanOneApi:
    def __init__(self, dim, dead_time, Qx, Qv, Rx):
        self.linear_kalman = LinearKalman(dim)
        self.dead_time = dead_time
        self.Q = np.eye(dim * 2) * Qx
        for i in range(dim):
            self.Q[i + dim, i + dim] = Qv
        self.R = np.eye(dim) * Rx

    def smoothed(self, t, z):
        if self.linear_kalman.t is None or t - self.linear_kalman.t > self.dead_time:
            self.linear_kalman.init(t, z)
        self.linear_kalman.predict(t, self.Q)
        self.linear_kalman.update(z, self.R)
        return self.linear_kalman.kalman.x[: self.linear_kalman.dim]


def test_linear_kalman_one_api():
    dim = 3
    lkoa = LinearKalmanOneApi(3, 0.5, 0.2, 5, 5)
    times = np.arange(0, 2, 0.03)
    times = np.hstack([times[times < 0.3], times[times > 1.2]])

    measurements = np.tile(
        np.maximum(np.abs(times - 1) * 2, 1)[:, np.newaxis], (1, 3)
    ) + np.random.normal(0, 0.02, (len(times), dim))
    predictions = []

    for i, t in enumerate(times):
        predictions.append(lkoa.smoothed(t, measurements[i]))

    predictions = np.array(predictions)
    plt.figure(figsize=(12, 8))
    for i in range(dim):
        plt.subplot(dim, 1, i + 1)
        plt.plot(times, measurements[:, i], label="Measurements")
        plt.plot(times, predictions[:, i], label="Kalman Filter Prediction")
        plt.xlabel("Time")
        plt.ylabel(f"Dimension {i + 1}")
        plt.legend()
    plt.tight_layout()
    plt.show()


def test_linear_kalman():
    dim = 3
    lk = LinearKalman(dim)
    z = np.array([1.0, 2.0, 3.0])
    t = 0.0
    lk.init(t, z)
    Q = np.eye(dim * 2) * 0.2
    for i in range(dim):
        Q[i + dim, i + dim] = 100.0
    R = np.eye(dim) * 5.0  # 观测噪声协方差矩阵
    times = np.arange(0, 2, 0.03)
    measurements = np.tile(
        np.maximum(np.abs(times - 1) * 2, 1)[:, np.newaxis], (1, 3)
    ) + np.random.normal(0, 0.02, (len(times), dim))

    predictions = []

    for i, t in enumerate(times):
        lk.predict(t, Q)
        lk.update(measurements[i], R)
        predictions.append(lk.kalman.x[:dim])

    predictions = np.array(predictions)

    # 绘图
    plt.figure(figsize=(12, 8))
    for i in range(dim):
        plt.subplot(dim, 1, i + 1)
        plt.plot(times, measurements[:, i], label="Measurements")
        plt.plot(times, predictions[:, i], label="Kalman Filter Prediction")
        plt.xlabel("Time")
        plt.ylabel(f"Dimension {i + 1}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# [untested]
class LinearKalmanAliveApi:
    def __init__(self, dim, dead_time, Qx, Qv, Rx):
        self.linear_kalman = LinearKalman(dim)
        self.dead_time = dead_time
        self.Q = np.eye(dim * 2) * Qx
        for i in range(dim):
            self.Q[i + dim, i + dim] = Qv
        self.R = np.eye(dim) * Rx

    def update(self, t, z):
        if self.dead(t):
            self.linear_kalman.init(t, z)
        self.linear_kalman.predict(t, self.Q)
        self.linear_kalman.update(z, self.R)

    def get_pos(self, t):
        if self.dead(t):
            return None
        return self.linear_kalman.predicted(t)[:self.linear_kalman.dim]

    def get_state(self, t):
        if self.dead(t):
            return None
        return self.linear_kalman.predicted(t)

    def dead(self, t):
        return self.linear_kalman.t is None or t - self.linear_kalman.t > self.dead_time



# 运行测试
if __name__ == "__main__":
    # test_linear_kalman()
    test_linear_kalman_one_api()
