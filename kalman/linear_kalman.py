import numpy as np
from matplotlib import pyplot as plt


class Kalman:
    def __init__(self, x, P, H):
        self.x = x  # [state * 1]
        self.P = P  # [state * state]
        self.H = H  # [obs * state]

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P


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


class LinearKalmanOneApi:
    def __init__(self, dim, dead_time, Q, R):
        self.linear_kalman = LinearKalman(dim)
        self.dead_time = dead_time
        self.Q = Q
        self.R = R

    def smoothed(self, t, z):
        if self.linear_kalman.t is None or t - self.linear_kalman.t > self.dead_time:
            self.linear_kalman.init(t, z)
        self.linear_kalman.predict(t, self.Q)
        self.linear_kalman.update(z, self.R)
        return self.linear_kalman.kalman.x[: self.linear_kalman.dim]


def test_linear_kalman_one_api():
    dim = 3
    dead_time = 0.1
    R = np.eye(dim) * 5.0
    Q = np.eye(dim * 2) * 0.2
    for i in range(dim):
        Q[i + dim, i + dim] = 100.0
    lkoa = LinearKalmanOneApi(dim, dead_time, Q, R)
    times = np.arange(0, 2, 0.03)
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


# 运行测试
if __name__ == "__main__":
    # test_linear_kalman()
    test_linear_kalman_one_api()
