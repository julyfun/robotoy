import numpy as np

class Kalman:
    def __init__(self, x: np.ndarray, P: np.ndarray, H: np.ndarray):
        self.x = x  # [state * 1]
        self.P = P  # [state * state]
        self.H = H  # [obs * state]

    def predict(self, F: np.ndarray, Q: np.ndarray):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, R: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

    # [untested]
    def predicted(self, F):
        return F @ self.x
