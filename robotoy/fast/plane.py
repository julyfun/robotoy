import numpy as np

def left_dir(v: np.ndarray):
    assert(v.shape == (2,))
    return np.array([-v[1], v[0]])

