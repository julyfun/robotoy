from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

XyzXyzw = Tuple[np.ndarray, np.ndarray]

# don't np.vectorize it.
def to_degrees(x):
    return x / np.pi * 180.0


def to_radians(x):
    return x / 180.0 * np.pi


def vec(*x):
    return np.array(x)


def normalized(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def matrix_from_xyz_xyzw(trans: XyzXyzw) -> np.ndarray:
    # Create a 4x4 identity matrix
    xyz, xyzw = trans
    se3_matrix = np.eye(4)

    # Set the translation part
    se3_matrix[:3, 3] = xyz

    # Create a rotation matrix from the quaternion
    rotation = R.from_quat(xyzw)
    se3_matrix[:3, :3] = rotation.as_matrix()

    return se3_matrix


def xyz_xyzw_from_matrix(mat) -> XyzXyzw:
    # Extract the translation part
    xyz = mat[:3, 3]

    # Extract the rotation part
    rotation = mat[:3, :3]
    xyzw = R.from_matrix(rotation).as_quat()

    return xyz, xyzw
