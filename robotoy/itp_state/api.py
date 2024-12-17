import numpy as np

from ..build.itp_state import ItpState
from ..build.ring_buffer import RingBuffer


class ItpRingApi:
    def __init__(self, fps, buf_time, v_max, a_max, j_max):
        self.itp_state = ItpState()
        self.itp_state.init(v_max=v_max, a_max=a_max, j_max=j_max, fps=fps)
        self.itp_state_t = -1e9

        self.fps = fps
        self.buf_time = buf_time
        self.buf_size = int(buf_time * fps)
        self.ring = RingBuffer(self.buf_size)

    def interpolate(self, time, tar_x, init_x, init_v):
        if self.ring.get_valid_len() == 0:
            self.itp_state.init(x0=init_x, v0=init_v)
            self.itp_state_t = time
        point_needed = self.buf_size - self.ring.get_valid_len()
        # 插到最后一个的目标是在 buf_time 后来到目标位置
        res = self.itp_state.interpolate(
            tar_x,
            np.zeros_like(tar_x),
            point_needed,
            first_delta_t=self.itp_state_t - time + self.buf_time,
        )
        for itp in res:
            self.ring.push(itp[0])
        self.itp_state_t += point_needed / self.fps


def test_itp_ring_api():
    def arr(*li):
        return np.array(li)

    itp_ring_api = ItpRingApi(100, 0.2, arr(1), arr(1), arr(1))
    itp_ring_api.interpolate(
        1173,
        arr(3),
        arr(0),
        arr(0),
    )


if __name__ == "__main__":
    test_itp_ring_api()
