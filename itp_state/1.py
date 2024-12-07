import matplotlib.pyplot as plt
import numpy as np

Flt = np.float32


@np.vectorize
def limited_by(x, x_abs_max: Flt):
    return np.clip(x, -x_abs_max, x_abs_max)


@np.vectorize
def sign(x):
    return 1 if x >= 0 else -1


def itpltn_best_v_a(
    x0,
    v0,  # .
    a0,  # .
    x_tar,
    v_tar,  # .
    f,
    v_max,
    a_max,
    j_max,
    a_in_use,
):
    # don't modify the original data
    v0 = np.copy(v0)
    a0 = np.copy(a0)
    v_tar = np.copy(v_tar)

    d_h = x_tar - (x0 + x0 + v0 / f + 0.5 * a0 / f / f) / 2
    di = np.where(d_h < 0, -1, 1)
    d_h *= di
    v0 *= di
    a0 *= di
    v_tar *= di

    v_tar = np.where(
        sign(v_tar) == sign(d_h),
        sign(v0) * np.min([
            np.abs(v_tar),
            np.sqrt(v0 ** 2 + 2.0 * a_max * np.abs(d_h))
        ]
        ),
        0.0
    )
    v_sug = limited_by(
        np.sqrt(
            2.0 * a_max * d_h + v_tar ** 2
        ) * a_in_use,
        v_max
    )

    a_limited = limited_by((v_sug - v0) * f, a_max)
    j_limited = limited_by((a_limited - a0) * f, j_max)
    return j_limited * di


def itpltn_best_v_j(
    x0,
    v0,
    a0,
    x_tar,
    v_tar,
    f,
    v_max,
    a_max,
    a_in_use,
    j_max
):
    d = x_tar - x0
    d_next_control_if_v_unchanged = x_tar - (x0 + v0 / f)
    d_half = (d + d_next_control_if_v_unchanged) / 2.0
    # v_tar 不可太大
    v_tar = np.where(
        sign(v_tar) == sign(d),
        sign(v0) * np.min([
            np.abs(v_tar),
            np.sqrt(v0 ** 2 + 2.0 * a_max * np.abs(d))
        ]
        ),
        0.0
    )
    # dv 曲线上的点
    v_curve_at_half_d = limited_by(
        sign(d_half) * np.sqrt(
            2.0 * a_max * np.abs(d_half) + v_tar ** 2
        ) * a_in_use,
        v_max
    )
    # [TODO] 检查距离小的策略是否正确
    # v_fixed = np.where(
    #     np.signbit(v_curve_at_half_d) != np.signbit(d),
    #     limited_by(v_tar, v_max),
    #     v_curve_at_half_d
    # )
    a_limited = limited_by((v_curve_at_half_d - v0) * f, a_max)
    # a_damped = a_limited - sign(v0) * a_max * DAMP
    j_limited = limited_by((a_limited - a0) * f, j_max)
    return j_limited


class ItpState:
    def __init__(self, x0, v0, v_max, a_max, j_max, fps):
        ...
        self.dof = len(x0)
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.fps = fps
        self.pre_sent_x = x0
        self.pre_sent_v = v0
        self.pre_sent_a = np.array([0.0] * self.dof, dtype=Flt)
        self.pre_sent_j = np.array([0.0] * self.dof, dtype=Flt)

        # [hydra]
        self.control_eps_deg = Flt(1e-3)
        self.a_in_use = 0.5

    # [TODO] 注意传入的 x_tar 可能发生了 2pi 突跃，取决于 ik
    def interpolate(self, x_tar, v_tar, points_needed):
        ...
        ret = []
        done = 0
        while points_needed > 0:
            if done == (1 << self.dof) - 1:
                ret.append(x_tar)
                points_needed -= 1
                continue
            j = itpltn_best_v_j(
                self.pre_sent_x,
                self.pre_sent_v,
                self.pre_sent_a,
                x_tar,
                v_tar,
                self.fps,
                self.v_max,
                self.a_max,
                self.a_in_use,
                self.j_max,
            )
            so_a = self.pre_sent_a + j / self.fps
            so_v = self.pre_sent_v + so_a / self.fps
            so_x = self.pre_sent_x + so_v / self.fps
            ret.append(so_x)
            points_needed -= 1
            bin_done = np.sum((np.abs(so_x - x_tar) <
                              self.control_eps_deg) * (1 << np.arange(self.dof)))
            done = bin_done
            self.pre_sent_j = j
            self.pre_sent_a = so_a
            self.pre_sent_v = so_v
            self.pre_sent_x = so_x

        return ret


def test_interpolate():
    # 初始化参数
    x0 = np.array([2, 2, 2, 2, 2, 2], dtype=np.float32)
    v0 = np.array([1, -1, 2, -2, 0, 0], dtype=np.float32)
    v_max = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    a_max = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float32)
    j_max = np.array([400, 400, 400, 400, 400, 400], dtype=np.float32)
    # x0 = np.array([2], dtype=np.float32)
    # v_max = np.array([3.0], dtype=np.float32)
# a_max = np.array([4.0], dtype=np.float32)
    # j_max = np.array([400], dtype=np.float32)
    fps = 300.0

    # 创建 ItpState 对象
    itp_state = ItpState(x0, v0, v_max, a_max, j_max, fps)

    # 目标位置和速度
    x_tar = np.array([1.1, 2.3, 2.4, 2.5, 2.6, 2.7], dtype=np.float32)
    v_tar = np.array([2.0] * 6, dtype=np.float32)
    points_needed = 400

    # 调用 interpolate 方法
    result = itp_state.interpolate(x_tar, v_tar, points_needed)

    # 提取插值结果
    x_vals = np.array(result)
    t = np.arange(0, points_needed) / fps

    # 绘制结果
    plt.figure(figsize=(12, 8))
    for i in range(x_vals.shape[1]):
        plt.plot(t, x_vals[:, i][:], marker='o',
                 linestyle='-', label=f'Dimension {i+1}')
    plt.title('Interpolation Path')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.legend()
    plt.grid(True)
    plt.show()


# 运行测试函数
test_interpolate()
