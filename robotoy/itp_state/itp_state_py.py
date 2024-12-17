import time
import numpy as np
import scipy.optimize as opt
from loguru import logger
from matplotlib import pyplot as plt
from loguru import logger

m_logger = logger.bind(name="itp_state")

Flt = np.float32
EPS = 1e-6


def arr(*l):
    return np.array(l, dtype=Flt)


# [begin]


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
    v_abs_max,
    a_abs_max,
    a_adjust_factor,
):
    # don't modify the original data
    v0 = np.copy(v0)
    a0 = np.copy(a0)
    v_tar = np.copy(v_tar)

    d_h = x_tar - (x0 + x0 + v0 / f + 0.5 * a0 / f / f) / 2
    dir = np.where(d_h < 0, -1, 1)
    d_h *= dir
    v0 *= dir
    a0 *= dir
    v_tar *= dir
    m_logger.info(f"    ***")
    m_logger.info(f"    d_h: {d_h}")
    m_logger.info(f"    v0: {v0}")
    m_logger.info(f"    a0: {a0}")
    m_logger.info(f"    v_tar: {v_tar}")
    v0 -= v_tar
    v_max = np.maximum(v_abs_max - v_tar, 0)
    v_min = np.minimum(-v_abs_max - v_tar, 0)

    # v_tar = np.where(
    #     sign(v_tar) == sign(d_h),
    #     sign(v0)
    #     * np.min([np.abs(v_tar), np.sqrt(v0**2 + 2.0 * a_abs_max * np.abs(d_h))]),
    #     0.0,
    # )
    # 以下基于 -v_tar 坐标
    v_sug = limited_by(np.sqrt(2.0 * a_abs_max * d_h), v_max)
    m_logger.info(f"    v_sug: {v_sug}")
    a_limited = limited_by((v_sug - v0) * f, a_abs_max * a_adjust_factor)
    m_logger.info(f"    a_limited: {a_limited}")
    return a_limited * dir


def itpltn_best_v_j(
    x0,
    v0,  # .
    a0,  # .
    x_tar,
    v_tar,  # .
    f,
    v_abs_max,
    a_abs_max,
    j_abs_max,
):
    v0 = np.copy(v0)
    a0 = np.copy(a0)
    v_tar = np.copy(v_tar)
    # d half t if v & a unchanged
    d_h = x_tar - (x0 + x0 + v0 / f + 0.5 * a0 / f**2) / 2
    m_logger.info(f"d_h: {d_h}")
    m_logger.info(f"v0: {v0}")
    m_logger.info(f"a0: {a0}")
    dir = np.where(d_h < 0, -1, 1)
    d_h *= dir
    v0 *= dir
    a0 *= dir
    v_tar *= dir

    # [minus v_tar.begin]
    v0 -= v_tar
    v_max = np.maximum(v_abs_max - v_tar, 0)
    v_min = np.minimum(-v_abs_max - v_tar, 0)
    # d -> v_sug, a_sug

    @np.vectorize
    def sug_inv_t(d, v0, a0, v_m1, a_m, j_m):
        def solve_t_sug(coe, d, init):
            coe[0] -= d
            poly = np.polynomial.Polynomial(coe)
            der = poly.deriv()
            return opt.newton(poly, x0=init, fprime=der)

        if a_m**2 >= v_m1 * j_m:
            """
            两段的情况
            """
            t1 = np.sqrt(v_m1 / j_m)
            d1 = j_m * (v_m1 / j_m) ** (3 / 2) / 6
            d2 = v_m1 * np.sqrt(v_m1 / j_m)
            m_logger.info(f"d1: {d1} | d2: {d2}")
            if d >= d2:
                return v_m1, 0
            if d >= d1:
                d_in_t2_coe = [
                    j_m * (v_m1 / j_m) ** (3 / 2) / 3,
                    -v_m1,
                    j_m * np.sqrt(v_m1 / j_m),
                    -j_m / 6,
                ]
                t_sug = solve_t_sug(d_in_t2_coe, d, t1)
                v_sug = (
                    -j_m * t_sug**2 / 2
                    + 2 * j_m * t_sug * np.sqrt(v_m1 / j_m)
                    - 2 * j_m * v_m1 / j_m
                    + v_m1
                )
                a_sug = j_m * np.sqrt(v_m1 / j_m) - j_m * (t_sug - np.sqrt(v_m1 / j_m))
                return v_sug, a_sug
            t_sug = 6 ** (1 / 3) * (d / j_m) ** (1 / 3)
            v_sug = j_m * t_sug**2 / 2
            a_sug = j_m * t_sug
            m_logger.info(f"t_sug: {t_sug}")
            return v_sug, a_sug
        t1 = a_m / j_m
        t2 = v_m1 / a_m
        t3 = a_m / j_m + v_m1 / a_m
        d1 = a_m**3 / (6 * j_m**2)
        d2 = a_m**3 / (6 * j_m**2) - a_m * v_m1 / (2 * j_m) + v_m1**2 / (2 * a_m)
        d3 = v_m1 * (a_m**2 + j_m * v_m1) / (2 * a_m * j_m)
        if d >= d3:
            return v_m1, 0
        if d >= d2:
            # [int3 这段的解]
            d_in_t3_coe = [
                (a_m**6 + j_m**3 * v_m1**3) / (6 * a_m**3 * j_m**2),
                (-(a_m**4) - j_m**2 * v_m1**2) / (2 * a_m**2 * j_m),
                (a_m**2 + j_m * v_m1) / (2 * a_m),
                -j_m / 6,
            ]
            t_sug = solve_t_sug(d_in_t3_coe, d, t2)
            v_sug = (
                -(a_m**2) / (2 * j_m)
                + a_m * t_sug
                - j_m * t_sug**2 / 2
                + j_m * t_sug * v_m1 / a_m
                - j_m * v_m1**2 / (2 * a_m**2)
            )
            a_sug = a_m - j_m * (t_sug - v_m1 / a_m)
            # return -(a_m**2) / (2 * j_m) + v_m1, a_m
            return v_sug, a_sug
        if d >= d1:
            # [int2 这段的解 (曲线 + 直线)]
            t_sug = (
                3 * a_m**2 + np.sqrt(3) * np.sqrt(a_m * (-(a_m**3) + 24 * d * j_m**2))
            ) / (6 * a_m * j_m)
            v_sug = -(a_m**2) / (2 * j_m) + a_m * t_sug
            return v_sug, a_m
        t_sug = 6 ** (1 / 3) * (d / j_m) ** (1 / 3)
        v_sug = j_m * t_sug**2 / 2
        a_sug = j_m * t_sug
        return v_sug, a_sug

    v_sug, a_sug = sug_inv_t(d_h, v0, a0, v_max, a_abs_max, j_abs_max)
    v_sug, a_sug = v_sug, -a_sug  # 反向积分

    m_logger.info(f"v_sug: {v_sug}")
    m_logger.info(f"a_sug: {a_sug}")
    j = itpltn_best_v_a(
        v0,
        a0,
        np.zeros_like(x0),
        v_sug,
        a_sug * (1.0 if np.abs(v_sug) < EPS else np.clip(v0 / v_sug, 2.0 / 3.0, 1.0)),
        f,
        a_abs_max,
        j_abs_max,
        a_adjust_factor=1.1,
    )
    return j * dir


class ItpState:
    def __init__(self):
        self.v_max = None
        self.a_max = None
        self.j_max = None
        self.fps = None
        self.pre_sent_x = None
        self.pre_sent_v = None
        self.pre_sent_a = None
        self.pre_sent_j = None

    def init(self, x0=None, v0=None, v_max=None, a_max=None, j_max=None, fps=None):
        params = {
            "pre_sent_x": x0,
            "pre_sent_v": v0,
            "v_max": v_max,
            "a_max": a_max,
            "j_max": j_max,
            "fps": fps,
        }
        for attr, value in params.items():
            if value is not None:
                setattr(self, attr, value)
        if any(
            map(
                lambda x: x is not None,
                [x0, v0],
            )
        ):
            shape = x0 if x0 is not None else v0
            self.pre_sent_a = np.zeros_like(shape, dtype=Flt)
            self.pre_sent_j = np.zeros_like(shape, dtype=Flt)
        # [hydra]

    # [TODO] 注意传入的 x_tar 可能发生了 2pi 突跃，取决于 ik
    def interpolate(self, x_tar, v_tar, points_needed, first_delta_t=0):
        # if any none
        if any(
            map(
                lambda x: x is None,
                [
                    self.v_max,
                    self.a_max,
                    self.j_max,
                    self.fps,
                    self.pre_sent_x,
                    self.pre_sent_v,
                    self.pre_sent_a,
                    self.pre_sent_j,
                ],
            )
        ):
            raise ValueError("ItpState not initialized")
        ret = []
        st = time.time()
        for i in range(points_needed):
            m_logger.info(f"[ItpState] last itp time: {(time.time() - st) * 1000}")
            st = time.time()
            m_logger.info("---")
            m_logger.info(f"delta: {i * 1.0 / self.fps}")
            t_tar = np.ones_like(x_tar) * (first_delta_t + i / self.fps)
            x_tar_t = x_tar + v_tar * t_tar
            v_tar_t = v_tar

            j = itpltn_best_v_j(
                self.pre_sent_x,
                self.pre_sent_v,
                self.pre_sent_a,
                x_tar_t,
                limited_by(v_tar_t, self.v_max),
                self.fps,
                self.v_max,
                self.a_max,
                self.j_max,
            )
            so_a = self.pre_sent_a + j / self.fps
            so_v = self.pre_sent_v + so_a / self.fps
            so_x = self.pre_sent_x + so_v / self.fps
            m_logger.info(f"so_x: {so_x}")
            m_logger.info(f"so_v: {so_v}")
            m_logger.info(f"so_a: {so_a}")
            m_logger.info(f"j: {j}")
            ret.append((so_x, so_v, so_a, j))
            self.pre_sent_j = j
            self.pre_sent_a = so_a
            self.pre_sent_v = so_v
            self.pre_sent_x = so_x

        return ret
