import numpy as np
import time
import inspect
from itp_state import arr

from build.itp_state import ItpState

# from itp_state import ItpState
from matplotlib import pyplot as plt
from loguru import logger
import sys


logger.remove()
logger.add(
    sys.stdout, filter=lambda r: r["extra"].get("name") in ["itp_statex", "test2"]
)

logger = logger.bind(name="test2")
logger.info(f"Using ItpState from {inspect.getmodule(ItpState).__file__}")

Flt = np.float64


def test():
    V_MAX = [2.0] * 3 + [3.14] * 5 + [10.0] * 22
    A_MAX = [4.0] * 3 + [6.0] * 5 + [20.0] * 22
    J_MAX = [400.0] * 3 + [400.0] * 5 + [1000.0] * 22
    cache_q_ros = np.zeros(30)
    cache_q_v_ros = np.zeros(30)

    # [itp states]
    arm_wrist_itp_state = ItpState()
    arm_wrist_v_max = np.array(V_MAX[:8])
    arm_wrist_a_max = np.array(A_MAX[:8])
    arm_wrist_j_max = np.array(J_MAX[:8])
    arm_wrist_fps = 120
    arm_wrist_itp_state.init(
        v_max=arm_wrist_v_max,
        a_max=arm_wrist_a_max,
        j_max=arm_wrist_j_max,
        fps=arm_wrist_fps,
    )
    arm_wrist_itp_t = -10086.0

    hand_itp_state = ItpState()
    hand_v_max = np.array(V_MAX[8:])
    hand_a_max = np.array(A_MAX[8:])
    hand_j_max = np.array(J_MAX[8:])
    hand_fps = 300
    hand_itp_state.init(
        v_max=hand_v_max,
        a_max=hand_a_max,
        j_max=hand_j_max,
        fps=hand_fps,
    )
    hand_itp_t = -10086.0

    TARGET_FUTURE = 0.10
    MAX_ITP_GAP_T = 1.0

    with open("save_q_1733405729.6084485.json") as f:
        data = f.read()

    json_q = eval(data)
    input_t = 1000.0
    input_ts = []
    input_qs = []

    itp_qvaj = []
    itp_ts = []
    for q in json_q:
        input_t += 0.03
        input_ts.append(input_t)
        # [arm wrist]
        # wrist_target_q = q[:8]
        # tar_itp_t = timestamp + TARGET_FUTURE
        # if tar_itp_t - arm_wrist_itp_t > MAX_ITP_GAP_T:
        #     logger.error(f"tar_itp_t: {tar_itp_t}, arm_wrist_itp_t: {arm_wrist_itp_t}")
        #     arm_wrist_itp_state.init(
        #         x0=cache_q_ros[:8],
        #         v0=cache_q_v_ros[:8],
        #     )
        #     arm_wrist_itp_t = timestamp
        # points_needed = int((tar_itp_t - arm_wrist_itp_t) * arm_wrist_fps)
        # res = arm_wrist_itp_state.interpolate(
        #     wrist_target_q,
        #     np.zeros_like(wrist_target_q),
        #     points_needed,
        #     first_delta_t=arm_wrist_itp_t - tar_itp_t,
        # )
        # for itp in res:
        #     # arm_wrist_ctrl_sig_ring.push(itp[0])
        #     vals.append(itp)
        #     t_vals.append(len(vals) / arm_wrist_fps)
        # arm_wrist_itp_t += points_needed / arm_wrist_fps

        # [hand]
        hand_target_q = q[8:]
        input_qs.append(hand_target_q)
        tar_itp_t = input_t + TARGET_FUTURE
        if tar_itp_t - hand_itp_t > MAX_ITP_GAP_T:
            logger.error(f"tar_itp_t: {tar_itp_t}, hand_itp_t: {hand_itp_t}")
            hand_itp_state.init(
                x0=cache_q_ros[8:],
                v0=cache_q_v_ros[8:],
            )
            hand_itp_t = input_t
        points_needed = int((tar_itp_t - hand_itp_t) * hand_fps)
        res = hand_itp_state.interpolate(
            hand_target_q,
            np.zeros_like(hand_target_q),
            points_needed,
            first_delta_t=hand_itp_t - tar_itp_t,
        )
        for i, itp in enumerate(res):
            # hand_ctrl_sig_ring.push(itp[0])
            itp_qvaj.append(itp)
            itp_ts.append(hand_itp_t + i / hand_fps)
        hand_itp_t += points_needed / hand_fps

    itp_qvaj = np.array(itp_qvaj)
    x_vals = np.array(itp_qvaj[:, 0, :3])
    v_vals = np.array(itp_qvaj[:, 1, :3])
    a_vals = np.array(itp_qvaj[:, 2, :3])
    j_vals = np.array(itp_qvaj[:, 3, :3])

    input_qs = np.array(input_qs)
    input_qs = np.array(input_qs[:, :3])

    def plot_subplots(t, data, titles, y_labels, extra_data=None, extra_t=None):
        plt.figure(figsize=(12, 8))
        for i, (data_vals, title, y_label) in enumerate(zip(data, titles, y_labels)):
            plt.subplot(4, 1, i + 1)
            for j in range(data_vals.shape[1]):
                plt.plot(
                    t,
                    data_vals[:, j],
                    marker="o",
                    linestyle="-",
                    label=f"Dimension {j+1}",
                )
            if extra_data is not None and extra_t is not None and i < len(extra_data):
                plt.plot(
                    extra_t,
                    extra_data[i],
                    marker="x",
                    linestyle="--",
                    label="Sample Data",
                )
            plt.axhline(
                0, color="gray", linestyle="--", linewidth=0.5
            )  # 添加 y=0 的水平线
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    data = [x_vals, v_vals, a_vals, j_vals]
    titles = ["X Position", "Velocity", "Acceleration", "Jerk"]
    y_labels = ["X Position", "Velocity", "Acceleration", "Jerk"]
    extra_data = [input_qs]
    extra_t = input_ts

    plot_subplots(itp_ts, data, titles, y_labels, extra_data, extra_t)


test()
