import numpy as np
import time
import inspect
from itp_state import arr


from matplotlib import pyplot as plt
from loguru import logger
import sys
import os

if 1:
    # parent
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # from itp_state import ItpState
    from build.itp_state import ItpState


logger.remove()
logger.add(
    sys.stdout, filter=lambda r: r["extra"].get("name") in ["itp_statex", "test2"]
)

logger = logger.bind(name="test2")
logger.info(f"Using ItpState from {inspect.getmodule(ItpState).__file__}")

Flt = np.float64


def test1():
    # 初始化参数
    x0 = np.array([2, 2, 2, 2, 2, 2], dtype=Flt)
    v0 = np.array([1, -1, 2, -2, 0, 0], dtype=Flt)
    v_max = np.array([3.0, 3.0, 3.0, 3.0, 300.0, 300.0], dtype=Flt)
    a_max = np.array([40.0, 40.0, 400.0, 0.8, 400.0, 4.0], dtype=Flt)
    j_max = np.array([400, 4000, 400, 400, 400, 400], dtype=Flt)
    fps = 300.0

    # 创建 ItpState 对象
    itp_state = ItpState()
    itp_state.init(x0, v0, v_max, a_max, j_max, fps)

    # 目标位置和速度
    x_tar = np.array([1.1, 2.3, 2.4, 2.5, 2.6, 2.7], dtype=Flt)
    v_tar = np.array([0.0] * 6, dtype=Flt)
    points_needed = int(2.0 * fps)

    # 调用 interpolate 方法
    result = itp_state.interpolate(x_tar, v_tar, points_needed, first_delta_t=0.0)

    # 提取插值结果
    result = np.array(result)
    x_vals = result[:, 0]
    t = np.arange(0, points_needed) / fps

    # 绘制结果
    plt.figure(figsize=(12, 8))
    for i in range(x_vals.shape[1]):
        plt.plot(
            t, x_vals[:, i][:], marker="o", linestyle="-", label=f"Dimension {i+1}"
        )
    plt.title("Interpolation Path")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position")
    plt.legend()
    plt.grid(True)
    plt.show()


def test2():
    x0 = arr(0)
    v0 = arr(-2)
    v_max = arr(5)
    a_max = arr(40)
    j_max = arr(400)
    fps = 300.0
    itp_state = ItpState(x0, v0, v_max, a_max, j_max, fps)
    x_tar = arr(2)
    v_tar = arr(1)
    points_needed = int(1.0 * fps)
    res = itp_state.interpolate(x_tar, v_tar, points_needed)
    res = np.array(res)

    x_vals = np.array(res[:, 0])
    v_vals = np.array(res[:, 1])
    a_vals = np.array(res[:, 2])
    j_vals = np.array(res[:, 3])
    t = np.arange(0, points_needed) / fps

    def plot_subplots(t, data, titles, y_labels):
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
                plt.axhline(
                    0, color="red", linestyle="--", linewidth=2
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

    plot_subplots(t, data, titles, y_labels)


def test3():
    x0 = arr(2)
    v0 = arr(2)
    v_max = arr(3.0)
    a_max = arr(40.0)
    j_max = arr(400)
    fps = 300.0
    itp_state = ItpState()
    itp_state.init(v_max=v_max, a_max=a_max, j_max=j_max, fps=fps)
    itp_state.init(x0=x0, v0=v0)

    # 2hz 振幅 1m 的正弦序列，时间间隔 1 / 30s，持续 2s
    t_samples = np.arange(0, 10, 1 / 60)
    freq = 10.0
    amp = 0.5
    x_samples = np.sin(2 * np.pi * freq * t_samples) * amp
    v_samples = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t_samples) * amp
    # v_samples = np.zeros_like(x_samples)

    # v_samples = np.zeros_like(x_samples)
    last_itpltn_t = 0
    res = []
    t_vals = []
    st = time.time()
    for i, (x, v, t) in enumerate(zip(x_samples, v_samples, t_samples)):
        # if i == 444:
        #     itp_state.init(x0=arr(-10), v0=arr(0))
        # 插到 t
        points_needed = int((t - last_itpltn_t) * fps)
        res += itp_state.interpolate(
            arr(x), arr(v), points_needed, first_delta_t=(last_itpltn_t - t)
        )
        t_vals += [last_itpltn_t + i / fps for i in range(points_needed)]
        last_itpltn_t += points_needed / fps
    logger.info(f"itp time: {(time.time() - st) * 1000}")

    res = np.array(res)
    x_vals = np.array(res[:, 0])
    v_vals = np.array(res[:, 1])
    a_vals = np.array(res[:, 2])
    j_vals = np.array(res[:, 3])

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
    extra_data = [x_samples, v_samples]
    extra_t = t_samples

    plot_subplots(t_vals, data, titles, y_labels, extra_data, extra_t)


def test4():
    x0 = arr(2)
    v0 = arr(2)
    v_max = arr(3.0)
    a_max = arr(400.0)
    j_max = arr(400)
    fps = 300.0
    itp_state = ItpState()
    itp_state.init(v_max=v_max, a_max=a_max, j_max=j_max, fps=fps)
    itp_state.init(x0=x0, v0=v0)

    t_samples = np.arange(0, 1.0, 1 / 60)
    x_samples = 2.4 * np.ones_like(t_samples)
    v_samples = np.zeros_like(t_samples)
    last_itpltn_t = 0
    res = []
    t_vals = []
    for i, (x, v, t) in enumerate(zip(x_samples, v_samples, t_samples)):
        points_needed = int((t - last_itpltn_t) * fps)
        # logger.info(f"start: {last_itpltn_t}")
        res += itp_state.interpolate(
            arr(x), arr(v), points_needed, first_delta_t=(last_itpltn_t - t)
        )
        t_vals += [last_itpltn_t + i / fps for i in range(points_needed)]
        last_itpltn_t += points_needed / fps

    res = np.array(res)
    x_vals = np.array(res[:, 0])
    v_vals = np.array(res[:, 1])
    a_vals = np.array(res[:, 2])
    j_vals = np.array(res[:, 3])

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
    extra_data = [x_samples, v_samples]
    extra_t = t_samples

    plot_subplots(t_vals, data, titles, y_labels, extra_data, extra_t)


test3()
