import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
with open("save_itp_q_1733405654.9756134.json") as f:
    data = f.read()

# 解析数据
l = eval(data)

# 提取包含 8 个元素的子列表
filtered_l = [item for item in l if len(item) == 8]

# 转换为 NumPy 数组便于处理
filtered_array = np.array(filtered_l)

# 生成时间轴，间隔为 0.01 秒
time = np.arange(0, 0.01 * len(filtered_array), 0.01)

# 检查时间长度是否匹配
if len(time) > len(filtered_array):
    time = time[: len(filtered_array)]
elif len(time) < len(filtered_array):
    filtered_array = filtered_array[: len(time)]

# 绘制 8 条曲线
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.plot(time, filtered_array[:, i], label=f"Element {i+1}")

plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("变化曲线")
plt.legend()
plt.grid(True)
plt.show()
