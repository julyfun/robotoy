import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open("save_itp_q_1733565152.7185934.json", "r") as file:
    data = json.load(file)

# 初始化数据列表
timestamps = []
positions = [[] for _ in range(8)]

# 提取时间戳和质点位置
for entry in data:
    positions_data, timestamp = entry
    if len(positions_data) == 8:
        timestamps.append(timestamp)
        for i in range(8):
            positions[i].append(positions_data[i])

# 绘制图形
plt.figure(figsize=(10, 6))
for i in range(8):
    plt.plot(timestamps, positions[i], label=f"质点 {i+1}", marker="o")

plt.xlabel("时间")
plt.ylabel("位置")
plt.title("质点位置随时间变化图")
plt.legend()
plt.show()
