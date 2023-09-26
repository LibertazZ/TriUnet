# 可视化dBZ与降雨量正相关关系

import os
import numpy as np
import matplotlib.pyplot as plt

path = "/media/disk8T/zhn/dataset/NJU_CPOL_kdpRain/"
# path = '/media/disk8T/zhn/dataset/NJU_CPOL_update2308/dBZ/3.0km/'

file_names = sorted(os.listdir(path))

min_ = float('inf')
max_ = float('-inf')



arrays = []
for idx, file in enumerate(file_names):
    if idx !=70:
        continue
    new_path = path + file
    print(new_path)  # 找到一段连续降雨的记录
    frams = sorted([f for f in os.listdir(new_path) if f.endswith('.npy')])
    for fram in frams:
        new_new_path = new_path + '/' + fram
        data = np.load(new_new_path)
        # 生成1000个示例的256x256的随机数组（你可以替换成你的数据加载逻辑）

        if np.max(data) > max_:
            max_ = np.max(data)
        if np.min(data) < min_:
            min_ = np.min(data)

        arrays.append(data)

data = arrays[0:10]

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距

# 定义颜色区间和颜色映射
color_intervals = np.arange(0, 71, 10)  # 定义颜色区间为[0-10]和[10-20]
cmap = plt.get_cmap('viridis', len(color_intervals) - 1)  # 使用'viridis'颜色映射，根据颜色区间的数量选择颜色

from matplotlib.colors import BoundaryNorm
# 创建颜色标准化器
norm = BoundaryNorm(color_intervals, cmap.N, clip=True)

# 提取并绘制每个热力图
for i in range(10):
    row, col = i // 5, i % 5
    ax = axes[row, col]
    heatmap_data = data[i]  # 从你的NumPy数组中提取第i个热力图数据
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, origin='lower', extent=[0, 256, 0, 256])
    ax.set_title(f'Heatmap {i + 1}')

# 添加颜色条
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', ticks=color_intervals)
cbar.set_label('Color Intervals')

# 显示图形
plt.savefig('rain.png')