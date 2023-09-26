# 可视化真值与预测结果

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from datasets.dataloader import *
from FURENet.FURENet import FURENet

j = 103

dataset = DataSet_for_7km()

train_size = int(0.9 * len(dataset))  # 80% 用于训练集
# val_size = len(dataset) - train_size  # 剩余部分用于验证集
train_dataset, val_dataset = Subset(dataset, list(range(train_size))), Subset(dataset, list(range(train_size, len(dataset))))

net = FURENet(10)
net.load_state_dict(torch.load('/media/disk8T/zhn/frunet/log/10.pth')['net'])

zin, zdr, kdp, zout = val_dataset[j]

zin, zdr, kdp, zout = torch.from_numpy(zin).unsqueeze(0), torch.from_numpy(zdr).unsqueeze(0), torch.from_numpy(kdp).unsqueeze(0), torch.from_numpy(zout).unsqueeze(0)

min_, max_ = -21.096962, 67.22222

pre_out = net(zin, zdr, kdp)

zout = (zout * (max_-min_) + min_).detach().squeeze(0).numpy()

pre_out = torch.clamp(pre_out, 0, 1)

pre_out = (pre_out * (max_-min_) + min_).detach().squeeze(0).numpy()
# 创建一个包含10个子图的画布


fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距

# 定义颜色区间和颜色映射
color_intervals = np.arange(0, 81, 10)  # 定义颜色区间为[0-10]和[10-20]
cmap = plt.get_cmap('viridis', len(color_intervals) - 1)  # 使用'viridis'颜色映射，根据颜色区间的数量选择颜色

from matplotlib.colors import BoundaryNorm
# 创建颜色标准化器
norm = BoundaryNorm(color_intervals, cmap.N, clip=True)

# 提取并绘制每个热力图
for i in range(10):
    row, col = i // 5, i % 5
    ax = axes[row, col]
    heatmap_data = zout[i]  # 从你的NumPy数组中提取第i个热力图数据
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, origin='lower', extent=[0, 256, 0, 256])
    ax.set_title(f'Heatmap {i + 1}')

# 添加颜色条
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', ticks=color_intervals)
cbar.set_label('Color Intervals')

# 显示图形
plt.savefig('zout_{}.png'.format(j))
plt.show()

# fig, axes = plt.subplots(2, 5, figsize=(12, 6))
# plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距

# # 提取并绘制每个热力图
# for i in range(10):
#     row, col = i // 5, i % 5
#     ax = axes[row, col]
#     heatmap_data = pre_out[i]  # 从你的NumPy数组中提取第i个热力图数据
#     ax.imshow(heatmap_data, cmap='viridis')  # 使用'viridis'颜色映射，你可以根据需要更改
#     ax.set_title(f'Heatmap {i + 1}')

# # 显示图形
# plt.show()
# plt.savefig('preout.png')

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距

# 定义颜色区间和颜色映射
color_intervals = np.arange(0, 81, 10)  # 定义颜色区间为[0-10]和[10-20]
cmap = plt.get_cmap('viridis', len(color_intervals) - 1)  # 使用'viridis'颜色映射，根据颜色区间的数量选择颜色

from matplotlib.colors import BoundaryNorm
# 创建颜色标准化器
norm = BoundaryNorm(color_intervals, cmap.N, clip=True)

# 提取并绘制每个热力图
for i in range(10):
    row, col = i // 5, i % 5
    ax = axes[row, col]
    heatmap_data = pre_out[i]  # 从你的NumPy数组中提取第i个热力图数据
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, origin='lower', extent=[0, 256, 0, 256])
    ax.set_title(f'Heatmap {i + 1}')

# 添加颜色条
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', ticks=color_intervals)
cbar.set_label('Color Intervals')

# 显示图形
plt.savefig('predict_{}.png'.format(j))
plt.show()