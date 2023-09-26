# 可视化真值与预测结果

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from datasets.dataloader import *
from FURENet.FURENet import *
from matplotlib.colors import BoundaryNorm
import torch

j = 58

dataset = DataSet_for_ZH()

train_size = int(0.8 * len(dataset))  # 80% 用于训练集
# val_size = len(dataset) - train_size  # 剩余部分用于验证集
train_dataset, val_dataset = Subset(dataset, list(range(train_size))), Subset(dataset, list(range(train_size, len(dataset))))

net = FURENet_1(1)
net.load_state_dict(torch.load("/media/disk8T/zhn/Q4/log_1/90.pth")['net'])

zin, rain = val_dataset[j]

zin, rain= torch.from_numpy(zin).unsqueeze(0), torch.from_numpy(rain).unsqueeze(0)

pre_out = net(zin)

rain = rain * 1200

pre_out = torch.clamp(pre_out, 0, 1)

pre_out = pre_out * 1200

print(torch.max(pre_out))
print(torch.max(rain))
# 创建一个包含10个子图的画布

heatmap1 = rain.squeeze(0).detach()  # 从你的数据中获取第一个热力图数据
heatmap2 = pre_out.squeeze(0).detach() # 从你的数据中获取第二个热力图数据

data = torch.cat((heatmap1, heatmap2))
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距

# 定义颜色区间和颜色映射
color_intervals = np.arange(0, 200, 10)  # 定义颜色区间为[0-10]和[10-20]
cmap = plt.get_cmap('viridis', len(color_intervals) - 1)  # 使用'viridis'颜色映射，根据颜色区间的数量选择颜色

# 提取并绘制每个热力图
for i in range(2):
    ax = axes[i]
    heatmap_data = data[i]  # 从你的NumPy数组中提取第i个热力图数据
    norm = BoundaryNorm(color_intervals, cmap.N, clip=True)  # 创建颜色标准化器
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, origin='lower', extent=[0, 256, 0, 256])
    ax.set_title(f'Heatmap {i + 1}')

# 添加颜色条
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', ticks=color_intervals)
cbar.set_label('Color Intervals')

# 显示图形
plt.savefig('test3_{}.tif'.format(j))
plt.show()






