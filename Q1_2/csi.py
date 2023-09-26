# 可视化真值与预测结果

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader
from datasets.dataloader import *
from FURENet.FURENet import FURENet
from utils.utils import csi

dataset = DataSet_for_7km()

train_size = int(0.9 * len(dataset))  # 80% 用于训练集
# val_size = len(dataset) - train_size  # 剩余部分用于验证集
train_dataset, val_dataset = Subset(dataset, list(range(train_size))), Subset(dataset, list(range(train_size, len(dataset))))

net = FURENet(10)
net.load_state_dict(torch.load('/media/disk8T/zhn/frunet/log/89.pth')['net'])

keep1 = [0] * 10
keep2 = [0] * 10
count1 = 0
for k in range(len(val_dataset)):

    zin, zdr, kdp, zout = val_dataset[k]

    zin, zdr, kdp, zout = torch.from_numpy(zin).unsqueeze(0), torch.from_numpy(zdr).unsqueeze(0), torch.from_numpy(kdp).unsqueeze(0), torch.from_numpy(zout).unsqueeze(0)

    min_, max_ = dataset.norm_param['dBZ']

    pre_out = net(zin, zdr, kdp)

    zout = (zout * (max_-min_) + min_).detach().squeeze(0).numpy()

    pre_out = torch.clamp(pre_out, 0, 1)

    pre_out = (pre_out * (max_-min_) + min_).detach().squeeze(0).numpy()
    # 创建一个包含10个子图的画布


    for i in range(10):
        n11, n10, n01, n00 = csi(zout[i], pre_out[i])
        if (n11 + n10 + n01 + n00) == 0:
            pass
        else:
            keep1[i] += (n11 + n00) / (n11 + n10 + n01 + n00)
        if (n11 + n10 + n01) == 0:
            pass
        else:
            keep2[i] += (n11) / (n11 + n10 + n01)
    count1 += 1
    if count1 == 100:
        break
    # print(count1)
keep1 = [i/count1 for i in keep1]
keep2 = [i/count1 for i in keep2]
print(keep1)
print(keep2)
