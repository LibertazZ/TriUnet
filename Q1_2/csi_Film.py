# 可视化真值与预测结果

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader,random_split
from datasets.dataloader import *
from FURENet.FURENet import FURENet_Film
from utils.utils import csi

dataset = DataSet_for_all()

train_size = int(0.8 * len(dataset))  # 80% 用于训练集
val_size = len(dataset) - train_size  # 剩余部分用于验证集

# 使用 random_split 函数自动进行数据集划分
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


net = FURENet_Film(10)
net.load_state_dict(torch.load('/media/disk8T/zhn/frunet/log_2/80.pth')['net'])

keep1 = [0] * 10
keep2 = [0] * 10
keep3 = [0] * 10
keeper = [keep1, keep2, keep3]
keep1p = [0] * 10
keep2p = [0] * 10
keep3p = [0] * 10
keeperp = [keep1p, keep2p, keep3p]
count1 = 0
count2 = 0
count3 = 0
count = [count1, count2, count3]
for k in range(len(val_dataset)):

    zin, zdr, kdp, lambda_, zout = val_dataset[k]

    if lambda_[0] == 1:
        idx = 0
        min_, max_ = -33.916622, 79.81039
    elif lambda_[1] == 1:
        idx = 1
        min_, max_ = -39.972763, 99.68255
    else:
        idx = 2
        min_, max_ = -21.096962, 67.22222

    zin, zdr, kdp, lambda_, zout = torch.from_numpy(zin).unsqueeze(0), torch.from_numpy(zdr).unsqueeze(0), torch.from_numpy(kdp).unsqueeze(0),torch.from_numpy(lambda_).float().unsqueeze(0), torch.from_numpy(zout).unsqueeze(0)

    pre_out = net(zin, zdr, kdp, lambda_)

    zout = (zout * (max_-min_) + min_).detach().squeeze(0).numpy()

    pre_out = torch.clamp(pre_out, 0, 1)

    pre_out = (pre_out * (max_-min_) + min_).detach().squeeze(0).numpy()
    # 创建一个包含10个子图的画布


    for i in range(10):
        n11, n10, n01, n00 = csi(zout[i], pre_out[i])
        if (n11 + n10 + n01 + n00) == 0:
            pass
        else:
            keeperp[idx][i] += (n11 + n00) / (n11 + n10 + n01 + n00)
        if (n11 + n10 + n01) == 0:
            pass
        else:
            keeper[idx][i] += (n11) / (n11 + n10 + n01)
    count[idx] += 1
    print(count)
    if min(count) == 100:
        break
    # print(count1)
print(keeper)
print(keeperp)
print(count)
