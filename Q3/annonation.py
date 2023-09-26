import os
import numpy as np
from utils.utils import rain_sample

Ith = 100
Ath = 256

# 指定要读取的文件夹路径
# folder_path = '/media/disk8T/zhn/dataset/NJU_CPOL_update2308/'

# f1 = ['ZDR','KDP','dBZ']

# f2 = ['1.0km','3.0km','7.0km']

# for i2 in f2:
#     for i1 in f1:
#         path = folder_path + i1 + '/' + i2
#         print(path)
#         file_names = os.listdir(path)
#         for i3 in file_names:
#             path = path + '/' + i3

#             y_files = [f for f in os.listdir(path) if f.endswith('.npy')]

#             print(y_files)
#             break

#         # # 如果你只想获取文件而不包括子文件夹，可以使用列表推导式来筛选
#         # file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]

path = "/media/disk8T/zhn/dataset/NJU_CPOL_kdpRain/"
path_to = '/media/disk8T/zhn/Q3/datasets/rain/'
file_names = sorted(os.listdir(path))

count = 0
for file in file_names:
    new_path = path + file
    print(new_path)  # 找到一段连续降雨的记录
    frams = sorted([f for f in os.listdir(new_path) if f.endswith('.npy')])

    for fram in frams:
        new_new_path = new_path + '/' + fram
        data = np.load(new_new_path)
        Aev = np.sum(data > Ith)
        if Aev < Ath:
            # print('{}'.format(fram))
            continue
        print(Aev)
        name = path_to + "{}.txt".format(count)
        with open(name, "w") as fi:
            fi.write(new_new_path +'\n')
        count += 1

