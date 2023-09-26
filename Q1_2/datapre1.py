# 可视化数据分布为直方图

import os
import numpy as np
import matplotlib.pyplot as plt

path = "/media/disk8T/zhn/dataset/NJU_CPOL_kdpRain/"

file_names = sorted(os.listdir(path))

min_ = float('inf')
max_ = float('-inf')



i = 0
arrays = []
for idx, file in enumerate(file_names):
    if idx <= i * 30 or idx >= (i+1) * 30:
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

print(min_)
print(max_)

print(len(arrays))
        # 将所有数组合并成一个大数组
combined_array = np.concatenate(arrays)

# 计算大数组的直方图
histogram, edges = np.histogram(combined_array[combined_array > 100], bins=50)

# 绘制直方图
plt.plot(edges[:-1], histogram, color='blue', label='Combined Distribution')

plt.title('30 data_dirs')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# 显示图形
plt.savefig('{}__.png'.format(i))
plt.show()


# # 计算大数组的直方图
# histogram, edges = np.histogram(combined_array[combined_array < 100], bins=50)

# # 绘制直方图
# plt.plot(edges[:-1], histogram, color='blue', label='Combined Distribution')

# plt.title('30 data_dirs')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True)

# # 显示图形
# plt.savefig('{}_.png'.format(i))
# plt.show()







