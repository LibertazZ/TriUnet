import os

from utils.utils import rain_sample

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

path = "/media/disk8T/zhn/dataset/NJU_CPOL_update2308/dBZ/3.0km/"
path_to = '/media/disk8T/zhn/frunet/datasets/3.0km/'
file_names = sorted(os.listdir(path))

count = 0
for file in file_names:
    new_path = path + file
    print(new_path)  # 找到一段连续降雨的记录
    frams = sorted([f for f in os.listdir(new_path) if f.endswith('.npy')])

    len_ = len(frams)
    # samples = len_ // 20 # 当前取到的样本对个数
    # for i in range(samples):
    #     zin, zout = frams[i*20:(i+1)*20][0:10], frams[i*20:(i+1)*20][10:20]
        
    #     # 使用 "w" 模式打开文件，这将创建一个新文件或覆盖已存在的同名文件
    #     if rain_sample(new_path, zout):
    #         file_name = path_to + "{}.txt".format(count)
    #         count += 1
    #         with open(file_name, "w") as file:
    #             for text_to_write in zin:
    #                 file.write(new_path + '/' + text_to_write+'\n')
    #             for text_to_write in zout:
    #                 file.write(new_path + '/' + text_to_write+'\n')
    if len_ >= 20:
        for i in range(len_-19):
            zin, zout = frams[i:i+20][0:10], frams[i:i+20][10:20]
            
            # 使用 "w" 模式打开文件，这将创建一个新文件或覆盖已存在的同名文件
            if rain_sample(new_path, zout):
                file_name = path_to + "{}.txt".format(count)
                count += 1
                with open(file_name, "w") as file:
                    for text_to_write in zin:
                        file.write(new_path + '/' + text_to_write+'\n')
                    for text_to_write in zout:
                        file.write(new_path + '/' + text_to_write+'\n')
