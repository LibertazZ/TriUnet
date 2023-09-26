import os

import numpy as np

from torch.utils.data import Dataset

class DataSet_for_rain(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/3.0km'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-39.972763, 99.68255],

            'ZDR': [-32.29785, 35.27572]}

    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        assert len(lines) == 1
        path = lines[0]
        zin_3km = np.load(path)
        zdr_3km = np.load(path.replace('dBZ','ZDR'))
        zin_1km = np.load(path.replace('3km','1km'))
        zdr_1km = np.load(path.replace('3km','1km').replace('dBZ','ZDR'))
        zin_7km = np.load(path.replace('3km','7km'))
        zdr_7km = np.load(path.replace('3km','7km').replace('dBZ','ZDR'))

        rain = np.load(path.replace('_update2308/dBZ/3.0km/', '_kdpRain/'))
        
        zin, zdr, rain = np.stack((zin_3km, zin_1km, zin_7km), axis=0), np.stack((zdr_3km, zdr_1km, zdr_7km), axis=0), np.expand_dims(rain, axis=0)
        # zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])
        # zdr = (zdr - self.norm_param['ZDR'][0])/(self.norm_param['ZDR'][1]-self.norm_param['ZDR'][0])
        return zin, zdr, rain / 1225.708 # 归一化

    def __len__(self):
        return len(self.samples)
    

class DataSet_for_rain_2(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/rain'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-39.972763, 99.68255],

            'ZDR': [-32.29785, 35.27572]}


    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        assert len(lines) == 1
        path_rain = lines[0]
        path = path_rain.replace( '_kdpRain/','_update2308/dBZ/3.0km/')
        zin_3km = np.load(path)
        zdr_3km = np.load(path.replace('dBZ','ZDR'))
        zin_1km = np.load(path.replace('3km','1km'))
        zdr_1km = np.load(path.replace('3km','1km').replace('dBZ','ZDR'))
        zin_7km = np.load(path.replace('3km','7km'))
        zdr_7km = np.load(path.replace('3km','7km').replace('dBZ','ZDR'))

        rain = np.load(path_rain)
        
        zin, zdr, rain = np.stack((zin_3km, zin_1km, zin_7km), axis=0), np.stack((zdr_3km, zdr_1km, zdr_7km), axis=0), np.expand_dims(rain, axis=0)
        
        zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])
        zdr = (zdr - self.norm_param['ZDR'][0])/(self.norm_param['ZDR'][1]-self.norm_param['ZDR'][0])

        return zin, zdr, rain / 1225.708 # 归一化

    def __len__(self):
        return len(self.samples)
        

        
class DataSet_for_rain_3(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/rain'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-33.916622, 79.81039],

            'ZDR': [-32.29785, 31.435911],}


    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        assert len(lines) == 1
        path_rain = lines[0]
        path = path_rain.replace( '_kdpRain/','_update2308/dBZ/3.0km/')
        zin_3km = np.load(path)
        zdr_3km = np.load(path.replace('dBZ','ZDR'))

        rain = np.load(path_rain)
        
        zin, zdr, rain = np.expand_dims(zin_3km, axis=0), np.expand_dims(zdr_3km, axis=0), np.expand_dims(rain, axis=0)
        
        zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])
        zdr = (zdr - self.norm_param['ZDR'][0])/(self.norm_param['ZDR'][1]-self.norm_param['ZDR'][0])

        return zin, zdr, rain / 1225.708 # 归一化

    def __len__(self):
        return len(self.samples)
    

class DataSet_for_rain_4(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/3.0km'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-33.916622, 79.81039],

            'ZDR': [-32.29785, 31.435911],}


    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        assert len(lines) == 1
        path = lines[0]
        
        zin_3km = np.load(path)
        zdr_3km = np.load(path.replace('dBZ','ZDR'))
        path_rain = path.replace( '_update2308/dBZ/3.0km/','_kdpRain/')
        rain = np.load(path_rain)
        
        zin, zdr, rain = np.expand_dims(zin_3km, axis=0), np.expand_dims(zdr_3km, axis=0), np.expand_dims(rain, axis=0)
        
        zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])
        zdr = (zdr - self.norm_param['ZDR'][0])/(self.norm_param['ZDR'][1]-self.norm_param['ZDR'][0])

        return zin, zdr, rain/1225.708 # 归一化

    def __len__(self):
        return len(self.samples)




if __name__ == '__main__':
    dataset = DataSet_for_rain_2()
    print(len(dataset))
    zin, zdr, rain = dataset[1]
    print(zin.shape)
    print(zdr.shape)
    print(rain.shape)