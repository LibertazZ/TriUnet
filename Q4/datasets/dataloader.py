import os

import numpy as np

from torch.utils.data import Dataset
        
        
class DataSet_for_ZH(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/rain'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-33.916622, 79.81039],}


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

        rain = np.load(path_rain)
        
        zin, rain = np.expand_dims(zin_3km, axis=0), np.expand_dims(rain, axis=0)
        
        zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])

        return zin, rain / 1225.708 # 归一化

    def __len__(self):
        return len(self.samples)
    
class DataSet_for_all(Dataset):
    def __init__(self, root='/media/disk8T/zhn/Q3/datasets/rain'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {
            'dBZ': [-33.916622, 79.81039],

            'ZDR': [-32.29785, 31.435911],

            'KDP': [-16.620337, 52.45106],}


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
        kdp_3km = np.load(path.replace('dBZ','KDP'))

        rain = np.load(path_rain)
        
        zin, zdr, kdp, rain = np.expand_dims(zin_3km, axis=0), np.expand_dims(zdr_3km, axis=0),np.expand_dims(kdp_3km, axis=0),np.expand_dims(rain, axis=0)
        
        zin = (zin - self.norm_param['dBZ'][0])/(self.norm_param['dBZ'][1]-self.norm_param['dBZ'][0])
        zdr = (zdr - self.norm_param['ZDR'][0])/(self.norm_param['ZDR'][1]-self.norm_param['ZDR'][0])
        kdp = (kdp - self.norm_param['KDP'][0])/(self.norm_param['KDP'][1]-self.norm_param['KDP'][0])

        return zin, zdr, kdp, rain / 1225.708 # 归一化

    def __len__(self):
        return len(self.samples)
    





if __name__ == '__main__':
    pass