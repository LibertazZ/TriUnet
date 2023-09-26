import os

import numpy as np

import torch
from torch.utils.data import Dataset

class DataSet_for_one_to_statistics(Dataset):
    def __init__(self, root='/media/disk8T/zhn/frunet/datasets/7.0km'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]

    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        assert len(lines) == 20
        zin = []
        zdr = []
        kdp = []
        zout = []
        for line in lines[0:10]:
            zin.append(np.load(line))
            zdr.append(np.load(line.replace('dBZ','ZDR')))
            kdp.append(np.load(line.replace('dBZ','KDP')))
        for line in lines[10:20]:
            zout.append(np.load(line))
        zin, zdr, kdp, zout = np.stack(zin, axis=0), np.stack(zdr, axis=0), np.stack(kdp, axis=0), np.stack(zout, axis=0)
        return zin, zdr, kdp, zout

    def __len__(self):
        return len(self.samples)
    

class DataSet_for_3km(Dataset):
    def __init__(self, root='/media/disk8T/zhn/frunet/datasets/3.0km'):
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
        zin = []
        zdr = []
        kdp = []
        zout = []
        for line in lines[0:10]:
            zin.append(np.load(line))
            zdr.append(np.load(line.replace('dBZ','ZDR')))
            kdp.append(np.load(line.replace('dBZ','KDP')))
        for line in lines[10:20]:
            zout.append(np.load(line))
        zin, zdr, kdp, zout = np.stack(zin, axis=0), np.stack(zdr, axis=0), np.stack(kdp, axis=0), np.stack(zout, axis=0)

        mmin, mmax =self.norm_param['dBZ']
        zin  = (zin - mmin) / (mmax - mmin)
        zout = (zout - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['ZDR']
        zdr  = (zdr - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['KDP']
        kdp  = (kdp - mmin) / (mmax - mmin)

        return zin, zdr, kdp, zout

    def __len__(self):
        return len(self.samples)
    

class DataSet_for_1km(Dataset):
    def __init__(self, root='/media/disk8T/zhn/frunet/datasets/1.0km'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-39.972763, 99.68255],

            'ZDR': [-27.439419, 23.563492],

            'KDP': [-16.00439, 46.97083],}
        
    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        zin = []
        zdr = []
        kdp = []
        zout = []
        for line in lines[0:10]:
            zin.append(np.load(line))
            zdr.append(np.load(line.replace('dBZ','ZDR')))
            kdp.append(np.load(line.replace('dBZ','KDP')))
        for line in lines[10:20]:
            zout.append(np.load(line))
        zin, zdr, kdp, zout = np.stack(zin, axis=0), np.stack(zdr, axis=0), np.stack(kdp, axis=0), np.stack(zout, axis=0)

        mmin, mmax =self.norm_param['dBZ']
        zin  = (zin - mmin) / (mmax - mmin)
        zout = (zout - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['ZDR']
        zdr  = (zdr - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['KDP']
        kdp  = (kdp - mmin) / (mmax - mmin)

        return zin, zdr, kdp, zout

    def __len__(self):
        return len(self.samples)
    
class DataSet_for_7km(Dataset):
    def __init__(self, root='/media/disk8T/zhn/frunet/datasets/7.0km'):
        # 定义好 image 的路径
        self.samples = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param = {

            'dBZ': [-21.096962, 67.22222],

            'ZDR': [-23.482878, 35.27572],

            'KDP': [-15.989306, 36.962673],}
        
    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        zin = []
        zdr = []
        kdp = []
        zout = []
        for line in lines[0:10]:
            zin.append(np.load(line))
            zdr.append(np.load(line.replace('dBZ','ZDR')))
            kdp.append(np.load(line.replace('dBZ','KDP')))
        for line in lines[10:20]:
            zout.append(np.load(line))
        zin, zdr, kdp, zout = np.stack(zin, axis=0), np.stack(zdr, axis=0), np.stack(kdp, axis=0), np.stack(zout, axis=0)

        mmin, mmax =self.norm_param['dBZ']
        zin  = (zin - mmin) / (mmax - mmin)
        zout = (zout - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['ZDR']
        zdr  = (zdr - mmin) / (mmax - mmin)

        mmin, mmax =self.norm_param['KDP']
        kdp  = (kdp - mmin) / (mmax - mmin)

        return zin, zdr, kdp, zout

    def __len__(self):
        return len(self.samples)
    

class DataSet_for_all(Dataset):
    def __init__(self, root='/media/disk8T/zhn/frunet/datasets/3.0km'):
        # 定义好 image 的路径
        self.samples1 = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param1 = {

            'dBZ': [-33.916622, 79.81039],

            'ZDR': [-32.29785, 31.435911],

            'KDP': [-16.620337, 52.45106],}
        root = root.replace('3.0km', '1.0km')
        self.samples2 = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param2 = {

            'dBZ': [-39.972763, 99.68255],

            'ZDR': [-27.439419, 23.563492],

            'KDP': [-16.00439, 46.97083],}
        root = root.replace('1.0km', '7.0km')
        self.samples3 = [root + '/' + f for f in os.listdir(root) if f.endswith('.txt')]
        self.norm_param3 = {

            'dBZ': [-21.096962, 67.22222],

            'ZDR': [-23.482878, 35.27572],

            'KDP': [-15.989306, 36.962673],}
        
        self.num = [len(self.samples1), len(self.samples1)+len(self.samples2), len(self.samples1) + len(self.samples2) + len(self.samples3)]

    def __getitem__(self, index):
        if 0<=index<self.num[0]:
            path = self.samples1[index]
            norm_param = self.norm_param1
            lambda_ = [1.,0, 0]
            # idx = 0.
        elif self.num[0] <= index < self.num[1] :
            path = self.samples2[index - self.num[0]]
            norm_param = self.norm_param2
            lambda_ = [0,1.,0]
            # idx = 1.
        elif self.num[1] <= index < self.num[2] :
            path = self.samples3[index - self.num[1]]
            norm_param = self.norm_param3
            lambda_ = [0,0,1.]
            # idx = 2.
        with open(path, 'r') as file:
            # 使用readlines()函数读取文件内容并保存为列表
            lines = file.readlines()
            lines = [line[:-1] for line in lines]
        zin = []
        zdr = []
        kdp = []
        zout = []
        for line in lines[0:10]:
            zin.append(np.load(line))
            zdr.append(np.load(line.replace('dBZ','ZDR')))
            kdp.append(np.load(line.replace('dBZ','KDP')))
        for line in lines[10:20]:
            zout.append(np.load(line))
        zin, zdr, kdp, zout = np.stack(zin, axis=0), np.stack(zdr, axis=0), np.stack(kdp, axis=0), np.stack(zout, axis=0)

        mmin, mmax =norm_param['dBZ']
        zin  = (zin - mmin) / (mmax - mmin)
        zout = (zout - mmin) / (mmax - mmin)

        mmin, mmax =norm_param['ZDR']
        zdr  = (zdr - mmin) / (mmax - mmin)

        mmin, mmax =norm_param['KDP']
        kdp  = (kdp - mmin) / (mmax - mmin)

        return zin, zdr, kdp, np.array(lambda_), zout

    def __len__(self):
        return self.num[2]


if __name__ == '__main__':
    dataset = DataSet_for_one_to_statistics()
    zin, zdr, kdp, zout = dataset[10556]
    print(zin.shape)
    print(zdr.shape)
    print(kdp.shape)
    print(zout.shape)