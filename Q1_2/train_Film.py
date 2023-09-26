import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

import os
import argparse

from FURENet.FURENet import FURENet_Film
from datasets.dataloader import *
from utils.utils import progress_bar, create_logger


batch_size = 32
gpu = '3'

model_dir = '/media/disk8T/zhn/frunet/log_2'

logger = create_logger("{0}/logs_ST.txt".format(model_dir), 'info')

# gpu锟斤拷锟斤拷
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)

# 锟斤拷锟斤拷预锟斤拷锟斤拷
dataset = DataSet_for_all()

train_size = int(0.9 * len(dataset))  # 80% 用于训练集
# val_size = len(dataset) - train_size  # 剩余部分用于验证集
train_dataset, val_dataset = Subset(dataset, list(range(train_size))), Subset(dataset, list(range(train_size, len(dataset))))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size)

# Model
print('==> Building model..')
# if args.network == 'resnet18':
#     print('resnet18')
#     net = resnet18()
#     net = net.to(device)
# elif args.network == 'vgg19':
#     print('vgg19')
#     net = VGG19()
#     net = net.to(device)
# elif args.network == 'googlenet':
#     print('googlenet')
#     net = GoogLeNet()
#     net = net.to(device)
# elif args.network == 'densenet121':
#     print('densenet121')
#     net = DenseNet121()
#     net = net.to(device)

net = FURENet_Film(10)  #####################################################################
net = net.cuda()
print(net)

# net.load_state_dict(torch.load('/media/disk8T/zhn/frunet/log/50.pth'))

end_epochs = 100
#锟斤拷锟斤拷训锟斤拷锟斤拷锟斤拷
def BMSE(zout, outputs):
    return torch.pow((zout-outputs), 2)

criterion = BMSE
optimizer = optim.AdamW(net.parameters(), betas = (0.9, 0.999), weight_decay = 0.03, lr = 0.001)


# 训锟斤拷锟斤拷锟斤拷
def train(epoch):
    print('\nEpoch: %d' % epoch)
    logger.info('\n=> Training Epoch #{}'.format(epoch))
    net.train()
    total = 0
    for batch_idx, (zin, zdr, kdp, lambda_, zout) in enumerate(trainloader):
        # lr = learning_rate(epoch + (batch_idx+1)/len(trainloader))  # 闅廱atch_idx鏀瑰彉
        # print(lr)
        # optimizer.param_groups[0].update(lr=lr)
        zin, zdr, kdp, lambda_, zout = zin.cuda(), zdr.cuda(), kdp.cuda(), lambda_.float().cuda(), zout.cuda()
        optimizer.zero_grad()
        outputs = net(zin, zdr, kdp, lambda_)
        loss = criterion(zout, outputs)  # (N, 10, 256, 256)
        loss_log = torch.mean(loss)
        loss_log.backward()
        optimizer.step()

        loss_min = torch.min(loss).detach().cpu()
        loss_max = torch.max(loss).detach().cpu()
        loss_mean = loss_log.detach().cpu()
        outputs_min = torch.min(outputs).detach().cpu()
        outputs_max = torch.max(outputs).detach().cpu()
        total += len(zin)

        progress_bar(batch_idx, len(trainloader), 'loss_min: %.3f |loss_max: %.3f |loss_mean: %.3f |outputs_min: %.3f |outputs_max: %.3f |(%d/%d)'
                     % (loss_min, loss_max, loss_mean, outputs_min, outputs_max, total, len(train_dataset)))
    logger.info('Epoch: {0}, loss_min: {1:.4f}, loss_max: {2:.4f}, loss_mean: {3:.4f}, outputs_min: {4:.4f}, outputs_min: {5:.4f}'.format(epoch, loss_min, loss_max, loss_mean, outputs_min, outputs_max))    


def test(epoch):
    net.eval()
    test_loss = 0
    total = 0
    loss_min = float('inf')
    loss_max = float('-inf')
    outputs_min = float('inf')
    outputs_max = float('-inf')

    for batch_idx, (zin, zdr, kdp, lambda_, zout) in enumerate(valloader):
        # lr = learning_rate(epoch + (batch_idx+1)/len(trainloader))  # 闅廱atch_idx鏀瑰彉
        # print(lr)
        # optimizer.param_groups[0].update(lr=lr)
        zin, zdr, kdp, lambda_, zout = zin.cuda(), zdr.cuda(), kdp.cuda(), lambda_.float().cuda(), zout.cuda()
        outputs = net(zin, zdr, kdp, lambda_)
        loss = criterion(zout, outputs)  # (10, 256, 256)

        lossmin = torch.min(loss).detach().cpu()
        lossmax = torch.max(loss).detach().cpu()
        losssum = torch.mean(loss).detach().cpu()* len(zin)
        outputsmin = torch.min(outputs).detach().cpu()
        outputsmax = torch.max(outputs).detach().cpu()
        if lossmin < loss_min:
            loss_min = lossmin
        if outputsmin < outputs_min:
            outputs_min = outputsmin
        if lossmax > loss_max:
            loss_max = lossmax
        if outputsmax > outputs_max:
            outputs_max = outputsmax
        
        test_loss += losssum
        total += len(zin)

        progress_bar(batch_idx, len(valloader), 'loss_min: %.3f |loss_max: %.3f |loss_mean: %.3f |outputs_min: %.3f |outputs_max: %.3f'
                     % (loss_min, loss_max, test_loss/total, outputs_min, outputs_max))

    # 锟斤拷锟斤拷模锟斤拷.
    logger.info('Epoch: {0}, loss_min: {1:.4f}, loss_max: {2:.4f}, loss_mean: {3:.4f}, outputs_min: {4:.4f}, outputs_min: {5:.4f}'.format(epoch, loss_min, loss_max, test_loss/total, outputs_min, outputs_max))  
    if epoch % 10 == 0 or epoch >= end_epochs*0.75:
        print('Saving..')
        state = {
            'net': net.state_dict(),
        }
        torch.save(state, '{}/{}.pth'.format(model_dir, epoch))

for epoch in range(1, end_epochs+1):
    train(epoch)
    test(epoch)
    # scheduler.step()
