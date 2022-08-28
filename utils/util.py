#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 08/06/2021 09:18
# @Author : hfw
# @Version：V 0.1
# @File : utils.py
# @desc :
import glob
import os
import re
import torch

def findLastCheckpoint(save_dir):
    # 从模型的路径中查找
    file_list = glob.glob(os.path.join(save_dir, 'net_*.pkl'))
    # 如果file_list不为空，说明有已训练的模型，查找最新的模型
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*net_(.*).pkl.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    # 若file_list为空，说明无模型，让初始epoch为0，即从第0个开始训练
    else:
        initial_epoch = 0
    return initial_epoch


def homogeneousRe(img1, img2, gpu=False, device=None):
    # print(device)
    homoRe = img1 - img2
    homoRe = homoRe.to(device)


    # use GPU
    if gpu is True:
        homoRe[homoRe >= torch.tensor(0.0004).to(device)] = torch.ones(1).to(device)
        homoRe[homoRe < torch.tensor(0.0004).to(device)] = torch.zeros(1).to(device)

        homoRe = torch.ones(1).to(device) - homoRe

        return homoRe
    # use CPU
    else:
        homoRe[homoRe >= torch.tensor(0.0004)] = 1
        homoRe[homoRe < torch.tensor(0.0004)] = 0

        homoRe = 1 - homoRe

        return homoRe





