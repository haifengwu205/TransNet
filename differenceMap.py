#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 17/01/2022 19:49
# @Author : hfw
# @Version：V 0.1
# @File : differenceMap.py

import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
# import DTMConv
import torch
import numpy as np

class Solution:
    def __init__(self, opt):
        super(Solution, self).__init__()
        self.file_ext = opt.test_file_ext
        self.img_root = opt.test_dataset_path
        self.img2_root = opt.source_img2
        self.save_root = opt.test_save_path
        self.img1_root = os.path.join(self.img_root)
        self.img2_root = os.path.join(self.img2_root)
        self.img1_root = self.img1_root.replace('\\', '/')
        self.img2_root = self.img2_root.replace('\\', '/')
        self.dif_root = os.path.join(self.save_root)
        self.dif_root = self.dif_root.replace('\\', '/')
        if not os.path.exists(self.dif_root):
            os.makedirs(self.dif_root)
        self.img_names = os.listdir(self.img1_root)
        self.opt = opt

    def func(self):
        for img_name in self.img_names:
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(self.img1_root, img_name)
            img2_dir = os.path.join(self.img2_root, name.split('-A')[0] + "-B" + self.file_ext)
            dif_savePath = os.path.join(self.dif_root, name.split('-A')[0] + '.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            dif_savePath = dif_savePath.replace('\\', '/')

            # img1 = np.array(Image.open(img1_dir).convert('L'))
            # img2 = np.array(Image.open(img2_dir).convert('L'))

            img1 = np.array(Image.open(img1_dir))
            img2 = np.array(Image.open(img2_dir))

            dif = img1 - img2
            # dif = abs(dif)
            # dif[dif >= 0.1] = 255
            # dif[dif < 0.1] = 0
            dif = np.clip(dif, 0, 255)

            # dif = 255 - dif
            dif = Image.fromarray(dif.astype(np.uint8))
            dif.save(dif_savePath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img2', type=str, default='./dataset/Nature/img2/',
                        help='path of save path in train')
    parser.add_argument('--test_dataset_path', type=str, default='./results/test/v38_drpl_benchmark_O5_K9_D2_P4_lr0.001_bs32_Nature_1_m19/fusion/',
                        help='path of test dataset path in train')
    parser.add_argument('--test_save_path', type=str, default='./results/differenceMap_v38/Nature/',
                        help='path of save path in train')
    parser.add_argument('--test_file_ext', type=str, default='.png',
                        help='[.jpg, .png, .tif]')

    Solution(parser.parse_args()).func()