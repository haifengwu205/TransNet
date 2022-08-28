#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 06/12/2021 19:26
# @Author : hfw
# @Version：V 0.1
# @File : test_cnn.py
# @desc : test
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

import time


class Main:
    def __init__(self, opt):
        super(Main, self).__init__()
        self.file_ext = opt.test_file_ext
        self.img_root = opt.test_dataset_path
        self.save_root = opt.test_save_path
        self.img1_root = os.path.join(self.img_root, 'img2')
        self.img1_root = self.img1_root.replace('\\', '/')
        self.fusion_save_root = os.path.join(self.save_root, 'img2_gray')
        self.fusion_save_root = self.fusion_save_root.replace('\\', '/')
        if not os.path.exists(self.fusion_save_root):
            os.makedirs(self.fusion_save_root)
        self.img_names = os.listdir(self.img1_root)
        self.model_dir = opt.test_model
        # self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        self.device = torch.device("cpu")
        # print(self.device)
        self.opt = opt

    def func(self):

        begain_time = time.time()
        for img_name in self.img_names:
            print(img_name)
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(self.img1_root, img_name)
            fusion_savePath = os.path.join(self.fusion_save_root, name.split('-A')[0] + '.png')

            # img2_dir = os.path.join(self.img2_root, img_name)
            # decisionMap_savePath = os.path.join(self.decisionMap_save_root, img_name)
            # fusion_savePath = os.path.join(self.fusion_save_root, img_name)

            img1_dir = img1_dir.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')

            img1 = Image.open(img1_dir)

            input_transform = transforms.Compose([
                transforms.Grayscale(1),
            ])
            image_fusion = input_transform(img1)

            # image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
            image_fusion.save(fusion_savePath)
        end_time = time.time()
        print("Average time consumption：", (end_time - begain_time) / 20)


if __name__ == '__main__':
    begain_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, default='./dataset/Tsai/',
                        help='path of test dataset path in train')
    parser.add_argument('--test_save_path', type=str, default='./dataset/Tsai/',
                        help='path of save path in train')
    parser.add_argument('--test_file_ext', type=str, default='.png',
                        help='[.jpg, .png, .tif]')
    parser.add_argument('--test_model', type=str, default='./models/v38_drpl_benchmark_O5_K9_D2_P4_lr0.001_bs32/net_019.pkl', help='path of save path in train')
    # parser.add_argument('--my_model', type=str, default='./ckpt/v38_benchmark256_O1_K9_D2_P4_lr0.001_BCEWLoss_bs16/net_001.pkl', help='path of save path in train')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--dtmOrder', type=int, default=5, help='The order of DTM')
    parser.add_argument('--dtmConvKS', type=int, default=9, help='The kernel size of DTMConv')
    parser.add_argument('--dtmConvLayers', type=int, default=21, help='The number of DTMConv layer')
    parser.add_argument('--dtmConvChannel_in', type=int, default=2, help='The number of TMConv layer')
    parser.add_argument('--dtmConvChannel_out', type=int, default=2, help='The number of DTMConv layer')

    # Main(parser.parse_args()).func_model()
    Main(parser.parse_args()).func()
    end_time = time.time()
    print("Average time consumption：", (end_time - begain_time) / 20)