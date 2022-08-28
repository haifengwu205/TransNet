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
        self.img1_root = os.path.join(self.img_root, 'img1')
        self.img2_root = os.path.join(self.img_root, 'img2')
        self.img1_root = self.img1_root.replace('\\', '/')
        self.img2_root = self.img2_root.replace('\\', '/')
        self.decisionMap_save_root = os.path.join(self.save_root, 'decisionmap')
        self.fusion_save_root = os.path.join(self.save_root, 'fusion')
        self.decisionMap_save_root = self.decisionMap_save_root.replace('\\', '/')
        self.fusion_save_root = self.fusion_save_root.replace('\\', '/')
        if not os.path.exists(self.decisionMap_save_root):
            os.makedirs(self.decisionMap_save_root)
        if not os.path.exists(self.fusion_save_root):
            os.makedirs(self.fusion_save_root)
        self.img_names = os.listdir(self.img1_root)
        self.model_dir = opt.test_model
        # self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        self.device = torch.device("cpu")
        # print(self.device)
        self.opt = opt

    def func(self):
        model = torch.load(self.model_dir, map_location=self.device)
        # model.isTrain = False

        total_params = sum(p.numel() for p in model.parameters())
        print('Total parameters: %.6fM (%d)' % (total_params / 1e6, total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print('Total_trainable_params: %.6fM (%d)' % (total_trainable_params / 1e6, total_trainable_params))

        print(model)
        model.eval()
        torch.set_grad_enabled(False)
        begain_time = time.time()
        for img_name in self.img_names:
            print(img_name)
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(self.img1_root, img_name)
            img2_dir = os.path.join(self.img2_root, name.split('-A')[0] + "-B" + self.file_ext)
            decisionMap_savePath = os.path.join(self.decisionMap_save_root, name.split('-A')[0] + '.png')
            fusion_savePath = os.path.join(self.fusion_save_root, name.split('-A')[0] + '.png')

            # img2_dir = os.path.join(self.img2_root, img_name)
            # decisionMap_savePath = os.path.join(self.decisionMap_save_root, img_name)
            # fusion_savePath = os.path.join(self.fusion_save_root, img_name)

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')

            img1 = Image.open(img1_dir)
            img2 = Image.open(img2_dir)

            input_transform = transforms.Compose([
                # transforms.Resize((320, 480)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
            ])
            img1_tensor = input_transform(img1).unsqueeze(0)
            img2_tensor = input_transform(img2).unsqueeze(0)
            image_device = torch.cat((img1_tensor, img2_tensor), dim=1).to(self.device)

            # contrast_focus, output_focus = model(image_device)
            output_focus = model(image_device)

            output = torch.softmax(output_focus, dim=1)
            # output = torch.sigmoid(output_focus)
            image_mask = torch.max(output.cpu(), 1)[1]

            # save decisionMap
            img_decisionMap = np.multiply(image_mask.squeeze(0).numpy(), 255)
            img_decisionMap = 255 - img_decisionMap
            # img_decisionMap = img_decisionMap
            img_decisionMap = Image.fromarray(img_decisionMap.astype(np.uint8))
            img_decisionMap.save(decisionMap_savePath)

            # image fusion
            image_mask = image_mask.numpy()
            image_mask = 1 - image_mask
            image_mask = np.transpose(image_mask, (1, 2, 0))

            if len(np.array(img1).shape) == 2:
                image_mask = image_mask.squeeze()

            if len(np.array(img1).shape) == 3:
                image_mask = np.repeat(image_mask, 3, 2)

            image_fusion = img1 * image_mask + img2 * (1 - image_mask)
            # image_fusion[image_fusion < 0] = 0
            # image_fusion[image_fusion > 255] = 255
            image_fusion.clip(0, 255)
            image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
            image_fusion.save(fusion_savePath)
        end_time = time.time()
        print("Average time consumption：", (end_time - begain_time) / 20)

    # def func_model(self):
    #     model = torch.load(self.model_dir, map_location=self.device)
    #     origin_model_dict = model.state_dict()
    #     for k, v in origin_model_dict.items():
    #         print(k)
    #     # model.isTrain = False
    #
    #     my_model = myModel.Network(2, 2, self.device)
    #     my_model_dict = my_model.state_dict()
    #     # for k, v in my_model_dict.items():
    #     #     print(k)
    #
    #     origin_model_dict = {k: v for k, v in origin_model_dict.items() if k in my_model_dict}
    #     # for k, v in origin_model_dict.items():
    #     #     print(k)
    #
    #     my_model_dict.update(origin_model_dict)
    #     my_model.load_state_dict(my_model_dict)
    #
    #     total_params = sum(p.numel() for p in my_model.parameters())
    #     print('Total parameters: %.6fM (%d)' % (total_params / 1e6, total_params))
    #     total_trainable_params = sum(
    #         p.numel() for p in my_model.parameters() if p.requires_grad)
    #     print('Total_trainable_params: %.6fM (%d)' % (total_trainable_params / 1e6, total_trainable_params))
    #
    #     # print(model)
    #     begain_time = time.time()
    #     for img_name in self.img_names:
    #         print(img_name)
    #         name, suffix = img_name.split('.')
    #         img1_dir = os.path.join(self.img1_root, img_name)
    #         img2_dir = os.path.join(self.img2_root, name.split('-A')[0] + "-B" + self.file_ext)
    #         decisionMap_savePath = os.path.join(self.decisionMap_save_root, name.split('-A')[0] + '.png')
    #         fusion_savePath = os.path.join(self.fusion_save_root, name.split('-A')[0] + '.png')
    #
    #         img1_dir = img1_dir.replace('\\', '/')
    #         img2_dir = img2_dir.replace('\\', '/')
    #         decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
    #         fusion_savePath = fusion_savePath.replace('\\', '/')
    #
    #         img1 = Image.open(img1_dir)
    #         img2 = Image.open(img2_dir)
    #
    #         input_transform = transforms.Compose([
    #             transforms.Grayscale(1),
    #             transforms.ToTensor(),
    #             # transforms.Normalize(mean=[0.5], std=[0.5]),
    #         ])
    #
    #         img1_tensor = input_transform(img1).unsqueeze(0)
    #         img2_tensor = input_transform(img2).unsqueeze(0)
    #
    #         image_device = torch.cat((img1_tensor, img2_tensor), dim=1).to(self.device)
    #
    #         # contrast_focus, output_focus = my_model(image_device)
    #         output_focus = my_model(image_device)
    #
    #         output = torch.softmax(output_focus, dim=1)
    #         image_mask = torch.max(output.cpu(), 1)[1]
    #
    #         # save decisionMap
    #         img_decisionMap = np.multiply(image_mask.squeeze(0).numpy(), 255)
    #         img_decisionMap = 255 - img_decisionMap
    #         img_decisionMap = Image.fromarray(img_decisionMap.astype(np.uint8))
    #         img_decisionMap.save(decisionMap_savePath)
    #
    #         # image fusion
    #         image_mask = image_mask.numpy()
    #         image_mask = 1 - image_mask
    #         image_mask = np.transpose(image_mask, (1, 2, 0))
    #
    #         if len(np.array(img1).shape) == 2:
    #             image_mask = image_mask.squeeze()
    #
    #         if len(np.array(img1).shape) == 3:
    #             image_mask = np.repeat(image_mask, 3, 2)
    #
    #         image_fusion = img1 * image_mask + img2 * (1 - image_mask)
    #         # image_fusion[image_fusion < 0] = 0
    #         # image_fusion[image_fusion > 255] = 255
    #         image_fusion.clip(0, 255)
    #         image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
    #         image_fusion.save(fusion_savePath)
    #     end_time = time.time()
    #     print("Average time consumption：", (end_time - begain_time) / 20)


    def funbc(self):
        model = torch.load(self.model_dir, map_location=self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print('Total parameters: %.6fM (%d)' % (total_params / 1e6, total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print('Total_trainable_params: %.6fM (%d)' % (total_trainable_params / 1e6, total_trainable_params))

        print(model)

        for img_name in self.img_names:
            print(img_name)
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(self.img1_root, img_name)
            img2_dir = os.path.join(self.img2_root, name.split('-A')[0] + "-B" + self.file_ext)
            decisionMap_savePath = os.path.join(self.decisionMap_save_root, name.split('-A')[0] + '.png')
            fusion_savePath = os.path.join(self.fusion_save_root, name.split('-A')[0] + '.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')

            img1 = Image.open(img1_dir)
            img2 = Image.open(img2_dir)

            input_transform = transforms.Compose([

                transforms.Grayscale(1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            img1_tensor = input_transform(img1).unsqueeze(0)
            img2_tensor = input_transform(img2).unsqueeze(0)

            image_device = torch.cat((img1_tensor, img2_tensor), dim=1).to(self.device)

            # contrast_focus, output_focus = model(image_device)
            output_focus = model(image_device)

            output_focus = torch.sigmoid(output_focus)

            output1 = output_focus[:, 0, :, :].cpu().detach().numpy()
            output2 = output_focus[:, 0, :, :].cpu().detach().numpy()

            image_mask = (output1 + 1 - output2) / 2

            # save decisionMap
            img_decisionMap = np.multiply(output1.squeeze(), 255)
            # img_decisionMap = 255 - img_decisionMap
            img_decisionMap = Image.fromarray(img_decisionMap.astype(np.uint8))
            img_decisionMap.save(decisionMap_savePath)

            image_mask = np.transpose(image_mask, (1, 2, 0))

            # image fusion
            if len(np.array(img1).shape) == 2:
                image_mask = image_mask.squeeze()

            if len(np.array(img1).shape) == 3:
                image_mask = np.repeat(image_mask, 3, 2)

            image_fusion = img1 * image_mask + img2 * (1 - image_mask)
            # image_fusion[image_fusion < 0] = 0
            # image_fusion[image_fusion > 255] = 255
            image_fusion.clip(0, 255)
            image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
            image_fusion.save(fusion_savePath)


if __name__ == '__main__':
    begain_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, default='./dataset/Nature/',
                        help='path of test dataset path in train')
    parser.add_argument('--test_save_path', type=str, default='./results/test/v38_drpl_benchmark_O5_K9_D2_P4_lr0.001_bs32_Nature_1_m19/',
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