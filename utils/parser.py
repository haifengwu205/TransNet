#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 07/06/2021 22:11
# @Author : hfw
# @Version：V 0.1
# @File : parser.py
# @desc :

import argparse


def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epoch', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='train learning rate')
    parser.add_argument('--batch_size_label', type=int, default=32, help='train batch size')
    parser.add_argument('--batch_size_nolabel', type=int, default=4, help='train batch size')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--train_test_dataset_path', type=str, default='./dataset/Lytro/', help='path of test dataset path in train')
    parser.add_argument('--train_test_save_path', type=str, default='./results/train/v34/O1_K13_D2_P12_RL_bs32/', help='path of save path in train')
    parser.add_argument('--save_models', type=str, default='./ckpt/v34/O1_K13_D2_P12_RL_bs32/', help='path of save path in train')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--dtmOrder', type=int, default=5, help='The order of DTM')
    parser.add_argument('--dtmConvKS', type=int, default=9, help='The kernel size of DTMConv')
    parser.add_argument('--dtmConvLayers', type=int, default=21, help='The number of DTMConv layer')
    parser.add_argument('--dtmConvChannel_in', type=int, default=2, help='The number of DTMConv layer')
    parser.add_argument('--dtmConvChannel_out', type=int, default=2, help='The number of DTMConv layer')
    parser.add_argument('--OFKS', type=int, default=9, help='The order of DTM')
    parser.add_argument('--IsDilation', action='store_true', default=True, help='is or not dilation')
    parser.add_argument('--OFdilation', type=int, default=2, help='The order of DTM')
    # parser.add_argument('--contrastHead', action='store_true', default=True, help='The order of DTM')

    return parser.parse_args()