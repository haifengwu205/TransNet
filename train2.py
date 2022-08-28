#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 07/06/2021 15:39
# @Author : hfw
# @Version：V 0.1
# @File : train.py
# @desc :
import argparse

from utils import parser, data_transforms_label, data_transforms_nolabel, util, ssimLoss, gradientLoss, lp_lssim_loss
import torchvision.transforms as transforms
from dataset import datasetList
from torch.utils.data import DataLoader
import pandas as pd
import torch
from ckpt import modelX as model
import os
import datetime
from PIL import Image
import numpy as np
from loss import focalLoss
# from pytorch_loss import FocalLossV3
from loss.decoder_contrast1 import dec_deeplabv3_contrast

# opt = parser.Parser()
def main(opt):
    #************************************************ 1 **********************************************
    image_size = opt.image_size
    input_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    target_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    gray_transform = transforms.Compose([
        transforms.Grayscale(1)
    ])

    co_transform_label = data_transforms_label.Compose([
        data_transforms_label.RandomVerticalFlip(),
        data_transforms_label.RandomHorizontalFlip(),
    ])

    co_transform_nolabel = data_transforms_nolabel.Compose([
        data_transforms_nolabel.RandomVerticalFlip(),
        data_transforms_nolabel.RandomHorizontalFlip(),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # ************************************************ 2 **********************************************
    # ================================= load train data =============================
    # train_set_label = datasetList.data_set(csv_path='./dataset/dataset_csv/data_all_v4.csv',
    train_set_label = datasetList.data_set(csv_path='./dataset/dataset_csv/data_all_v4.csv',
                                     input_transform=input_transform,
                                     target_transform=target_transform,
                                     gray_transform=gray_transform,
                                     co_transform=co_transform_label,
                                     hasLabel=False,
                                     gpu=True,
                                     device=device)
    print(len(train_set_label))
    train_loader_label = DataLoader(train_set_label, batch_size=opt.batch_size_label, shuffle=True, num_workers=opt.workers)
    # # ================================= load train data =============================
    # train_set_nolabel = datasetList.data_set(csv_path='./dataset/dataset_csv/data_all_v3_nolabel.csv',
    #                                        input_transform=input_transform,
    #                                        target_transform=target_transform,
    #                                        gray_transform=gray_transform,
    #                                        co_transform=co_transform_nolabel,
    #                                        hasLabel=False)
    # print(len(train_set_nolabel))
    # train_loader_nolabel = DataLoader(train_set_nolabel, batch_size=opt.batch_size_nolabel, shuffle=False, num_workers=opt.workers)
    # ================================= train note =============================
    df = pd.DataFrame(columns=['time', 'step', 'iteration', 'loss'])
    df.to_csv("./results/train_note/loss.csv", index=False)
    # ====================================== train ==================================

    print(device)
    model_net = model.Network(opt.dtmConvChannel_in, opt.dtmConvChannel_out, opt, device)
    print("1: ", model_net)
    # 查找最新的一个模型，从最新的一个模型继续训练，这样能避免每次暂停后从第一个开始训练
    # pkl: 后缀
    # l: low-frequecy
    # n: 有归一化  nn: 无归一化
    # p5: p = 5
    save_dir = opt.save_models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    initial_epoch = util.findLastCheckpoint(save_dir=save_dir)  # load the last ckpt in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # ckpt.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model_net = torch.load(os.path.join(save_dir, 'net_%03d.pkl' % initial_epoch))
    print("2: ", model_net)
    model_net.to(device)
    model_net.train()
    loss_function = torch.nn.BCEWithLogitsLoss().to(device)
    # loss_function = torch.nn.BCELoss().to(device)
    # loss_function = ND_Crossentropy.WeightedCrossEntropyLossV2()
    # loss_function = loss_contrast.ContrastCELoss()
    # loss_function = torch.nn.MSELoss(reduction='mean')
    # loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = dtt_loss.DttLoss()
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.learning_rate)  # 0.92-0.99
    # optimizer = torch.optim.SGD(model_net.parameters(), lr=opt.learning_rate)  # 0.92-0.99
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        # , factor=0.1, patience=5,
        #          verbose=False, threshold=1e-4, threshold_mode='rel',
        #          cooldown=0, min_lr=0, eps=1e-8)
    # train(opt, model_net, opt.Epoch, initial_epoch, train_loader_label, train_loader_nolabel, loss_function, optimizer, scheduler, device)
    train_label(opt, model_net, opt.Epoch, initial_epoch, train_loader_label, loss_function, optimizer, scheduler, device)


def train_label(opt, model_net, EPOCH, initial_epoch, train_loader_label, loss_function, optimizer, scheduler, device):
    for epoch in range(initial_epoch, EPOCH):
        # 1.Train
        train_loss = []
        train_accuracy = 0
        loss_log = 0
        for i, data1 in enumerate(train_loader_label):
            # data1_img1, data1_img2, label, label_onehot, homoRe = data1
            data1_img1, data1_img2, label, label_onehot = data1

            # img_foreground, img_background, img_groundTruth, label_one_hot, label
            # data1_img1, data1_img2, label, label_onehot, homoRe = data1_img1.to(device), data1_img2.to(device), label.to(device), label_onehot.to(device), homoRe.to(device)
            data1_img1, data1_img2, label, label_onehot = data1_img1.to(device), data1_img2.to(device), label.to(device), label_onehot.to(device)

            # print(len(data1[0][0]))
            # print(len(data2))

            data1_img = torch.cat((data1_img1, data1_img2), 1)
            # 1. forward propagation
            # outputs_focus, contrast_focus = model_net(data1_img)
            # outputs_focus, focus_maps = model_net(data1_img)
            outputs_focus = model_net(data1_img)

            # outputs_focus = torch.softmax(outputs_focus, dim=1)
            # 2. compute loss
            # loss1 = loss_function(outputs_focus, label_onehot)
            # output = torch.softmax(outputs_focus, dim=1)
            output = torch.sigmoid(outputs_focus)
            pred_out = torch.max(output, 1)[1]
            # pred_out_device = pred_out.unsqueeze(1)
            # gloss = gradient_loss.gradient_loss_merge(pred_out_device.float(), label, device)
            # 像素对比学习
            # loss2 = loss_contrast.ContrastCELoss()(contrast_focus, outputs_focus, label)
            # 区域对比学习
            # loss2 = dec_deeplabv3_contrast(channels=64, num_classes=2, inner_planes=64, device=device)(contrast_focus, outputs_focus)

            # loss = loss1 + gloss
            # loss2 = torch.nn.MSELoss()(output, label_onehot)
            # loss = complementEntropy.ComplementEntropy(device)(outputs_focus, label_onehot)
            # loss1 = loss_function(outputs_focus, label_onehot)
            # print(outputs_focus.shape, label_onehot.shape)
            loss1 = loss_function(outputs_focus, label_onehot)
            # loss2 = FocalLossV3()(outputs_focus, label_onehot)
            # loss2 = loss_function(focus_maps, label_onehot) * 0.1
            # loss2 = loss_function(focus_maps, label_onehot)
            # loss2 = dec_deeplabv3_contrast(channels=64, num_classes=2, inner_planes=64, device=device)(contrast_focus, torch.sigmoid(outputs_focus))
            loss = loss1
            # loss = loss1 + 0.01 * loss2
            # loss = loss1 + loss2
            # 3. clear gradients
            optimizer.zero_grad()
            # 4. backward propagation
            loss.backward()
            # 5. weight optimizations
            optimizer.step()
            # 6. train loss
            train_loss.append(loss.item())
            # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "label_loss: ", loss1_label,
            #       "GT_loss: ", loss2_label, "recon_loss: ", loss4_label, "loss2: ", loss2)
            if i % 10 == 0:
                # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "loss2: ", loss2)
                # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "loss2: ", loss2)
                print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1)
            # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "label_loss: ", loss1_label)
            # 7. train accuracy

            # train_accuracy += torch.eq(torch.ones(1).to(device) - pred_out, torch.squeeze(label).long()).sum().float().item()
            train_accuracy += torch.eq(torch.ones(1).to(device) - pred_out, torch.squeeze(label[:, 0]).long()).sum().float().item()
            if i % 10 == 0:
                # print("Epoch:", epoch, " i:", i + 1, " train loss:", np.mean(train_loss))
                image_mask = torch.max(output[0].unsqueeze(0).cpu(), 1)[1]

                image_decisionMap = image_mask.numpy().squeeze(0) * 255

                image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
                image_decisionMap.save("./results/train_fusion/decision.png")

            # 写入loss
            time = '%s' % datetime.datetime.now()
            step = 'step[%d]' % epoch
            iteration = 'iter[%d]' % i
            Loss = '%f' % loss
            losslist = [time, step, iteration, Loss]
            data = pd.DataFrame([losslist])
            data.to_csv("./results/train_note/loss.csv", mode='a', header=False, index=False)

        accuracy = train_accuracy / (len(train_loader_label.dataset) * 65536)
        print("Epoch:", epoch, " train loss: ", np.mean(train_loss), " train accuracy:", accuracy)
        # 2.valid
        # if epoch % 10 == 0:
        #     valid(epoch, valid_loader, model_net, device, loss_function)
        # 3.scheduler
        scheduler.step()
        # scheduler.step(np.mean(train_loss))
        # 4.save ckpt
        model_savePath = opt.save_models
        if not os.path.exists(model_savePath):
            os.makedirs(model_savePath)
        torch.save(model_net, os.path.join(model_savePath, 'net_%03d.pkl' % (epoch + 1)))
        # 5.train fusion

        img_root = opt.train_test_dataset_path
        save_root = opt.train_test_save_path
        img1_root = os.path.join(img_root, 'img1')
        img2_root = os.path.join(img_root, 'img2')
        img1_root = img1_root.replace('\\', '/')
        img2_root = img2_root.replace('\\', '/')
        decisionMap_save_root = os.path.join(save_root, 'decisionmap')
        fusion_save_root = os.path.join(save_root, 'fusion')
        decisionMap_save_root = decisionMap_save_root.replace('\\', '/')
        fusion_save_root = fusion_save_root.replace('\\', '/')
        if not os.path.exists(decisionMap_save_root):
            os.makedirs(decisionMap_save_root)
        if not os.path.exists(fusion_save_root):
            os.makedirs(fusion_save_root)
        img_names = os.listdir(img1_root)
        for img_name in img_names:
            name, suffix = img_name.split('.')
            img1_dir = os.path.join(img1_root, img_name)
            img2_dir = os.path.join(img2_root, name.split('-A')[0] + "-B.png")
            focus_maps_savePath = os.path.join(decisionMap_save_root, str(epoch) + "_" + name.split('-A')[0] + '_focus.png')
            decisionMap_savePath = os.path.join(decisionMap_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_re_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '_rec.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            focus_maps_savePath = focus_maps_savePath.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')
            fusion_re_savePath = fusion_re_savePath.replace('\\', '/')

            train_fusion(model_net, img1_dir, img2_dir, focus_maps_savePath, decisionMap_savePath, fusion_savePath, fusion_re_savePath, device)

    train_log_txt_formatter = "[Epoch], {epoch: 02d}, [Loss], {loss}\n"
    to_write = train_log_txt_formatter.format(epoch=epoch,
                                              loss=" ".join((["{}".format(np.mean(train_loss), '.5f')]))
                                              )


def train_fusion(train_model, image1_path, image2_path, focus_maps_savePath, decisionMap_savePath, fusionMap_savePath, fusionMap_re_savePath, device):
    train_model.eval()
    torch.set_grad_enabled(False)
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    input_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image1_tensor = input_transform(image1).unsqueeze(0)
    image2_tensor = input_transform(image2).unsqueeze(0)

    # homoRe = utils.homogeneousRe(image1_tensor, image2_tensor)
    # homoRe = homoRe.to(device)

    image_device = torch.cat((image1_tensor, image2_tensor), dim=1).to(device)

    # output_focus, contrast_focus = train_model(image_device)
    # output_focus, focus_maps = train_model(image_device)
    output_focus = train_model(image_device)

    # focus_maps
    # focus_maps = torch.softmax(focus_maps, dim=1)
    # focus_mask = torch.max(focus_maps.cpu(), 1)[1]
    # # focus_mask = focus_maps
    # focus_mask = focus_mask.cpu().detach().numpy().squeeze() * 255
    # focus_mask[focus_mask < 0] = 0
    # focus_mask[focus_mask > 255] = 255
    # focus_mask = 255 - focus_mask
    # focus_mask = Image.fromarray(focus_mask.astype(np.uint8))
    # focus_mask.save(focus_maps_savePath)

    # output = torch.softmax(output_focus, dim=1)
    output = torch.sigmoid(output_focus)

    image_mask = torch.max(output.cpu(), 1)[1]

    image_decisionMap = image_mask.numpy().squeeze(0) * 255
    image_decisionMap[image_decisionMap < 0] = 0
    image_decisionMap[image_decisionMap > 255] = 255
    image_decisionMap = 255 - image_decisionMap

    image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
    image_decisionMap.save(decisionMap_savePath)

    image_mask = image_mask.numpy()
    image_mask = np.transpose(image_mask, (1, 2, 0))

    if len(np.array(image1).shape) == 3:
        image_mask = np.repeat(image_mask, 3, 2)

    image_mask = 1 - image_mask

    image_fusion = image1 * image_mask + image2 * (1 - image_mask)
    image_fusion[image_fusion < 0] = 0
    image_fusion[image_fusion > 255] = 255
    image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
    image_fusion.save(fusionMap_savePath)

    # output_fusion = output_fusion.squeeze().cpu().detach().numpy() * 255
    # output_fusion = np.transpose(output_fusion, (1, 2, 0))
    # output_fusion = np.clip(output_fusion, 0, 255)
    # output_fusion = Image.fromarray(output_fusion.astype(np.uint8))
    # output_fusion.save(fusionMap_re_savePath)
    torch.set_grad_enabled(True)

# 参数设置
def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epoch', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='train learning rate')
    parser.add_argument('--batch_size_label', type=int, default=32, help='train batch size')
    parser.add_argument('--batch_size_nolabel', type=int, default=4, help='train batch size')
    parser.add_argument('--image_size', type=int, default=128, help='image size')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--train_test_dataset_path', type=str, default='./dataset/Lytro/', help='path of test dataset path in train')
    parser.add_argument('--train_test_save_path', type=str, default='./results/train/v39_benchmark256_O5_K9_D2_P4_lr0.001_BCEWLoss_bs16_dct/', help='path of save path in train')
    parser.add_argument('--save_models', type=str, default='./ckpt/v39_benchmark256_O5_K9_D2_P4_lr0.001_BCEWLoss_bs16_dct/', help='path of save path in train')
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


if __name__ == '__main__':
    # os.chdir("/xiaobin_fix/whf/pytorch_code/Semi_Surpervised")
    # os.chdir("/home/WuHF/whf/pytorchCode/Semi-supervised")
    main(Parser())

