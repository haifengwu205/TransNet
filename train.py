#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 07/06/2021 15:39
# @Author : hfw
# @Version：V 0.1
# @File : train.py
# @desc :
from utils import parser, data_transforms_label, data_transforms_nolabel, util, ssimLoss, gradientLoss, lp_lssim_loss
import torchvision.transforms as transforms
from dataset import datasetList
from torch.utils.data import DataLoader
import pandas as pd
import torch
from ckpt import modelL
import os
import datetime
from PIL import Image
import numpy as np
from itertools import cycle
import torch.nn as nn


opt = parser.Parser()
def main():
    #************************************************ 1 **********************************************
    image_size = opt.image_size
    input_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        # transforms.Grayscale(1),
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

    # ************************************************ 2 **********************************************
    # ================================= load train data =============================
    train_set_label = datasetList.data_set(csv_path='./dataset/dataset_csv/data_all_v3_label.csv',
                                     input_transform=input_transform,
                                     target_transform=target_transform,
                                     gray_transform=gray_transform,
                                     co_transform=co_transform_label,
                                     hasLabel=True)
    print(len(train_set_label))
    train_loader_label = DataLoader(train_set_label, batch_size=opt.batch_size_label, shuffle=False, num_workers=opt.workers)
    # ================================= load train data =============================
    train_set_nolabel = datasetList.data_set(csv_path='./dataset/dataset_csv/data_all_v3_nolabel.csv',
                                           input_transform=input_transform,
                                           target_transform=target_transform,
                                           gray_transform=gray_transform,
                                           co_transform=co_transform_nolabel,
                                           hasLabel=False)
    print(len(train_set_nolabel))
    train_loader_nolabel = DataLoader(train_set_nolabel, batch_size=opt.batch_size_nolabel, shuffle=False, num_workers=opt.workers)
    # ================================= train note =============================
    df = pd.DataFrame(columns=['time', 'step', 'iteration', 'loss'])
    df.to_csv("./results/train_note/loss.csv", index=False)
    # ====================================== train ==================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_net = modelL.Network(6, 2, device)
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
    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = torch.nn.MSELoss(reduction='mean')
    # loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = dtt_loss.DttLoss()
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.learning_rate, betas=(0.92, 0.99))  # 0.92-0.99
    # optimizer = torch.optim.SGD(model_net.parameters(), lr=opt.learning_rate)  # 0.92-0.99
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    train(model_net, opt.Epoch, initial_epoch, train_loader_label, train_loader_nolabel, loss_function, optimizer, scheduler, device)
    # train_label(model_net, opt.Epoch, initial_epoch, train_loader_label, loss_function, optimizer, scheduler, device)


def train_label(model_net, EPOCH, initial_epoch, train_loader_label, loss_function, optimizer, scheduler, device):
    for epoch in range(initial_epoch, EPOCH):
        # 1.Train
        train_loss = []
        train_accuracy = 0
        for i, data1 in enumerate(train_loader_label):
            data1_img1, data1_img2, data1_gt, data1_labels, data1_label = data1

            # img_foreground, img_background, img_groundTruth, label_one_hot, label
            data1_img1, data1_img2, data1_gt, data1_labels, data1_label = data1_img1.to(device), data1_img2.to(device), data1_gt.to(device), data1_labels.to(device), data1_label.to(device)

            # print(len(data1[0][0]))
            # print(len(data2))

            data1_img = torch.cat((data1_img1, data1_img2), 1)
            # 1. forward propagation
            outputs_focus = model_net(data1_img)

            # outputs_focus = torch.softmax(outputs_focus, dim=1)
            # 2. compute loss
            loss = loss_function(outputs_focus, data1_labels)
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
            print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss)
            # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "label_loss: ", loss1_label)
            # 7. train accuracy
            output = torch.softmax(outputs_focus, dim=1)
            pred_out = torch.max(output[:opt.batch_size_label, :, :, :], 1)[1]
            train_accuracy += torch.eq(pred_out, torch.squeeze(data1_label).long()).sum().float().item()
            if i % 10 == 0:
                print("Epoch:", epoch, " i:", i + 1, " train loss:", np.mean(train_loss))
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
            img2_dir = os.path.join(img2_root, name.split('-A')[0] + "-B.jpg")
            decisionMap_savePath = os.path.join(decisionMap_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_re_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '_rec.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')
            fusion_re_savePath = fusion_re_savePath.replace('\\', '/')

            train_fusion(model_net, img1_dir, img2_dir, decisionMap_savePath, fusion_savePath, fusion_re_savePath, device)


def train(model_net, EPOCH, initial_epoch, train_loader_label, train_loader_nolabel, loss_function, optimizer, scheduler, device):
    for epoch in range(initial_epoch, EPOCH):
        # 1.Train
        train_loss = []
        train_accuracy = 0
        for i, (data1, data2) in enumerate(zip(train_loader_label, cycle(train_loader_nolabel))):
            data1_img1, data1_img2, data1_gt, data1_labels, data1_label = data1
            data2_img1, data2_img2 = data2

            # img_foreground, img_background, img_groundTruth, label_one_hot, label
            data1_img1, data1_img2, data1_gt, data1_labels, data1_label = data1_img1.to(device), data1_img2.to(device), data1_gt.to(device), data1_labels.to(device), data1_label.to(device)
            # img_foreground, img_background
            data2_img1, data2_img2 = data2_img1.to(device), data2_img2.to(device)
            # print(len(data1[0][0]))
            # print(len(data2))

            data1_img = torch.cat((data1_img1, data1_img2), 1)
            data2_img = torch.cat((data2_img1, data2_img2), 1)
            data_img = torch.cat((data1_img, data2_img), 0)
            # 1. forward propagation
            outputs_fusion, outputs_focus = model_net(data_img)

            outputs_focus = torch.softmax(outputs_focus, dim=1)
            # 2. compute loss
            output1_fusion = outputs_fusion[:opt.batch_size_label, :, :, :]
            output1_focus = outputs_focus[:opt.batch_size_label, :, :, :]

            output2_fusion = outputs_fusion[opt.batch_size_label:, :, :, :]
            output2_focus = outputs_focus[opt.batch_size_label:, :, :, :]

            # loss for label

            # print(data1_labels.shape)
            # print(output1_focus.shape)
            print(output1_focus.shape, "  ", data1_labels.shape)
            loss1_label = nn.BCELoss()(output1_focus, data1_labels)
            # print(loss1_label.type())
            fusion_label = output1_focus[:, 0, :, :].unsqueeze(1) * data1_img1 + output1_focus[:, 1, :, :].unsqueeze(1) * data1_img2
            # loss2_label = 1 - ssimLoss.SSIMLoss()(output1_fusion, fusion_label)
            loss2_label = nn.MSELoss()(output1_fusion, fusion_label)
            # loss3_label = gradientLoss.gradient_loss_merge(data1_gt, fusion_label, opt.cuda, device)
            # loss4_label = torch.norm(data1_gt - output1_fusion, 2, 1).mean()
            loss3_label = nn.L1Loss()(output1_fusion, data1_gt)
            loss1 = 1 * loss1_label + 0.1 * loss2_label + 0.1 * loss3_label
            # loss for no label
            fusion_nolabel = output2_focus[:, 0, :, :].unsqueeze(1) * data2_img1 + output2_focus[:, 1, :, :].unsqueeze(1) * data2_img2
            # loss2_nolabel = 1 - ssimLoss.SSIMLoss()(output2_fusion, fusion_nolabel)
            loss_nolabel = nn.L1Loss()(output2_fusion, fusion_nolabel)
            loss2 = loss_nolabel
            # print(loss2)
            loss = loss1 + loss2
            # loss = loss1
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
            print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "label_loss: ", loss1_label,
                  "SSIM_loss", loss2_label, "recon_loss: ", loss3_label, "loss2: ", loss2)
            # print("Epoch: ", epoch, "iteration: ", i, "train_loss: ", loss, "loss1: ", loss1, "label_loss: ", loss1_label)
            # 7. train accuracy
            output = torch.softmax(outputs_focus, dim=1)
            pred_out = torch.max(output[:opt.batch_size_label, :, :, :], 1)[1]
            train_accuracy += torch.eq(pred_out, torch.squeeze(data1_label).long()).sum().float().item()
            if i % 10 == 0:
                print("Epoch:", epoch, " i:", i + 1, " train loss:", np.mean(train_loss))
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
            img2_dir = os.path.join(img2_root, name.split('-A')[0] + "-B.jpg")
            decisionMap_savePath = os.path.join(decisionMap_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '.png')
            fusion_re_savePath = os.path.join(fusion_save_root, str(epoch) + "_" + name.split('-A')[0] + '_rec.png')

            img1_dir = img1_dir.replace('\\', '/')
            img2_dir = img2_dir.replace('\\', '/')
            decisionMap_savePath = decisionMap_savePath.replace('\\', '/')
            fusion_savePath = fusion_savePath.replace('\\', '/')
            fusion_re_savePath = fusion_re_savePath.replace('\\', '/')

            train_fusion(model_net, img1_dir, img2_dir, decisionMap_savePath, fusion_savePath, fusion_re_savePath, device)


def train_fusion(train_model, image1_path, image2_path, decisionMap_savePath, fusionMap_savePath, fusionMap_re_savePath, device):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    input_transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    image1_tensor = input_transform(image1).unsqueeze(0)
    image2_tensor = input_transform(image2).unsqueeze(0)

    image_device = torch.cat((image1_tensor, image2_tensor), dim=1).to(device)

    output_fusion, output_focus = train_model(image_device)
    # output_focus = train_model(image_device)

    output = torch.softmax(output_focus, dim=1)

    image_mask = torch.max(output.cpu(), 1)[1]

    image_decisionMap = image_mask.numpy().squeeze(0) * 255
    image_decisionMap[image_decisionMap < 0] = 0
    image_decisionMap[image_decisionMap > 255] = 255
    print("decisionmap_shpae: ", image_decisionMap.shape)

    image_decisionMap = Image.fromarray(image_decisionMap.astype(np.uint8))
    image_decisionMap.save(decisionMap_savePath)

    image_mask = image_mask.numpy()
    image_mask = np.transpose(image_mask, (1, 2, 0))

    if len(np.array(image1).shape) == 3:
        image_mask = np.repeat(image_mask, 3, 2)

    image_fusion = image1 * image_mask + image2 * (1 - image_mask)
    image_fusion[image_fusion < 0] = 0
    image_fusion[image_fusion > 255] = 255
    image_fusion = Image.fromarray(image_fusion.astype(np.uint8))
    image_fusion.save(fusionMap_savePath)

    output_fusion = output_fusion.squeeze().cpu().detach().numpy() * 255
    output_fusion = np.transpose(output_fusion, (1, 2, 0))
    output_fusion = np.clip(output_fusion, 0, 255)
    output_fusion = Image.fromarray(output_fusion.astype(np.uint8))
    output_fusion.save(fusionMap_re_savePath)


if __name__ == '__main__':
    # os.chdir("/xiaobin_fix/whf/pytorch_code/Semi_Surpervised")
    # os.chdir("/home/Bxl/whf/pytorchCode/Semi-Supervised")
    main()

