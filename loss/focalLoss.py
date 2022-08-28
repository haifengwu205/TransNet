#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 25/11/2021 19:02
# @Author : hfw
# @Version：V 0.1
# @File : focalLoss.py
# @desc :


import torch
import torch.nn as nn
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


if __name__ == '__main__':
    a = torch.softmax(torch.randn((2, 2, 32, 32)), dim=1)
    target1 = torch.ones((2, 1, 32, 32))
    target2 = torch.zeros((2, 1, 32, 32))
    target = torch.cat((target1, target2), dim=1)

    loss = FocalLoss()(a, target)
    print(loss)