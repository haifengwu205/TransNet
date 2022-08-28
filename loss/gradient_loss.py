# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from pylab import *
from PIL import Image
import torchvision.transforms as transforms


def LaplaceAlogrithm(image, device):
    assert torch.is_tensor(image) is True

    # laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)[np.newaxis,:,:].repeat(3,0)
    laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).to(device)
    print(laplace_operator.shape)

    image = image - F.conv2d(image,laplace_operator.unsqueeze(0),padding = 1,stride = 1)
    return image


def gradient_loss_merge(img1, gt, device): #exclusion loss: to make the gradient between img1 and img2 more non-uniform.
    grad_img1 = LaplaceAlogrithm(img1, device)
    gt = LaplaceAlogrithm(gt, device)
    gt.requires_grad_(False)

    g_loss = F.l1_loss(grad_img1, gt, reduction='sum')

    return g_loss


if __name__ == '__main__':
    label = Image.open(r"E:\pytorchCode\Semi-Supervised\results\train\v22_CE0.01_F64_lossSum_K9_bs16_pd4\decisionmap\0_lytro-01.png")
    decisionmap = Image.open(r"E:\pytorchCode\Semi-Supervised\results\train\v22_CE0.01_F64_lossSum_K9_bs16_pd4\decisionmap\1_lytro-01.png")
    input_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = input_transform(label).unsqueeze(0).to(device)
    decisionmap = input_transform(decisionmap).unsqueeze(0).to(device)
    print(label.shape)
    gloss = gradient_loss_merge(label, label, device)
    print(gloss)


