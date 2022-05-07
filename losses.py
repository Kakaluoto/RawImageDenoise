# coding=utf-8
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from math import exp
from data_manager import DataManager
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import MS_SSIM


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class PSNRLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, x, y):
        MSELoss = nn.MSELoss()
        mse = MSELoss(x, y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = 10 * torch.log10(1 / mse)
        return -loss


class MSSSIMLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, data_range=1, size_average=True, channel=4):
        super(MSSSIMLoss, self).__init__()
        self.ms_ssim_module = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, x, y):
        # loss = 1 - self.ms_ssim_module(x, y)
        loss = self.ms_ssim_module(x, y)
        return -loss


if __name__ == '__main__':
    gt_path = './data/gt'
    noise_path = './data/denoise_result'
    psnr = PSNRLoss()
    msssim = MSSSIMLoss()
    data_manager = DataManager(gt_path=gt_path, noise_path=noise_path, do_augment=True)
    train_loader = DataLoader(dataset=data_manager,
                              batch_size=1,
                              shuffle=True,
                              num_workers=2)
    for noise, gt in train_loader:
        psnr_loss = psnr(noise, gt)
        msssim_loss = msssim(noise, gt)
