# coding:utf-8
import os
import torch
from torch import optim
from torch import nn
from torch.utils import data
import tqdm
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from data_manager import DataManager
import torch.optim.lr_scheduler as lr_scheduler
from losses import CharbonnierLoss, PSNRLoss, MSSSIMLoss
from unetTorch import Unet
from RIDnet import RIDnet
from RIDnet_v2 import RIDnet_v2
from RIDnet_v3 import RIDnet_v3
from unet_v1 import Unet_v1
from unet_v2 import Unet_v2
from unet_v3 import Unet_v3


class Config(object):
    gt_path = '../dataset/dataset/ground truth'
    noise_path = '../dataset/dataset/noise'
    # gt_path = '../bigdataset/ground truth'
    # noise_path = '../bigdataset/noise'
    num_workers = 4
    batch_size = 1
    max_epoch = 10000
    last_epoch = 0  # 上次训练到的位置
    lr = 2.5e-5  # 生成器的学习
    adam_beta1 = 0.9  # adam的beta1参数
    adam_beta2 = 0.999  # adam的beta2参数
    sgd_momentum = 0.8
    save_round = 1  # 每2个epoch保存一次模型
    val_round = 100  # 每100个step
    save_model_path = './checkpoints/RIDnet_v3'  # 'netd_num.pth'模型参数文件
    best_model_path = 'models/RIDnet_v3'
    val_gt_path = './data/gt/62_gt.dng'
    val_noise_path = './data/noise/62_noise.dng'
    logs_path = 'logs/RIDnet_v3/records.txt'
    tensorboard_path = 'runs/RIDnet_v3'
    lr_step_size = 60
    lr_gamma = 0.8
    model_save_number = 50
    best_score = 0


opt = Config()


def train(denoise_net):
    # 加载数据集
    data_manager = DataManager(gt_path=opt.gt_path, noise_path=opt.noise_path)
    dataloader = data.DataLoader(data_manager, batch_size=opt.batch_size,
                                 shuffle=True, num_workers=opt.num_workers,
                                 drop_last=True, pin_memory=True, persistent_workers=True)

    print("dataset:" + str(len(data_manager)) + ",dataloader:" + str(len(dataloader)))

    L1Loss = nn.L1Loss()
    psnrLoss = PSNRLoss()
    msssimLoss = MSSSIMLoss()
    model_pth_list = sorted(os.listdir(opt.save_model_path), key=lambda x: int(x.split("_")[0]))
    if len(model_pth_list) != 0:
        try:
            checkpoint = torch.load(os.path.join(opt.save_model_path, model_pth_list[-1]))
            denoise_net.load_state_dict(checkpoint['model'])
            opt.last_epoch = checkpoint['epoch']
            opt.best_score = checkpoint['best_score']
        except Exception as e:
            pass

        # 定义优化器和损失函数
    optimizer = optim.AdamW([{'params': denoise_net.parameters(), 'initial_lr': opt.lr}],
                            lr=opt.lr, betas=(opt.adam_beta1, opt.adam_beta2))
    # optimizer = optim.AdamW(denoise_net.parameters(), lr=opt.lr, betas=(opt.adam_beta1, opt.adam_beta2))
    # optimizer = optim.SGD([{'params': denoise_net.parameters(), 'initial_lr': opt.lr}], lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma, last_epoch=opt.last_epoch)

    # print(scheduler.get_lr())

    denoise_net = denoise_net.cuda()

    writer = SummaryWriter(opt.tensorboard_path)
    now = time.perf_counter()
    epochs = range(opt.last_epoch + 1, opt.max_epoch)
    for epoch in iter(epochs):
        loss = 0
        for step, (noise_images, ground_truth_images) in tqdm.tqdm(enumerate(dataloader)):
            noise_images = noise_images.cuda()
            ground_truth_images = ground_truth_images.cuda()
            optimizer.zero_grad()
            denoise_images = denoise_net(noise_images)
            l1loss = L1Loss(denoise_images, ground_truth_images) * 1e4
            psnrloss = psnrLoss(denoise_images, ground_truth_images)
            msssimloss = msssimLoss(denoise_images, ground_truth_images)
            loss = l1loss + psnrloss + msssimloss
            # loss = psnrloss + msssimloss
            loss.backward()
            optimizer.step()
            # if step % opt.val_round == 0:
        scheduler.step()
        denoise_net.eval()
        psnr, ssim = data_manager.val_psnr_ssim(denoise_net, opt.val_gt_path, opt.val_noise_path)
        denoise_net.train()
        # 输出友好信息
        print("Epoch:{},Loss:{:.6f},PSNR:{:.3f},SSIM:{:.3f},L1:{:.6f},LR:{:.8f},Time:{:.4f}s"
              .format(epoch, loss, psnr, ssim, l1loss, optimizer.param_groups[0]['lr'],
                      time.perf_counter() - now))
        with open(opt.logs_path, 'a+') as f:
            f.write("Epoch:{},Loss:{:.6f},PSNR:{:.3f},SSIM:{:.3f},L1:{:.6f},LR:{:.8f},Time:{:.4f}s"
                    .format(epoch, loss, psnr, ssim, l1loss, optimizer.param_groups[0]['lr'],
                            time.perf_counter() - now) + '\n')
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('psnr', psnr, global_step=epoch)
        writer.add_scalar('ssim', ssim, global_step=epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        if epoch % opt.save_round == 0:  # 这样做就可以每次save_round个epoch保存一个checkpoint
            if (psnr + ssim) >= opt.best_score:
                opt.best_score = psnr + ssim
                state = {'model': denoise_net.state_dict(), 'epoch': epoch, 'best_score': opt.best_score}
                torch.save(state, opt.best_model_path + '/{:.6f}_best_{}.pth'.format(opt.best_score, epoch))

            state = {'model': denoise_net.state_dict(), 'epoch': epoch, 'best_score': opt.best_score}
            torch.save(state, opt.save_model_path + '/%s_denoise_model.pth' % epoch)
            model_pth_list = sorted(os.listdir(opt.save_model_path), key=lambda x: int(x.split("_")[0]))
            if len(model_pth_list) > opt.model_save_number:
                remove_list = list(set(model_pth_list) - set(model_pth_list[-opt.model_save_number:]))
                for remove_pth in remove_list:
                    os.remove(os.path.join(opt.save_model_path, remove_pth))


if __name__ == '__main__':
    # fire.Fire()
    denoise_net = RIDnet_v3(in_channels=4, out_channels=4, num_feautres=32)
    train(denoise_net)
