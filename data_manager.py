# coding=utf-8
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rawpy
import os
import matplotlib.pyplot as plt
from torch import nn
import skimage.metrics
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class DataManager(Dataset):
    def __init__(self, gt_path=None, noise_path=None, black_level=1024, white_level=16383, do_augment=True):
        self.black_level = black_level
        self.white_level = white_level
        if gt_path is not None and noise_path is not None:
            self.noise_images, self.ground_truth_images = self.load_img_path(gt_path=gt_path, noise_path=noise_path)
        else:
            self.noise_images, self.ground_truth_images = [], []
        if do_augment:
            data_augmentation = [
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                RandomSelectRotation([0, 90, 180, 270]),  # 随机旋转
            ]
            self.data_augmentation = transforms.Compose(data_augmentation)
        else:
            self.data_augmentation = None

    def __getitem__(self, index):
        noise_image = self.noise_images[index]
        ground_truth_image = self.ground_truth_images[index]
        seed = np.random.randint(2147483647)
        if self.data_augmentation is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            noise_image = self.preprocess(input_path=noise_image,
                                          black_level=self.black_level, white_level=self.white_level)
            noise_image = self.data_augmentation(noise_image)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            ground_truth_image = self.preprocess(input_path=ground_truth_image,
                                                 black_level=self.black_level, white_level=self.white_level)
            ground_truth_image = self.data_augmentation(ground_truth_image)
        else:
            noise_image = self.preprocess(input_path=noise_image,
                                          black_level=self.black_level, white_level=self.white_level)
            ground_truth_image = self.preprocess(input_path=ground_truth_image,
                                                 black_level=self.black_level, white_level=self.white_level)
        return noise_image, ground_truth_image

    def __len__(self):
        return len(self.noise_images)

    def load_img_path(self, gt_path='.', noise_path='.'):
        gt_img_names = [file_name for file_name in os.listdir(gt_path) if file_name.endswith("dng")]
        gt_img_names = sorted(gt_img_names, key=lambda x: int(x.split("_")[0]))
        noise_img_names = [file_name for file_name in os.listdir(noise_path) if file_name.endswith("dng")]
        noise_img_names = sorted(noise_img_names, key=lambda x: int(x.split("_")[0]))
        ground_truth_images = [os.path.join(gt_path, file_name) for file_name in gt_img_names]
        noise_images = [os.path.join(noise_path, file_name) for file_name in noise_img_names]
        return noise_images, ground_truth_images

    def preprocess(self, input_path, black_level=1024, white_level=16383):
        raw_data_expand_c, height, width = self.read_image(input_path)
        raw_data_expand_c_normal = self.normalization(raw_data_expand_c, black_level, white_level)
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal.reshape(height // 2, width // 2, 4), (2, 0, 1))).float()
        return raw_data_expand_c_normal

    def postprocess(self, result_data, height, width, black_level=1024, white_level=16383):
        result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
        result_data = self.inv_normalization(result_data, black_level, white_level)
        result_write_data = self.write_image(result_data, height, width)
        return result_write_data

    def normalization(self, input_data, black_level, white_level):
        output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
        return output_data

    def inv_normalization(self, input_data, black_level, white_level):
        output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
        output_data = output_data.astype(np.uint16)
        return output_data

    def read_image(self, input_path):
        raw = rawpy.imread(input_path)
        raw_data = raw.raw_image_visible
        height = raw_data.shape[0]
        width = raw_data.shape[1]

        raw_data_expand = np.expand_dims(raw_data, axis=2)
        raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                            raw_data_expand[0:height:2, 1:width:2, :],
                                            raw_data_expand[1:height:2, 0:width:2, :],
                                            raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
        return raw_data_expand_c, height, width

    def write_image(self, input_data, height, width):
        output_data = np.zeros((height, width), dtype=np.uint16)
        for channel_y in range(2):
            for channel_x in range(2):
                output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
        return output_data

    def val_psnr_ssim(self, net, ground_path, input_path):
        """
        pre-process
        """
        raw_data_expand_c, height, width = self.read_image(input_path)
        raw_data_expand_c_normal = self.normalization(raw_data_expand_c, self.black_level, self.white_level)
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
        raw_data_expand_c_normal = raw_data_expand_c_normal.cuda()
        """
        inference
        """
        result_data = net(raw_data_expand_c_normal)

        """
        post-process
        """
        result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
        result_data = self.inv_normalization(result_data, self.black_level, self.white_level)
        result_write_data = self.write_image(result_data, height, width)
        """
        Example: obtain ground truth
        """
        gt = rawpy.imread(ground_path).raw_image_visible
        """
        obtain psnr and ssim
        """
        psnr = skimage.metrics.peak_signal_noise_ratio(
            gt.astype(np.float64), result_write_data.astype(np.float64), data_range=self.white_level)
        ssim = skimage.metrics.structural_similarity(
            gt.astype(np.float64), result_write_data.astype(np.float64), multichannel=True, data_range=self.white_level)
        # print('psnr:', psnr)
        # print('ssim:', ssim)
        return psnr, ssim


class RandomSelectRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        # print(angle)
        return F.rotate(x, angle)


if __name__ == '__main__':
    # gt_path = '../dataset/dataset/ground truth'
    # noise_path = '../dataset/dataset/noise'
    gt_path = '../bigdataset/ground truth'
    noise_path = '../bigdataset/noise'
    data_manager = DataManager(gt_path=gt_path, noise_path=noise_path, do_augment=True)
    for noise, gt in zip(data_manager.noise_images, data_manager.ground_truth_images):
        print(noise, gt)
    train_loader = DataLoader(dataset=data_manager,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0)
    nosie_G_L1 = []
    gt_G_L1 = []
    L1Loss = nn.L1Loss()
    for nosie_img, gt_img in train_loader:
        nosie_G_L1.append(L1Loss(nosie_img[0, 1, :, :], nosie_img[0, 2, :, :]).item())
        gt_G_L1.append(L1Loss(gt_img[0, 1, :, :], gt_img[0, 2, :, :]).item())

    l1_G_noise = np.array(nosie_G_L1)
    l1_G_gt = np.array(gt_G_L1)
    print("noise:", np.mean(l1_G_noise))
    print("gt:", np.mean(l1_G_gt))

    plt.figure()
    plt.hist(nosie_G_L1, bins=100, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.title("noise")
    plt.ylabel('L1')
    plt.xlabel('num')

    plt.figure()
    plt.hist(gt_G_L1, bins=100, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.title("ground truth")
    plt.ylabel('L1')
    plt.xlabel('num')

    plt.show()
