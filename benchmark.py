import os
from torch import nn
import torch
import tqdm
from data_manager import DataManager
from torch.utils import data
from losses import PSNRLoss, MSSSIMLoss
from RIDnet import RIDnet
from RIDnet_v2 import RIDnet_v2


class BenchMark:
    def __init__(self, gt_path, noise_path, weights_path):
        data_manager = DataManager(gt_path=gt_path, noise_path=noise_path)
        self.weights_names = [file_name for file_name in os.listdir(weights_path) if file_name.endswith("pth")]
        self.weights_path_list = [os.path.join(weights_path, file_name) for file_name in self.weights_names]
        self.data_loader = data.DataLoader(data_manager, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
        self.psnr_loss = PSNRLoss()
        self.ssim_loss = MSSSIMLoss()

    def compute_avg_score(self, net: nn.Module):
        total_psnr, total_ssim = 0, 0
        with torch.no_grad():
            for noise_img, gt_img in tqdm.tqdm(self.data_loader):
                noise_img = noise_img.cuda()
                gt_img = gt_img.cuda()
                denoise_img = net(noise_img)
                psnr = -self.psnr_loss(denoise_img, gt_img).item()
                ssim = -self.ssim_loss(denoise_img, gt_img).item()
                total_psnr += psnr
                total_ssim += ssim
        avg_psnr = total_psnr / len(self.data_loader)
        avg_ssim = total_ssim / len(self.data_loader)
        avg_score = avg_psnr + avg_ssim
        return avg_psnr, avg_ssim, avg_score

    def run_benchmark(self, net: nn.Module):
        model_rank = []
        if net.training:
            net.eval()
        for i, weights_path in enumerate(self.weights_path_list):
            checkpoint = torch.load(weights_path)
            net.load_state_dict(checkpoint['model'])
            net = net.cuda()
            psnr, ssim, score = self.compute_avg_score(net)
            model_rank.append({"weight": self.weights_names[i], "psnr": psnr, "ssim": ssim, "score": score})
            print(self.weights_names[i], psnr, ssim, score)
        model_rank = sorted(model_rank, key=lambda x: x["score"])
        return model_rank


if __name__ == '__main__':
    gt_path = '../dataset/dataset/ground truth'
    noise_path = '../dataset/dataset/noise'
    weights_path = './checkpoints/RIDnet_v2'
    bench_mark_path = './data/bench_mark/ridnet_v2/benchmark.txt'
    denoise_net = RIDnet_v2(in_channels=4, out_channels=4, num_feautres=32)
    denoise_net.eval()
    benchmark = BenchMark(gt_path, noise_path, weights_path)
    model_rank = benchmark.run_benchmark(denoise_net)
    for model_weight in model_rank:
        with open(bench_mark_path, "a+") as f:
            f.write("weight:{},PSNR:{:.3f},SSIM:{:.3f},Score:{:.4f}s".format(model_weight["weight"],
                                                                             model_weight["psnr"],
                                                                             model_weight["ssim"],
                                                                             model_weight["score"]) + '\n')
    print(model_rank)
