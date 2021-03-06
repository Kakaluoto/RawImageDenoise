import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
from unet_ppp import UNet_3Plus_s
from unet_ppp import UNet_3Plus
from unet_v1 import Unet_v1
from unet_v2 import Unet_v2
from unet_v3 import Unet_v3
from RIDnet import RIDnet
from RIDnet_v2 import RIDnet_v2
import argparse


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def read_image(input_path):
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


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


def denoise_raw(net, input_path, output_path, ground_path,
                model_path, black_level, white_level, do_metric=True, do_plot=False):
    """
    pre-process
    """
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
    raw_data_expand_c_normal = raw_data_expand_c_normal.cuda()
    # net = Unet()
    if model_path is not None:
        checkpoint = torch.load(os.path.join(model_path))
        net.load_state_dict(checkpoint['model'])
        net = net.cuda()
        # net.load_state_dict(torch.load(model_path))
    net.eval()

    """
    inference
    """
    result_data = net(raw_data_expand_c_normal)

    """
    post-process
    """
    result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    write_back_dng(input_path, output_path, result_write_data)

    if do_metric:
        """
        Example: obtain ground truth
        """
        gt = rawpy.imread(ground_path).raw_image_visible
        """
        obtain psnr and ssim
        """
        psnr = skimage.metrics.peak_signal_noise_ratio(
            gt.astype(np.float64), result_write_data.astype(np.float64), data_range=white_level)
        ssim = skimage.metrics.structural_similarity(
            gt.astype(np.float64), result_write_data.astype(np.float64), multichannel=True, data_range=white_level)
        print('psnr:', psnr)
        print('ssim:', ssim)
    if do_plot:
        """
        Example: this demo_code shows your input or gt or result image
        """
        f0 = rawpy.imread(ground_path)
        f1 = rawpy.imread(input_path)
        f2 = rawpy.imread(output_path)
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(f0.postprocess(use_camera_wb=True))
        axarr[1].imshow(f1.postprocess(use_camera_wb=True))
        axarr[2].imshow(f2.postprocess(use_camera_wb=True))
        axarr[0].set_title('gt')
        axarr[1].set_title('noisy')
        axarr[2].set_title('de-noise')
        plt.show()


def main(args):
    model_path = args.model_path
    black_level = args.black_level
    white_level = args.white_level
    input_path = args.input_path
    output_path = args.output_path
    ground_path = args.ground_path
    unet = Unet()
    denoise_raw(unet, input_path, output_path, ground_path, model_path, black_level, white_level)


def generate_result(net):
    model_path = "models/RIDnet_v2/40.963536_best_4563.pth"
    black_level = 1024
    white_level = 16383
    # input_path = "../testset/"
    input_path = "../2rdhanddenoise/"
    output_path = "data/denoise_result/"
    ground_path = "./data/gt/demo.dng"
    input_file_names = os.listdir(input_path)
    input_files = [os.path.join(input_path, name) for name in input_file_names]
    output_files = [os.path.join(output_path, "denoise%s.dng" % num) for num in range(len(input_file_names))]
    for input_file, output_file in zip(input_files, output_files):
        denoise_raw(net, input_file, output_file, ground_path, model_path, black_level, white_level, do_metric=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default="./models/denoise_model_16.pth")
    # parser.add_argument('--black_level', type=int, default=1024)
    # parser.add_argument('--white_level', type=int, default=16383)
    # parser.add_argument('--input_path', type=str, default="./data/noise/demo_noise.dng")
    # parser.add_argument('--output_path', type=str, default="./data/result/demo_torch_res.dng")
    # parser.add_argument('--ground_path', type=str, default="./data/gt/demo.dng")
    #
    # args = parser.parse_args()v1
    # main(args)
    denoise_net = RIDnet_v2(in_channels=4, out_channels=4, num_feautres=32)
    # denoise_net = Unet()
    generate_result(denoise_net)
