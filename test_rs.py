import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.optim
from osgeo import gdal
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision import transforms
from tqdm import tqdm

from models import *
from models.downsampler import Downsampler
from utils.sr_utils import *


def image_save(image, path):
    '''
    purely save your image using OpenCV without COMPRESSION

    Args:
        image: (height, width, nc)
    save:
        OpenCV:
            cv2.imwrite
    '''
    if not type(image).__module__ == np.__name__:
        try:
            image = image.numpy().squeeze()
        except AttributeError:
            image = np.array(image).squeeze()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image.squeeze(), [cv2.IMWRITE_TIFF_COMPRESSION, 0])


def upscale(feat, scale_factor: int = 2):
    # resolution decrease
    if scale_factor == 1:
        return feat
    else:
        return F.avg_pool2d(feat, scale_factor)


def sliced_wasserstein(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape  # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(
            A.shape[1], dirs_per_repeat
        )  # (descriptor_component, direction)
        dirs /= np.sqrt(
            np.sum(np.square(dirs), axis=0, keepdims=True)
        )  # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)  # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(
            projA, axis=0
        )  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)  # pointwise wasserstein distances
        results.append(np.mean(dists))  # average over neighborhoods and directions
    return np.mean(results)  # average over repeats


def calc_metrics(scale: int = 4, data_type: str = 'Wind'):
    if data_type == 'Wind':
        original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        gt = (
            '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/new_iters/resized_gt_'
            + str(scale)
            + 'X/u_v'
        )
        dip = (
            '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/new_iters/result_dip_'
            + str(scale)
            + 'X/u_v'
        )
    elif data_type == 'Solar':
        original = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed/'
        gt = (
            '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/torch_resize/new_iters/resized_gt_'
            + str(scale)
            + 'X/'
        )
        dip = (
            '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/torch_resize/new_iters/result_dip_'
            + str(scale)
            + 'X/'
        )
        original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        gt = (
            '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/resized_gt_'
            + str(scale)
            + 'X/u_v'
        )
        dip = (
            '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/result_dip_'
            + str(scale)
            + 'X/u_v'
        )

    image_list = glob(original + '/*.npy')
    metrics = {'mse': [None] * len(image_list), 'std': []}
    metrics_min = 0
    metrics_max = 0

    for idx, f in enumerate(tqdm(image_list)):
        name = os.path.basename(f)[:-4]
        # c h w
        img_gt = np.stack(
            (
                gdal.Open(gt + '/' + name + '_channel0.tif').ReadAsArray(),
                gdal.Open(gt + '/' + name + '_channel1.tif').ReadAsArray(),
            )
        )
        img_dip = np.stack(
            (
                gdal.Open(dip + '/' + name + '_channel0.tif').ReadAsArray(),
                gdal.Open(dip + '/' + name + '_channel1.tif').ReadAsArray(),
            )
        )

        mse = np.mean(np.square(img_gt - img_dip))
        # swd = sliced_wasserstein(img_gt, img_dip)
        metrics['mse'][idx] = mse / np.square(img_gt.mean())

        metrics_min = metrics_min if img_gt.min() > metrics_min else img_gt.min()
        metrics_max = metrics_max if img_gt.max() < metrics_max else img_gt.max()

    os.makedirs('results/torch_resize/' + data_type, exist_ok=True)
    np.save(
        os.path.join(
            'results/torch_resize/' + data_type, "error_mse_" + str(scale) + "X.npy"
        ),
        metrics['mse'],
    )

    text_file = open(
        os.path.join(
            'results/torch_resize/' + data_type,
            "mean_metrics_mse_" + str(scale) + "X.txt",
        ),
        "w",
    )

    drange = metrics_max - metrics_min
    text_file.write("\n" + "Data Range: " + str(drange) + "\n")
    text_file.write(str(metrics_min) + ", " + str(metrics_max) + "\n")

    std0 = 'std0'

    text_file.write("\n" + "MSE/(mean^2) --> mean" + "\n")
    text_file.write(std0 + ": ")
    text_file.write(str(np.mean(metrics['mse'])) + "\n")

    text_file.write("\n" + "MSE/(mean^2) --> median" + "\n")
    text_file.write(std0 + ": ")
    text_file.write(str(np.median(metrics['mse'])) + "\n")

    print("Validation metrics saved!")


i = 0


def generate_sample(label, factor: int = 4, device='PC', data_type='Wind'):
    print('scale', factor, 'start!')

    seed = 66
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: ", seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

    imsize = -1
    enforse_div32 = 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
    PLOT = False

    # default num_scales is 5, which need the low resolution image size should be at least 16
    # we need to decrease num_scales to 4 to fit our data (8*8 --> 32*32 or 64*64)
    num_scales = 4
    lr_size = 8

    if data_type == 'Wind':
        if device == 'HPCC':
            path = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        else:
            path = '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
        max_scale = 64
    elif data_type == 'Solar':
        if device == 'HPCC':
            path = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
        else:
            path = '/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014'
        max_scale = 32

    torch_resize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (lr_size * max_scale, lr_size * max_scale),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ]
    )

    f = os.path.join(path, label)
    images = np.load(f).astype(np.float32)

    for c in range(images.shape[2]):
        global i
        i = 0
        img = images[:, :, c]

        img = torch_resize(Image.fromarray(img))
        # lores = upscale(img, max_scale)
        hires = Image.fromarray(np.array(upscale(img, max_scale // factor)).squeeze())
        pixel_min, pixel_max = hires.getextrema()

        # img = Image.fromarray(img).resize((lr_size * factor, lr_size * factor), Image.NEAREST)
        # pixel_min, pixel_max = img.getextrema()

        resized_path = (
            'results/'
            + data_type
            + '/resized_gt_'
            + str(factor)
            + 'X'
            + '_channel'
            + str(c)
            + '.tif'
        )
        image_save(hires, resized_path)
        hires = ((np.array(hires) - pixel_min) / max((pixel_max - pixel_min), 1e-5)) * (
            255 - 0
        ) + 0
        path_to_image = (
            'results/'
            + data_type
            + '/resized_gt_0-255_'
            + str(factor)
            + 'X'
            + '_channel'
            + str(c)
            + '.tif'
        )
        image_save(hires, path_to_image)

        # Starts here
        imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)

        imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(
            imgs['LR_pil'], imgs['HR_pil']
        )

        input_depth = 32
        INPUT = 'noise'
        pad = 'reflection'
        OPT_OVER = 'net'
        KERNEL_TYPE = 'lanczos2'
        LR = 0.01
        tv_weight = 0.0
        OPTIMIZER = 'adam'

        if factor == 4:
            # num_iter = 2000
            num_iter = 5000
            reg_noise_std = 0.03
        elif factor == 8:
            # num_iter = 4000
            num_iter = 6000
            reg_noise_std = 0.05
        else:
            assert False, 'We did not experiment with other factors'

        net_input = (
            get_noise(
                input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])
            )
            .type(dtype)
            .detach()
        )

        NET_TYPE = 'skip'  # UNet, ResNet
        n_channels = 1
        net = get_net(
            input_depth,
            'skip',
            pad,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=num_scales,  # default 5
            upsample_mode='bilinear',
            n_channels=n_channels,
        ).type(dtype)

        # Losses
        mse = torch.nn.MSELoss().type(dtype)

        img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

        downsampler = Downsampler(
            n_planes=n_channels,
            factor=factor,
            kernel_type=KERNEL_TYPE,
            phase=0.5,
            preserve_size=True,
        ).type(dtype)

        def closure():
            global i, net_input

            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

            out_HR = net(net_input)
            out_LR = downsampler(out_HR)

            total_loss = mse(out_LR, img_LR_var)

            if tv_weight > 0:
                total_loss += tv_weight * tv_loss(out_HR)

            total_loss.backward()

            # Log
            psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
            psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
            print(
                'Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR),
                '\r',
                end='',
            )

            # History
            psnr_history.append([psnr_LR, psnr_HR])

            if PLOT and i % 1000 == 0:
                out_HR_np = torch_to_np(out_HR)
                plot_image_grid(
                    [imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)],
                    factor=13,
                    nrow=3,
                )

            i += 1

            return total_loss

        psnr_history = []
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

        out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
        result_deep_prior = put_in_center(
            out_HR_np, imgs['orig_np'].shape[1:], n_channels=n_channels
        )

        # original data range
        result_original = (
            (result_deep_prior - result_deep_prior.min())
            / max((result_deep_prior.max() - result_deep_prior.min()), 1e-5)
        ) * (pixel_max - pixel_min) + pixel_min
        result_path = (
            'results/'
            + data_type
            + '/result_dip_'
            + str(factor)
            + 'X'
            + '_channel'
            + str(c)
            + '.tif'
        )
        image_save(result_original, result_path)
    np.save(
        'results/' + data_type + '/fake_DIP_' + str(factor) + 'X.npy',
        np.stack(
            (
                gdal.Open(
                    (
                        'results/'
                        + data_type
                        + '/result_dip_'
                        + str(factor)
                        + 'X'
                        + '_channel0.tif'
                    )
                ).ReadAsArray(),
                gdal.Open(
                    (
                        'results/'
                        + data_type
                        + '/result_dip_'
                        + str(factor)
                        + 'X'
                        + '_channel1.tif'
                    )
                ).ReadAsArray(),
            )
        ),
    )
    np.save(
        'results/' + data_type + '/hires_DIP_' + str(factor) + 'X.npy',
        np.stack(
            (
                gdal.Open(
                    (
                        'results/'
                        + data_type
                        + '/resized_gt_'
                        + str(factor)
                        + 'X'
                        + '_channel0.tif'
                    )
                ).ReadAsArray(),
                gdal.Open(
                    (
                        'results/'
                        + data_type
                        + '/resized_gt_'
                        + str(factor)
                        + 'X'
                        + '_channel1.tif'
                    )
                ).ReadAsArray(),
            )
        ),
    )


if __name__ == '__main__':
    # 4X
    # calc_metrics(scale=4, data_type='Wind')

    generate_sample(
        label='wtk_grid2_2014-07-19-12:00:00.npy',
        factor=4,
        device='PC',
        data_type='Wind',
    )
    generate_sample(
        label='wtk_grid2_2014-07-19-12:00:00.npy',
        factor=8,
        device='PC',
        data_type='Wind',
    )

    # generate_sample(label='month1_day14_hour16_minute30.npy',factor=4,device='PC', data_type='Solar')
    # generate_sample(label='month1_day14_hour16_minute30.npy',factor=8,device='PC', data_type='Solar')
