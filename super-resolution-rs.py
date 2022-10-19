import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.optim
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIP')
    parser.add_argument('--type', type=str, help="Wind or Solar", default='Wind')
    parser.add_argument('--factor', type=int, help="4 or 8", default=4)
    parser.add_argument('--slice', type=int, help="0-19", default=0)
    parser.add_argument('--savePath', type=str, default='torch_resize/new_iters')
    parser.add_argument('--num_iter', type=int, default=0)
    args = parser.parse_args()

    factor = args.factor
    data_type = args.type
    if factor == 4:
        num_iter = args.num_iter if args.num_iter else 2000
        reg_noise_std = 0.03
    elif factor == 8:
        num_iter = args.num_iter if args.num_iter else 4000
        reg_noise_std = 0.05
    else:
        assert False, 'We did not experiment with other factors'

    print('Dataset', data_type)
    print('scale', factor, 'start!')
    print('slice', args.slice)

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

    # path_to_image = '/home/guiyli/Documents/DataSet/NSRDB/tifFiles/2014/dhi/month1_day1_hour8_minute30_dhi.tif'

    # default num_scales is 5, which need the low resolution image size should be at least 16
    # we need to decrease num_scales to 4 to fit our data (8*8 --> 32*32 or 64*64)
    num_scales = 4
    lr_size = 8

    # To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
    # e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8
    # path_to_image = 'data/sr/zebra_GT.png'

    # '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
    # '/lustre/scratch/guiyli/Dataset_WIND/npyFiles/u_v/2014'
    if data_type == 'Wind':
        path = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
        max_scale = 64
    elif data_type == 'Solar':
        path = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
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

    # generate ground truth image

    images_list = glob(path + '/*.npy')
    step = len(images_list) // 20
    start = args.slice * step
    stop = (args.slice + 1) * step
    processing = images_list[start:] if args.slice == 19 else images_list[start:stop]
    print("processing ", len(processing), 'images')
    for f in tqdm(processing):
        images = np.load(f).astype(np.float32)

        for c in range(images.shape[2]):
            img = images[:, :, c]

            img = torch_resize(Image.fromarray(img))
            # lores = upscale(img, max_scale)
            hires = Image.fromarray(np.array(upscale(img, max_scale // factor)).squeeze())
            pixel_min, pixel_max = hires.getextrema()

            # img = Image.fromarray(img).resize((lr_size * factor, lr_size * factor), Image.NEAREST)
            # pixel_min, pixel_max = img.getextrema()

            resized_path = (
                f.replace(
                    'Wind2014_removed' if data_type == 'Wind' else 'Solar2014_removed',
                    args.savePath+'/resized_gt_' + str(factor) + 'X'
                )[:-4]
                + '_channel'
                + str(c)
                + '.tif'
            )
            image_save(hires, resized_path)
            hires = ((np.array(hires) - pixel_min) / max((pixel_max - pixel_min), 1e-5)) * (
                255 - 0
            ) + 0

            path_to_image = (
                f.replace(
                    'Wind2014_removed' if data_type == 'Wind' else 'Solar2014_removed',
                    args.savePath+'/resized_gt_0-255_' + str(factor) + 'X',
                )[:-4]
                + '_channel'
                + str(c)
                + '.tif'
            )
            image_save(hires, path_to_image)

            # # generate ground truth image
            # img = gdal.Open(path_to_image).ReadAsArray()
            # img = Image.fromarray(img).resize((lr_size * factor, lr_size * factor), Image.NEAREST)
            # pixel_min, pixel_max = img.getextrema()
            # image_save(img, 'results/resized_gt_' + str(factor) + 'X.tif')
            # img = ((np.array(img) - pixel_min) / max((pixel_max - pixel_min), 1e-5)) * (255 - 0) + 0
            # image_save(img, 'results/resized_gt_0-255_' + str(factor) + 'X.tif')
            # image_save(img, 'results/resized_gt_0-255_' + str(factor) + 'X.png')
            # path_to_image = 'results/resized_gt_0-255_' + str(factor) + 'X.tif'

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

                if args.savePath=='debug':
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

            i = 0
            p = get_params(OPT_OVER, net, net_input)
            optimize(OPTIMIZER, p, closure, LR, num_iter)

            out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
            result_deep_prior = put_in_center(
                out_HR_np, imgs['orig_np'].shape[1:], n_channels=n_channels
            )

            # # 0-255
            # result_255 = (
            #     (result_deep_prior - result_deep_prior.min())
            #     / max((result_deep_prior.max() - result_deep_prior.min()), 1e-5)
            # ) * (255 - 0) + 0
            # image_save(result_255, 'results/result_deep_prior_0-255_' + str(factor) + 'X.png')

            # original data range
            result_original = (
                (result_deep_prior - result_deep_prior.min())
                / max((result_deep_prior.max() - result_deep_prior.min()), 1e-5)
            ) * (pixel_max - pixel_min) + pixel_min
            result_path = (
                f.replace(
                    'Wind2014_removed' if data_type == 'Wind' else 'Solar2014_removed',
                    args.savePath+'/result_dip_' + str(factor) + 'X'
                )[:-4]
                + '_channel'
                + str(c)
                + '.tif'
            )
            image_save(result_original, result_path)

            # # For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
            # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], out_HR_np], factor=4, nrow=1)
