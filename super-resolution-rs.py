import os

import cv2
import numpy as np
import torch
import torch.optim
from osgeo import gdal
from skimage.measure import compare_psnr

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


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize = -1
factor = 4  # 8
enforse_div32 = 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = False

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8
# path_to_image = 'data/sr/zebra_GT.png'
path_to_image = '/home/guiyli/Documents/DataSet/NSRDB/tifFiles/2014/dhi/month1_day1_hour8_minute30_dhi.tif'


img = gdal.Open(path_to_image).ReadAsArray()

img = Image.fromarray(img).resize((16 * factor, 16 * factor), Image.NEAREST)
pixel_min, pixel_max = img.getextrema()
image_save(img, 'results/resized_gt.tif')
img = ((np.array(img) - pixel_min) / max((pixel_max - pixel_min), 1e-5)) * (255 - 0) + 0
image_save(img, 'results/resized_gt_0-255.tif')
path_to_image = 'results/resized_gt_0-255.tif'


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
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

net_input = (
    get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]))
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
    num_scales=5,
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

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(
    out_HR_np, imgs['orig_np'].shape[1:], n_channels=n_channels
)

result_deep_prior = (
    (result_deep_prior - result_deep_prior.min())
    / max((result_deep_prior.max() - result_deep_prior.min()), 1e-5)
) * (pixel_max - pixel_min) + pixel_min
image_save(result_deep_prior, 'results/result_deep_prior.tif')
np.save('results/result_deep_prior.npy', result_deep_prior)

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], out_HR_np], factor=4, nrow=1)
