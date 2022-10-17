from glob import glob
from osgeo import gdal
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
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

# 4X
scale = 8
original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
gt = '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/resized_gt_'+str(scale)+'X/u_v'
dip = '/lustre/scratch/guiyli/Dataset_WIND/DIP/torch_resize/result_dip_'+str(scale)+'X/u_v'
metrics = {'mse': [], 'std': []}
metrics_min = 0
metrics_max = 0


for f in tqdm(glob(original+'/*.npy')):
    name = os.path.basename(f)[:-4]
    # c h w
    img_gt = np.stack((gdal.Open(gt+'/'+name+'_channel0.tif').ReadAsArray(),gdal.Open(gt+'/'+name+'_channel1.tif').ReadAsArray()))
    img_dip = np.stack((gdal.Open(dip+'/'+name+'_channel0.tif').ReadAsArray(),gdal.Open(dip+'/'+name+'_channel1.tif').ReadAsArray()))

    mse = np.mean(np.square(img_gt - img_dip))
    # swd = sliced_wasserstein(img_gt, img_dip)
    metrics['mse'].append(mse / np.square(img_gt.mean()))

    metrics_min = metrics_min if img_gt.min() > metrics_min else img_gt.min()
    metrics_max = metrics_max if img_gt.max() < metrics_max else img_gt.max()

os.makedirs('results/torch_resize',exist_ok=True)
np.save(
    os.path.join('results/torch_resize', "error_mse_"+str(scale)+"X.npy"), metrics['mse']
)

text_file = open(
        os.path.join('results/torch_resize', "mean_metrics_mse_"+str(scale)+"X.txt"),
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