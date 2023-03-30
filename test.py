import torch
from torch.utils import data
import torchvision
import argparse
import copy
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
from pathlib import Path

from utils import *
from dataset import Dataset
from model import Unet, GaussianDiffusion, EMA
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

device = torch.device('cuda:%d'%(1) if torch.cuda.is_available() else 'cpu')

img_size = 128
timesteps = 4000
loss_type = "l1"
ema_decay = 0.995
batch_size = 32

path = "result1/model-261.pt" 
weight = torch.load(path)
diffusion_weight = weight["ema"]


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device)


diffusion = GaussianDiffusion(
    model,
    image_size = img_size,
    timesteps = timesteps,   # number of steps
    loss_type = loss_type    # L1 or L2
).to(device)
diffusion.load_state_dict(diffusion_weight)


ema = EMA(ema_decay)
ema_model = copy.deepcopy(diffusion)
ema_model.load_state_dict(diffusion.state_dict())

def make_grid(imgs, nrow, padding=0):
    """Numpy配列の複数枚の画像を、1枚の画像にタイルします

    Arguments:
        imgs {np.ndarray} -- 複数枚の画像からなるテンソル
        nrow {int} -- 1行あたりにタイルする枚数

    Keyword Arguments:
        padding {int} -- グリッドの間隔 (default: {0})

    Returns:
        [np.ndarray] -- 3階テンソル。1枚の画像
    """
    assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)
    # border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0)) # 下と右だけにpaddingを入れる
        height += padding
        width += padding
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:
        x = x[:(height * ncol - padding),:(width * nrow - padding),:] # 右端と下端のpaddingを削除
    return x


for l in range(46):

    b = 36
    img = torch.randn((b, 3, 128,128), device=device)
    label = torch.full((b, ), l+46,device=device)

    for i in tqdm(reversed(range(0, 4000)), desc='sampling loop time step', total=4000):
        img = ema_model.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), label)

    imgs = img.permute(0, 2, 3, 1).cpu().numpy()
    save_img = make_grid(imgs, 6)
    save_img = np.clip(save_img, 0, 1)
    save_img = save_img*255

    cv2.imwrite("sample_%s.png"%(l+46), save_img)