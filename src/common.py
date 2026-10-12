"""Image IO, inference and the Neighbor2Neighbor sub-sampler.

`space_to_depth`, `generate_mask_pair` and `generate_subimages` are taken from
the official Neighbor2Neighbor code (https://github.com/TaoHuang2018/Neighbor2Neighbor,
BSD 3-Clause, Copyright (c) 2021 Tao Huang; licence text in licenses/).
"""
import numpy as np
import torch
from PIL import Image

from .data import image_files
from .metrics import to_uint8
from .unet_n2nbr import UNet


def load_images(folder):
    """Every image in a folder, as float32 arrays in [0, 1], H x W x 3."""
    return [np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
            for f in image_files(folder)]


def save_image(x, path):
    Image.fromarray(to_uint8(x)).save(path)


def denoise(net, noisy_hwc, device="cpu"):
    """Run a network on one H x W x 3 image.

    The U-Net downsamples five times, so the image is reflect-padded to a
    multiple of 32, as in the official validation code.
    """
    H, W = noisy_hwc.shape[:2]
    size = (max(H, W) + 31) // 32 * 32
    padded = np.pad(noisy_hwc, [[0, size - H], [0, size - W], [0, 0]], "reflect")
    t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        out = net(t)[:, :, :H, :W]
    return out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def build_net(n_feature=48, device="cpu"):
    return UNet(in_nc=3, out_nc=3, n_feature=n_feature).to(device)


def load_net(path, n_feature=48, device="cpu"):
    net = build_net(n_feature, device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# Neighbor2Neighbor sub-sampler (BSD 3-Clause, see above)

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size,
                           w // block_size)


def generate_mask_pair(img, generator=None):
    """One random adjacent pair per 2x2 cell. Returns two boolean masks."""
    n, c, h, w = img.shape
    n_cells = n * h // 2 * w // 2
    mask1 = torch.zeros(size=(n_cells * 4,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(n_cells * 4,), dtype=torch.bool, device=img.device)
    # the eight ordered pairs of horizontally or vertically adjacent positions
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n_cells,), dtype=torch.int64, device=img.device)
    torch.randint(low=0, high=8, size=(n_cells,), generator=generator,
                  out=rd_idx, device=img.device)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0, end=n_cells * 4, step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    """One pixel per 2x2 cell, chosen by `mask`: half the size in each axis."""
    n, c, h, w = img.shape
    sub = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype,
                      layout=img.layout, device=img.device)
    for i in range(c):
        per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        per_channel = per_channel.permute(0, 2, 3, 1).reshape(-1)
        sub[:, i:i + 1, :, :] = per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return sub
