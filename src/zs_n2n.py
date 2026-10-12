"""ZS-N2N (Mansour and Heckel, CVPR 2023).

No dataset and no stored model: a small network is trained from scratch on the
one noisy image to be cleaned. The training pair comes from two fixed 2x2
diagonal-averaging kernels with stride 2, and the network predicts the noise
(residual learning). Architecture and kernels follow the code published in the
paper (arXiv:2303.11253).
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import psnr, ssim, to_uint8


class TinyNet(nn.Module):
    """Two 3x3 convolutions and a 1x1, no normalisation, no pooling."""

    def __init__(self, n_chan=3, chan_embed=48):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def pair_downsampler(img):
    """The anti-diagonal and the main-diagonal average of every 2x2 cell."""
    c = img.shape[1]
    anti = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]],
                        dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    main = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]],
                        dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    return (F.conv2d(img, anti, stride=2, groups=c),
            F.conv2d(img, main, stride=2, groups=c))


def loss_func(net, noisy):
    d1, d2 = pair_downsampler(noisy)
    pred1 = d1 - net(d1)
    pred2 = d2 - net(d2)
    loss_res = 0.5 * (F.mse_loss(d1, pred2) + F.mse_loss(d2, pred1))

    # consistency: downsample then denoise must match denoise then downsample
    denoised = noisy - net(noisy)
    dd1, dd2 = pair_downsampler(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, dd1) + F.mse_loss(pred2, dd2))

    return loss_res + loss_cons


def _estimate(net, y):
    with torch.no_grad():
        return (y - net(y)).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def denoise_image(noisy_hwc, iterations=2000, lr=1e-3, chan_embed=48,
                  seed=0, device="cpu", clean_hwc=None, on_step=None):
    """Train from scratch on one H x W x 3 image in [0, 1] and denoise it.

    clean_hwc : optional, to report PSNR and SSIM
    on_step   : optional callback(step, psnr, estimate), called every step;
                psnr and estimate are given at step 1 and every 100 steps,
                otherwise None (used by the live demo)
    """
    torch.manual_seed(seed)
    net = TinyNet(noisy_hwc.shape[2], chan_embed).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    y = torch.from_numpy(np.ascontiguousarray(noisy_hwc)).permute(2, 0, 1)
    y = y.unsqueeze(0).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)

    t0 = time.time()
    for step in range(1, iterations + 1):
        opt.zero_grad(set_to_none=True)
        loss = loss_func(net, y)
        loss.backward()
        opt.step()
        sched.step()
        if on_step is not None:
            cur, est = None, None
            if step == 1 or step % 100 == 0:
                est = _estimate(net, y)
                if clean_hwc is not None:
                    cur = psnr(to_uint8(clean_hwc), to_uint8(est))
            on_step(step, cur, est)

    out = _estimate(net, y)
    info = {"params": n_params, "iterations": iterations,
            "seconds": round(time.time() - t0, 1)}
    if clean_hwc is not None:
        info["psnr"] = psnr(to_uint8(clean_hwc), to_uint8(out))
        info["ssim"] = ssim(to_uint8(clean_hwc), to_uint8(out))
        info["psnr_noisy"] = psnr(to_uint8(clean_hwc), to_uint8(noisy_hwc))
    return out, info
