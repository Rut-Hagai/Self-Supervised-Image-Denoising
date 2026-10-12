"""Neighbor2Neighbor (Huang et al., CVPR 2021).

Input: one of two sub-images drawn from the noisy image, one adjacent pair of
pixels per 2x2 cell. Target: the other sub-image. The regularisation term
corrects for the small signal difference between the two neighbours.

The sampler is the authors' code (see src/common.py).
"""
import torch

from ..common import generate_mask_pair, generate_subimages
from .base import Strategy


class Neighbor2Neighbor(Strategy):
    name = "neighbor2neighbor"

    def __init__(self, lambda1=1.0, lambda2=1.0, increase_ratio=2.0):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.increase_ratio = increase_ratio

    def step(self, net, clean, noise, gen, progress):
        noisy = noise(clean, gen)

        mask1, mask2 = generate_mask_pair(noisy, gen)
        sub1 = generate_subimages(noisy, mask1)
        sub2 = generate_subimages(noisy, mask2)

        # stop-gradient: the signal difference is estimated, not trained
        with torch.no_grad():
            denoised = net(noisy)
            sub1_den = generate_subimages(denoised, mask1)
            sub2_den = generate_subimages(denoised, mask2)

        out = net(sub1)
        diff = out - sub2
        exp_diff = sub1_den - sub2_den

        gamma = progress * self.increase_ratio  # ramps from 0 to increase_ratio
        loss_rec = torch.mean(diff ** 2)
        loss_reg = gamma * torch.mean((diff - exp_diff) ** 2)
        return self.lambda1 * loss_rec + self.lambda2 * loss_reg
