"""Noise2Noise (Lehtinen et al., ICML 2018).

Input: one noisy version of the crop. Target: a second, independent noisy
version of the same crop, as in the paper's synthetic experiments.
"""
import torch

from .base import Strategy


class Noise2Noise(Strategy):
    name = "noise2noise"

    def step(self, net, clean, noise, gen, progress):
        inp = noise(clean, gen)
        target = noise(clean, gen)
        return torch.mean((net(inp) - target) ** 2)
