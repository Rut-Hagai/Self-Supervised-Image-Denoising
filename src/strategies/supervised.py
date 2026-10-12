"""Supervised training, the reference. Input: noisy image. Target: clean image."""
import torch

from .base import Strategy


class Supervised(Strategy):
    name = "supervised"

    def step(self, net, clean, noise, gen, progress):
        noisy = noise(clean, gen)
        return torch.mean((net(noisy) - clean) ** 2)
