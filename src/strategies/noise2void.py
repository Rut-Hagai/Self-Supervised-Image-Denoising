"""Noise2Void (Krull et al., CVPR 2019).

Input: the noisy image with some pixels masked. Target: the same noisy image,
with the loss computed only at the masked pixels. The masked positions are
chosen by stratified sampling, 64 per 64 x 64 window as in the paper.

A masked pixel is replaced by the weighted mean of its neighbours (DONUT in
base.py). The paper instead copies the value of one randomly chosen neighbour.

Reference implementation: https://github.com/juglab/n2v (BSD 3-Clause).
"""
import torch

from .base import Strategy, neighbourhood_average


class Noise2Void(Strategy):
    name = "noise2void"

    def __init__(self, n_masked=64, window=64):
        self.n_masked = n_masked
        self.window = window

    def _mask(self, noisy, gen):
        """One masked pixel per cell of a regular grid, at a random offset."""
        n, c, h, w = noisy.shape
        rate = self.n_masked / float(self.window ** 2)
        cell = max(1, int(round((1.0 / rate) ** 0.5)))
        gh, gw = h // cell, w // cell
        oy = torch.randint(0, cell, (n, 1, gh, gw), generator=gen,
                           device=noisy.device)
        ox = torch.randint(0, cell, (n, 1, gh, gw), generator=gen,
                           device=noisy.device)
        ys = (torch.arange(gh, device=noisy.device).view(1, 1, gh, 1) * cell + oy)
        xs = (torch.arange(gw, device=noisy.device).view(1, 1, 1, gw) * cell + ox)
        mask = torch.zeros(n, 1, h, w, dtype=torch.bool, device=noisy.device)
        bidx = torch.arange(n, device=noisy.device).view(n, 1, 1, 1).expand(n, 1, gh, gw)
        mask[bidx.reshape(-1), 0,
             ys.expand(n, 1, gh, gw).reshape(-1),
             xs.expand(n, 1, gh, gw).reshape(-1)] = True
        return mask

    def step(self, net, clean, noise, gen, progress):
        noisy = noise(clean, gen)
        mask = self._mask(noisy, gen)
        net_input = torch.where(mask, neighbourhood_average(noisy), noisy)
        out = net(net_input)
        m = mask.expand_as(noisy)
        return torch.sum(((out - noisy) ** 2) * m) / torch.clamp(m.sum(), min=1)
