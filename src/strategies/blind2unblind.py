"""Blind2Unblind (Wang et al., CVPR 2022).

Global mask mapper: four masked versions of the image, one per position in a
2x2 cell, together cover every pixel once. Re-visible loss: the blind output
is mixed with a visible branch (stop-gradient) weighted by lambda.

Written from the paper's equations. The official repository
(https://github.com/zejinwang/Blind2Unblind) has no licence file, so it was
read to check behaviour and none of its code is used.
"""
import torch

from .base import Strategy, neighbourhood_average


class Blind2Unblind(Strategy):
    name = "blind2unblind"

    def __init__(self, cell=2, eta=1.0, lam_start=2.0, lam_end=20.0,
                 ramp_begins_at=0.8):
        self.cell = cell
        self.eta = eta
        self.lam_start = lam_start
        self.lam_end = lam_end
        self.ramp_begins_at = ramp_begins_at

    def _masked_versions(self, noisy):
        """cell^2 masked versions and their masks, stacked on the batch axis."""
        n, c, h, w = noisy.shape
        s = self.cell
        replacement = neighbourhood_average(noisy)
        inputs, masks = [], []
        for i in range(s):
            for j in range(s):
                m = torch.zeros(1, 1, h, w, device=noisy.device,
                                dtype=noisy.dtype)
                m[:, :, i::s, j::s] = 1.0
                masks.append(m.expand(n, 1, h, w))
                inputs.append(torch.where(m.bool(), replacement, noisy))
        return torch.cat(inputs, 0), torch.cat(masks, 0)

    def lam(self, progress):
        """lam_start, then a linear ramp to lam_end over the end of training."""
        if progress <= self.ramp_begins_at:
            return self.lam_start
        t = (progress - self.ramp_begins_at) / (1.0 - self.ramp_begins_at)
        return self.lam_start + t * (self.lam_end - self.lam_start)

    def step(self, net, clean, noise, gen, progress):
        noisy = noise(clean, gen)
        n, c, h, w = noisy.shape

        inputs, masks = self._masked_versions(noisy)
        outputs = net(inputs)
        blind = (outputs * masks).view(self.cell ** 2, n, c, h, w).sum(dim=0)

        with torch.no_grad():
            visible = net(noisy)

        diff = blind - noisy
        exp_diff = visible - noisy
        lam = self.lam(progress)

        # ||h(f(Omega_y)) + lam * f(y) - (lam + 1) y||^2
        revisible = diff + lam * exp_diff
        loss_blind = self.eta * torch.mean(diff ** 2)
        loss_rev = torch.mean(revisible ** 2)
        return loss_blind + loss_rev
