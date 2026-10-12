"""What separates the methods: what goes into the network and what the loss
compares the output against. Everything else is shared.
"""
import torch
import torch.nn.functional as F

# Weighted mean of the eight neighbours, centre excluded. A masked pixel is
# replaced by this rather than by a constant, so the network cannot recognise
# the masked positions. The kernel is the one Blind2Unblind uses.
DONUT = torch.tensor([[0.5, 1.0, 0.5],
                      [1.0, 0.0, 1.0],
                      [0.5, 1.0, 0.5]], dtype=torch.float32)
DONUT = (DONUT / DONUT.sum()).view(1, 1, 3, 3)


def neighbourhood_average(x):
    n, c, h, w = x.shape
    k = DONUT.to(x.device, x.dtype)
    return F.conv2d(x.reshape(n * c, 1, h, w), k, padding=1).view_as(x)


class Strategy:
    name = "base"

    def step(self, net, clean, noise, gen, progress):
        """Return the loss for one training step.

        clean    : B x 3 x H x W tensor in [0, 1]
        noise    : function adding noise to a tensor
        gen      : torch.Generator
        progress : fraction of training done, in [0, 1]
        """
        raise NotImplementedError
