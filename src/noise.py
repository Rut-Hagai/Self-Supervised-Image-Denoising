"""Noise models, parameterised as in Neighbor2Neighbor's AugmentNoise:

  gauss<sigma>   additive Gaussian, sigma on the 0-255 scale
  poisson<lam>   y = Poisson(lam * x) / lam, so a larger lam means weaker noise
"""
import numpy as np


class NoiseModel:
    def __init__(self, style):
        self.style = style
        if style.startswith("gauss"):
            self.kind = "gauss"
            self.param = float(style[len("gauss"):]) / 255.0
        elif style.startswith("poisson"):
            self.kind = "poisson"
            self.param = float(style[len("poisson"):])
        else:
            raise ValueError(f"unknown noise style: {style!r}")

    def __call__(self, x, rng):
        """x is float32 in [0, 1], any shape. Returns a noisy copy."""
        if self.kind == "gauss":
            return np.asarray(x + rng.normal(size=x.shape) * self.param,
                              dtype=np.float32)
        return np.asarray(rng.poisson(self.param * x) / self.param,
                          dtype=np.float32)

    @property
    def sigma(self):
        """Standard deviation on the 0-1 scale. Gaussian noise only."""
        if self.kind != "gauss":
            raise ValueError(f"{self.style} has no fixed standard deviation")
        return self.param
