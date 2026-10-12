"""Training data: random crops from a folder of clean images.

Noise is added inside the training step, not here, because Noise2Noise needs
two independent noise draws on the same clean crop.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# filter by extension so that Thumbs.db and .DS_Store are skipped
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
                    ".ppm", ".pgm", ".webp"}


def image_files(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)


class CleanPatches(Dataset):
    def __init__(self, folder, patch=128, length=None, seed=0):
        self.files = image_files(folder)
        if not self.files:
            raise FileNotFoundError(f"no images in {folder}")
        self.patch = patch
        self.length = length or len(self.files)
        self.seed = seed
        self._cache = {}

    def __len__(self):
        return self.length

    def _image(self, i):
        if i not in self._cache:
            im = Image.open(self.files[i]).convert("RGB")
            self._cache[i] = np.asarray(im, dtype=np.float32) / 255.0
        return self._cache[i]

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1000003 + idx)
        im = self._image(int(rng.integers(len(self.files))))
        h, w, _ = im.shape
        p = self.patch
        if h < p or w < p:
            raise ValueError(f"image smaller than patch size {p}")
        y = int(rng.integers(h - p + 1))
        x = int(rng.integers(w - p + 1))
        crop = im[y:y + p, x:x + p]
        k = int(rng.integers(4))
        crop = np.rot90(crop, k, axes=(0, 1))
        if rng.integers(2):
            crop = crop[:, ::-1]
        crop = np.ascontiguousarray(crop)
        return torch.from_numpy(crop).permute(2, 0, 1)


def torch_noise(style):
    """A noise function on tensors, for use inside a training step."""
    if style.startswith("gauss"):
        sigma = float(style[len("gauss"):]) / 255.0

        def add(x, gen):
            return x + torch.randn(x.shape, generator=gen,
                                   device=x.device, dtype=x.dtype) * sigma
    elif style.startswith("poisson"):
        lam = float(style[len("poisson"):])

        def add(x, gen):
            return torch.poisson(lam * x, generator=gen) / lam
    else:
        raise ValueError(f"unknown noise style {style!r}")
    return add
