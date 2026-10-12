"""The two source images and the noise used by every figure.

Both images ship with scikit-image:

    cameraman   CC0, by Lav Varshney
    astronaut   Eileen Collins, NASA Great Images, public domain

All noise comes from seeded generators, so the figures are reproducible.
"""
import numpy as np

SEED = 0


def _load(name, size):
    from skimage import data
    from skimage.transform import resize
    a = getattr(data, name)().astype(np.float64) / 255.0
    if size and size != a.shape[0]:
        a = resize(a, (size, size), anti_aliasing=True)
    return np.clip(a, 0, 1)


def camera(size=256):
    """Greyscale, floats in [0, 1]."""
    return _load("camera", size)


def astronaut(size=256):
    """Colour, floats in [0, 1]."""
    return _load("astronaut", size)


def rng(seed=SEED):
    return np.random.default_rng(seed)


def gauss(x, sigma, gen=None):
    """Gaussian noise of standard deviation `sigma`, in 0 to 255 units."""
    gen = gen or rng()
    return np.clip(x + gen.normal(0, sigma / 255.0, x.shape), 0, 1)


def poisson(x, peak, gen=None):
    """Poisson noise at a given photon count for the brightest pixel."""
    gen = gen or rng()
    return np.clip(gen.poisson(np.clip(x, 0, 1) * peak) / float(peak), 0, 1)


def psnr(x, y):
    """PSNR on 8-bit values, as in src/metrics.py."""
    a = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float64)
    b = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
