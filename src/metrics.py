"""PSNR and SSIM on uint8 images, as in the Neighbor2Neighbor and Blind2Unblind
validation code. Computing them on floats instead shifts PSNR by a few
hundredths of a dB, the same order as the differences being compared.
"""
import numpy as np
from skimage.metrics import structural_similarity


def to_uint8(x):
    return np.clip(np.asarray(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def psnr(a_uint8, b_uint8):
    a = np.asarray(a_uint8, dtype=np.float64)
    b = np.asarray(b_uint8, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a_uint8, b_uint8):
    a = np.asarray(a_uint8)
    b = np.asarray(b_uint8)
    kw = dict(data_range=255, gaussian_weights=True, sigma=1.5,
              use_sample_covariance=False)
    if a.ndim == 3:
        kw["channel_axis"] = 2
    return float(structural_similarity(a, b, **kw))
