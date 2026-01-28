"""Center-surround feature using Difference of Gaussians (DoG).

Biological motivation: early visual processing in the retina and LGN uses
center-surround receptive fields, responding to local contrast by comparing
fine-scale (center) activity against broader (surround) activity.
"""

from __future__ import annotations

import numpy as np

from .feature_base import Feature


class CenterSurroundFeature(Feature):
    """Compute a center-surround response via a DoG-like filter.

    This implementation approximates Gaussian blur with separable 1D kernels
    implemented in NumPy. The output is the magnitude of the difference between
    a narrow and wide blur, highlighting local contrast changes.
    """

    def __init__(self, sigma_center: float = 1.0, sigma_surround: float = 2.5) -> None:
        if sigma_center <= 0.0 or sigma_surround <= 0.0:
            raise ValueError("sigma values must be positive")
        if sigma_surround <= sigma_center:
            raise ValueError("sigma_surround must be greater than sigma_center")
        self._sigma_center = float(sigma_center)
        self._sigma_surround = float(sigma_surround)

    def _compute(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(np.float32, copy=False)
        elif image.ndim == 3:
            gray = image.astype(np.float32, copy=False).mean(axis=-1)
        else:
            raise ValueError("image must be 2D or 3D (H, W) or (H, W, C)")

        blurred_center = _gaussian_blur(gray, self._sigma_center)
        blurred_surround = _gaussian_blur(gray, self._sigma_surround)
        return np.abs(blurred_center - blurred_surround)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        return np.array([1.0], dtype=np.float32)
    return kernel / kernel_sum


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    pad = kernel.size // 2

    padded_rows = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
    blurred_rows = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=1,
        arr=padded_rows,
    )

    padded_cols = np.pad(blurred_rows, ((pad, pad), (0, 0)), mode="edge")
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"),
        axis=0,
        arr=padded_cols,
    )

    return blurred.astype(np.float32, copy=False)
