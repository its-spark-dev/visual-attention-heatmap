"""Edge-density feature using simple finite differences."""

from __future__ import annotations

import numpy as np

from .feature_base import Feature


class EdgeDensityFeature(Feature):
    """Estimate edge strength via gradient magnitude.

    The feature converts color images to grayscale by channel mean, then
    computes gradients using simple central differences. The raw gradient
    magnitude is returned; normalization is handled by the base class.
    """

    def _compute(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(np.float32, copy=False)
        elif image.ndim == 3:
            gray = image.astype(np.float32, copy=False).mean(axis=-1)
        else:
            raise ValueError("image must be 2D or 3D (H, W) or (H, W, C)")

        height, width = gray.shape
        if height < 2 or width < 2:
            return np.zeros_like(gray, dtype=np.float32)

        padded = np.pad(gray, pad_width=1, mode="edge")
        gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
        gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
        return np.sqrt(gx**2 + gy**2)
