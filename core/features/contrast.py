"""Contrast-based feature using global intensity deviation."""

from __future__ import annotations

import numpy as np

from .feature_base import Feature


class ContrastFeature(Feature):
    """Compute per-pixel contrast as deviation from the global mean intensity.

    For color images, a simple channel mean converts to grayscale. The output is
    a 2D map where higher values indicate larger deviation from the global mean.
    Normalization to [0, 1] is handled by the base class.
    """

    def _compute(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(np.float32, copy=False)
        elif image.ndim == 3:
            gray = image.astype(np.float32, copy=False).mean(axis=-1)
        else:
            raise ValueError("image must be 2D or 3D (H, W) or (H, W, C)")

        mean_intensity = float(gray.mean())
        return np.abs(gray - mean_intensity)
