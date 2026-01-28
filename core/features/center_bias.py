"""Center-bias feature.

Produces higher attention near the image center using a smooth radial falloff.
Pure NumPy, deterministic, and stateless.
"""

from __future__ import annotations

import numpy as np

from .feature_base import Feature


class CenterBiasFeature(Feature):
    """Assigns higher attention to pixels closer to the image center.

    The bias is computed as a radial falloff based on Euclidean distance from
    the center. The base class normalizes the result to [0, 1].
    """

    def _compute(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("image must have non-zero height and width")

        y_coords, x_coords = np.indices((height, width), dtype=np.float32)
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0

        dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        max_dist = float(dist.max())
        if max_dist == 0.0:
            return np.ones((height, width), dtype=np.float32)
        return 1.0 - (dist / max_dist)
