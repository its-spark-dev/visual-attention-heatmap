"""Weighted-sum fusion for attention feature maps."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from core.features.feature_base import Feature


def fuse_features(
    features: Sequence[Feature],
    image: np.ndarray,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    """Fuse multiple features into a single attention map using weighted sum.

    Args:
        features: Sequence of Feature instances.
        image: Input image as a NumPy array.
        weights: Optional iterable of weights; defaults to equal weights.

    Returns:
        A 2D NumPy array representing the fused attention map.
    """
    if not features:
        raise ValueError("features must be a non-empty sequence")

    if weights is None:
        weights_array = np.ones(len(features), dtype=np.float32)
    else:
        weights_array = np.asarray(list(weights), dtype=np.float32)
        if weights_array.ndim != 1:
            raise ValueError("weights must be a 1D sequence")
        if weights_array.size != len(features):
            raise ValueError("weights must match the number of features")

    total = float(weights_array.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights_array = weights_array / total

    fused = np.zeros(image.shape[:2], dtype=np.float32)
    for feature, weight in zip(features, weights_array):
        fused += weight * feature(image)

    return np.clip(fused, 0.0, 1.0)
