import numpy as np

from core.features.feature_base import Feature
from core.fusion import fuse_features


class _GradientFeature(Feature):
    def _compute(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        row = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
        return np.tile(row, (height, 1))


class _InverseGradientFeature(Feature):
    def _compute(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        row = np.linspace(1.0, 0.0, num=width, dtype=np.float32)
        return np.tile(row, (height, 1))


def test_fuse_features_equal_weights():
    image = np.zeros((4, 6, 3), dtype=np.float32)
    fused = fuse_features(
        [_GradientFeature(), _InverseGradientFeature()],
        image,
    )
    assert fused.shape == (4, 6)
    assert np.allclose(fused, 0.5, atol=1e-6)


def test_fuse_features_custom_weights():
    image = np.zeros((3, 5, 1), dtype=np.float32)
    fused = fuse_features(
        [_GradientFeature(), _InverseGradientFeature()],
        image,
        weights=[0.25, 0.75],
    )
    assert fused.shape == (3, 5)
    assert np.isfinite(fused).all()
    assert fused.min() >= 0.0
    assert fused.max() <= 1.0
