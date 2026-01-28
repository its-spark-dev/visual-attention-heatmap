import numpy as np

from core.features import CenterSurroundFeature


def test_center_surround_output_shape_and_range():
    image = np.zeros((5, 7, 3), dtype=np.float32)
    feature = CenterSurroundFeature()
    attention = feature(image)

    assert attention.shape == (5, 7)
    assert np.isfinite(attention).all()
    assert attention.min() >= 0.0
    assert attention.max() <= 1.0
