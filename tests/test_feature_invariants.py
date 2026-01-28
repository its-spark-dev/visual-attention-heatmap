import numpy as np
import pytest

from core.features import (
    CenterBiasFeature,
    CenterSurroundFeature,
    ContrastFeature,
    EdgeDensityFeature,
)


@pytest.mark.parametrize(
    "feature",
    [
        CenterBiasFeature(),
        CenterSurroundFeature(),
        ContrastFeature(),
        EdgeDensityFeature(),
    ],
)
def test_feature_output_shape(feature):
    image = np.arange(24, dtype=np.float32).reshape(4, 6)
    attention = feature(image)
    assert attention.shape == (4, 6)


@pytest.mark.parametrize(
    "feature",
    [
        CenterBiasFeature(),
        CenterSurroundFeature(),
        ContrastFeature(),
        EdgeDensityFeature(),
    ],
)
def test_feature_normalization_range(feature):
    image = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    attention = feature(image)
    assert attention.min() >= 0.0
    assert attention.max() <= 1.0


@pytest.mark.parametrize(
    "feature",
    [
        CenterBiasFeature(),
        CenterSurroundFeature(),
        ContrastFeature(),
        EdgeDensityFeature(),
    ],
)
def test_feature_determinism(feature):
    image = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    attention_first = feature(image)
    attention_second = feature(image)
    assert np.array_equal(attention_first, attention_second)
