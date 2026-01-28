"""Abstract base class for visual attention features.

This module defines the minimal interface and guarantees for feature modules in
the rule-based attention engine. Features must be deterministic, stateless, and
return a 2D attention map normalized to [0, 1]. No I/O, visualization, or
model-based logic belongs here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np


class Feature(ABC):
    """Base class for all visual attention features.

    Responsibilities:
    - Accept an input image as a NumPy array.
    - Produce a 2D attention map aligned with the input image.
    - Ensure the output is normalized to [0, 1].

    Constraints:
    - Deterministic: same input must produce the same output.
    - Stateless: no mutable instance state; configuration is fixed in code or
      provided externally without per-call mutation.
    - No I/O, visualization, or model-based logic.
    """

    __slots__: Final = ()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Compute a normalized attention map for the given image."""
        return self.compute(image)

    def compute(self, image: np.ndarray) -> np.ndarray:
        """Compute a normalized attention map for the given image.

        Subclasses implement `_compute` and return a 2D array. This wrapper
        validates inputs, enforces determinism-friendly normalization, and
        validates outputs.
        """
        self._validate_input(image)
        attention = self._compute(image)
        attention = self._normalize(attention)
        self._validate_output(attention, image)
        return attention

    @abstractmethod
    def _compute(self, image: np.ndarray) -> np.ndarray:
        """Return a raw 2D attention map for the given image.

        Implementations must be deterministic and avoid side effects.
        """

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim < 2:
            raise ValueError("image must have at least 2 dimensions")
        if not np.isfinite(image).all():
            raise ValueError("image must contain only finite values")

    @staticmethod
    def _normalize(attention: np.ndarray) -> np.ndarray:
        """Normalize a 2D attention map to [0, 1] deterministically."""
        attention = np.asarray(attention, dtype=np.float32)
        if attention.ndim != 2:
            raise ValueError("attention map must be 2D")
        if not np.isfinite(attention).all():
            raise ValueError("attention map must contain only finite values")

        min_val = float(attention.min())
        max_val = float(attention.max())
        if max_val == min_val:
            return np.zeros_like(attention, dtype=np.float32)
        normalized = (attention - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def _validate_output(attention: np.ndarray, image: np.ndarray) -> None:
        if attention.shape != image.shape[:2]:
            raise ValueError(
                "attention map must match image height and width (image.shape[:2])"
            )
