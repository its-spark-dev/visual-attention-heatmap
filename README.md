# visual-attention-heatmap

A rule-based visual attention simulation engine that generates saliency heatmaps
and evaluation metrics for images, without requiring eye-tracking hardware.

This project focuses on building a modular and extensible core engine that
approximates where human visual attention is likely to concentrate in an image,
using interpretable, rule-based visual features.

---

## Motivation

Understanding visual attention is important in many domains such as document
design, user interfaces, advertising, and content creation (e.g., thumbnails).
However, real eye-tracking experiments require specialized hardware and controlled
environments.

This project aims to provide a lightweight and interpretable alternative by
simulating visual attention patterns using image processing techniques.

---

## Goals

- Simulate human visual attention using rule-based visual features
- Generate attention (saliency) heatmaps from a single image
- Produce quantitative metrics describing attention distribution
- Design the core as a reusable and extensible engine
- Allow future integration of AI-based features

---

## Non-goals (for now)

- Training deep learning models
- Using real eye-tracking datasets
- Claiming psychological or neuroscientific accuracy
- Deploying a web application (planned later)

---

## High-level Pipeline

1. Input image
2. Extraction of multiple visual feature maps
   (e.g., contrast, edges, center bias, text regions)
3. Normalization of each feature map
4. Weighted fusion into a single attention map
5. Visualization as a heatmap and overlay
6. Computation of attention-related metrics

---

## Core Design Philosophy

- The core engine is **image-type agnostic**
- Each visual feature is implemented as an independent module
- All feature maps are normalized to a common scale
- The core performs no heuristic or semantic interpretation
- Image-type decisions and presets are handled outside the core

---

## Planned Use Cases

- Analysis of document images (e.g., text-heavy layouts)
- Evaluation and comparison of YouTube thumbnails
- Visual design and layout analysis
- Research prototyping in visual attention and HCI

---

## Project Status

- v0.1: Core attention engine (rule-based)
- v0.2: Document-aware features (OCR-based)
- v0.3: Thumbnail-specific metrics and presets
- Future: Web API and interactive visualization

---

## Notes

This project prioritizes clarity, modularity, and extensibility over
end-to-end performance. The goal is to build a strong and interpretable
foundation before adding more complex models.

---

## Available Features

- `CenterBiasFeature`: center-weighted attention with smooth radial falloff.
- `ContrastFeature`: per-pixel deviation from global mean intensity.
- `EdgeDensityFeature`: gradient magnitude from simple finite differences.
- `CenterSurroundFeature`: DoG-like center–surround response.

---

## Quick Example

```python
import numpy as np

from core.features import (
    CenterBiasFeature,
    CenterSurroundFeature,
    ContrastFeature,
    EdgeDensityFeature,
)
from core.fusion import fuse_features

image = np.zeros((240, 320, 3), dtype=np.float32)
feature = CenterBiasFeature()
attention_map = feature(image)
print(attention_map.shape)  # (240, 320)

contrast = ContrastFeature()
edges = EdgeDensityFeature()
surround = CenterSurroundFeature()
fused_map = fuse_features(
    [feature, contrast, edges, surround],
    image,
    weights=[0.5, 0.2, 0.2, 0.1],
)
print(fused_map.shape)  # (240, 320)
```
