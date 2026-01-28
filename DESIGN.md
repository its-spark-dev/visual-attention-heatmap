# Design Document

This document defines the architectural principles and design constraints
for the visual-attention-heatmap project.
It serves as a reference for maintaining consistency, clarity,
and long-term extensibility across the codebase.

---

## 1. Core Philosophy

- The project is centered around a reusable **core attention engine**
- The core must remain simple, interpretable, and extensible
- Rule-based methods form the foundation of the system
- Architectural clarity and separation of concerns take priority over performance

---

## 2. Core Responsibilities

The core module is responsible for:

- Accepting an input image as a numeric array
- Computing multiple independent visual feature maps
- Normalizing feature maps to a common scale
- Combining feature maps using weighted fusion
- Producing a final attention (saliency) map

The core module must NOT:

- Determine image type (document, thumbnail, natural image)
- Apply use-case-specific heuristics or presets
- Perform semantic or contextual interpretation
- Handle UI, I/O, or web-related logic

---

## 3. Feature Design Rules

- Each visual feature is implemented as an independent module
- Features must not depend on one another
- Each feature outputs a 2D array normalized to the range [0, 1]
- Features must be deterministic and interpretable
- Features must not perform global decision-making

Example feature categories include:

- Edge detection
- Contrast estimation
- Center bias modeling
- Text region emphasis (e.g., OCR-based)

---

## 4. Attention Engine Rules

- The attention engine is responsible only for feature aggregation
- It accepts a list of feature instances as input
- Feature weights are defined and supplied externally
- The engine must not contain image-type logic or presets
- Final normalization is applied after feature fusion

---

## 5. Separation of Concerns

- Core logic resides in `core/`
- Feature implementations reside in `core/features/`
- Visualization utilities exist outside the core
- Application- or use-case-specific logic exists outside the core

This separation allows the core engine to be reused across
different contexts such as local analysis scripts,
research experiments, or future web services.

---

## 6. Scope Evolution

The project is expected to evolve incrementally:

- v0.x: Pure rule-based core attention engine
- v1.x: Optional advanced or learned features integrated as plugins
- Future: Validation against eye-tracking data
- Future: Web-based interfaces built on top of the core

Design decisions should favor long-term extensibility
over short-term convenience.
