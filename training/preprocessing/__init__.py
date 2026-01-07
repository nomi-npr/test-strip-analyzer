"""Preprocessing module for test strip images."""

from .localization import (
    LocalizationConfig,
    LocalizationResult,
    crop_strip,
    find_strip_contour,
    localize_strip,
)
from .normalization import (
    NormalizationConfig,
    NormalizationResult,
    apply_clahe,
    apply_white_balance,
    ensure_horizontal,
    normalize_brightness,
    normalize_image,
    resize_image,
)
from .orientation import (
    ArrowDetectionConfig,
    ArrowDetectionResult,
    ArrowDirection,
    detect_arrows,
    orient_image,
    rotate_image,
)
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    PreprocessingPipeline,
    create_default_pipeline,
    preprocess_image,
)

__all__ = [
    # Orientation
    "ArrowDetectionConfig",
    "ArrowDetectionResult",
    "ArrowDirection",
    "detect_arrows",
    "orient_image",
    "rotate_image",
    # Localization
    "LocalizationConfig",
    "LocalizationResult",
    "find_strip_contour",
    "crop_strip",
    "localize_strip",
    # Normalization
    "NormalizationConfig",
    "NormalizationResult",
    "resize_image",
    "apply_white_balance",
    "apply_clahe",
    "normalize_brightness",
    "normalize_image",
    "ensure_horizontal",
    # Pipeline
    "PipelineConfig",
    "PipelineResult",
    "PreprocessingPipeline",
    "create_default_pipeline",
    "preprocess_image",
]
