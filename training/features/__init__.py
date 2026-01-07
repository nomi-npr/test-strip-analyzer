"""Feature extraction module for test strip analysis."""

from .color_roi_detection import (
    ColorROIConfig,
    ColorROIResult,
    ColorTransitionResult,
    detect_color_rois,
    detect_color_transition,
    detect_saturation_step,
    visualize_color_transition,
)
from .extraction import (
    FeatureConfig,
    ImageFeatures,
    extract_features,
    extract_features_from_file,
)
from .line_detection import (
    DynamicLineDetectionResult,
    DynamicROIConfig,
    LineDetectionResult,
    LineRegion,
    ROIConfig,
    detect_line_regions,
    detect_line_regions_dynamic,
    visualize_regions,
)
from .membrane_detection import (
    MembraneDetectionConfig,
    MembraneDetectionResult,
    find_membrane_start,
    visualize_membrane_detection,
)

__all__ = [
    # Color-based ROI detection (saturation step detection - recommended)
    "ColorROIConfig",
    "ColorROIResult",
    "ColorTransitionResult",
    "detect_color_rois",
    "detect_color_transition",
    "detect_saturation_step",
    "visualize_color_transition",
    # Line detection (static)
    "ROIConfig",
    "LineRegion",
    "LineDetectionResult",
    "detect_line_regions",
    "visualize_regions",
    # Line detection (dynamic - membrane-based)
    "DynamicROIConfig",
    "DynamicLineDetectionResult",
    "detect_line_regions_dynamic",
    # Membrane detection (texture-based - deprecated)
    "MembraneDetectionConfig",
    "MembraneDetectionResult",
    "find_membrane_start",
    "visualize_membrane_detection",
    # Feature extraction
    "FeatureConfig",
    "ImageFeatures",
    "extract_features",
    "extract_features_from_file",
]
