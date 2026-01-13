"""Rule-based strip analysis using line intensity profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RuleBasedConfig:
    band_start_ratio: float = 0.35
    band_end_ratio: float = 0.95
    control_threshold: float = 0.02
    test_threshold: float = 0.005
    blur_sigma: float = 3.0
    min_foreground_fraction: float = 0.01
    use_dynamic_band: bool = True
    use_membrane_anchor: bool = True
    band_margin_ratio: float = 0.03
    membrane_offset_ratio: float = 0.03
    min_band_height_ratio: float = 0.25
    inner_width_ratio: float = 0.6
    exclude_blue: bool = True
    membrane_window_ratio: float = 0.08
    membrane_search_start_ratio: float = 0.05
    membrane_search_end_ratio: float = 0.6
    membrane_brightness_weight: float = 0.5
    membrane_texture_weight: float = 0.35
    membrane_blue_weight: float = 0.15
    blue_row_threshold: float = 0.2
    blue_search_ratio: float = 0.6
    blue_min_run: int = 10
    blue_hue_low: int = 90
    blue_hue_high: int = 140
    blue_saturation_min: int = 60
    blue_value_min: int = 40
    peak_prominence: float = 0.015
    peak_window: int = 31
    max_peaks: int = 6
    control_band: tuple[float, float] = (0.05, 0.35)
    max_peak_width_ratio: float = 0.08
    use_template_matching: bool = True
    line_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    control_window: tuple[float, float] = (0.15, 0.45)
    test1_window: tuple[float, float] = (0.45, 0.65)
    test2_window: tuple[float, float] = (0.65, 0.90)
    line_percentile: float = 85.0
    background_percentile: float = 50.0
    background_window_ratio: float = 0.08
    template_spacing: tuple[float, float] = (0.16, 0.16)
    template_tolerance: float = 0.10
    peak_min_ratio: float = 0.7
    peak_min_distance_ratio: float = 0.08


@dataclass
class LineResult:
    control_strength: float
    test1_strength: float
    test2_strength: float
    control_row: int | None
    test1_row: int | None
    test2_row: int | None
    membrane_start: int | None
    membrane_end: int | None
    band_start: int
    band_end: int
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.generic):
                data[key] = value.item()
        return data


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0
        white = np.full_like(rgb, 255)
        image = (rgb * alpha[..., None] + white * (1.0 - alpha[..., None])).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


def segment_strip_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    background = (gray > 240) & (hsv[:, :, 1] < 20)
    mask = (~background).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_component(mask)
    return mask


def _rotate(image: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated_image, rotated_mask


def rectify_strip(image: np.ndarray, mask: np.ndarray, config: RuleBasedConfig) -> tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return image, mask
    points = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    angle = rect[2]
    if width < height:
        rotate_angle = angle
    else:
        rotate_angle = angle + 90
    rotated_image, rotated_mask = _rotate(image, mask, rotate_angle)
    ys, xs = np.where(rotated_mask > 0)
    if ys.size == 0:
        return rotated_image, rotated_mask
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped_image = rotated_image[y0 : y1 + 1, x0 : x1 + 1]
    cropped_mask = rotated_mask[y0 : y1 + 1, x0 : x1 + 1]
    if cropped_image.shape[1] > cropped_image.shape[0]:
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        cropped_mask = cv2.rotate(cropped_mask, cv2.ROTATE_90_CLOCKWISE)
    cropped_image, cropped_mask = _ensure_upright(cropped_image, cropped_mask, config)
    return cropped_image, cropped_mask


def _blue_score(image: np.ndarray, config: RuleBasedConfig) -> float:
    if image.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue_mask = (
        (h >= config.blue_hue_low)
        & (h <= config.blue_hue_high)
        & (s >= config.blue_saturation_min)
        & (v >= config.blue_value_min)
    )
    return float(np.mean(blue_mask))


def _blue_row_fraction(image: np.ndarray, config: RuleBasedConfig) -> np.ndarray:
    if image.size == 0:
        return np.zeros((0,), dtype=np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blue_mask = (
        (h >= config.blue_hue_low)
        & (h <= config.blue_hue_high)
        & (s >= config.blue_saturation_min)
        & (v >= config.blue_value_min)
    )
    return blue_mask.mean(axis=1).astype(np.float32)


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def _row_mask_within_strip(mask: np.ndarray, config: RuleBasedConfig) -> tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size > 0:
        x0, x1 = xs.min(), xs.max()
    else:
        x0, x1 = 0, mask.shape[1] - 1
    inner_width = max(1, int((x1 - x0 + 1) * config.inner_width_ratio))
    margin = max(0, int(((x1 - x0 + 1) - inner_width) / 2))
    inner_x0 = x0 + margin
    inner_x1 = x1 - margin
    return inner_x0, inner_x1


def _find_membrane_region(
    image: np.ndarray, mask: np.ndarray, config: RuleBasedConfig
) -> tuple[int, int] | None:
    height = image.shape[0]
    if height < 10:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.abs(sobel)
    blue_fraction = _blue_row_fraction(image, config)

    inner_x0, inner_x1 = _row_mask_within_strip(mask, config)

    brightness_row = np.full(height, np.nan, dtype=np.float32)
    texture_row = np.full(height, np.nan, dtype=np.float32)
    for row in range(height):
        row_mask = mask[row] > 0
        row_mask[:inner_x0] = False
        row_mask[inner_x1 + 1 :] = False
        if row_mask.any():
            brightness_row[row] = float(np.mean(gray[row, row_mask]))
            texture_row[row] = float(np.mean(texture[row, row_mask]))

    nan_mask = np.isnan(brightness_row)
    if nan_mask.all():
        return None
    brightness_row[nan_mask] = np.interp(
        np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), brightness_row[~nan_mask]
    )
    nan_mask = np.isnan(texture_row)
    if nan_mask.all():
        return None
    texture_row[nan_mask] = np.interp(
        np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), texture_row[~nan_mask]
    )

    brightness_norm = _normalize_feature(brightness_row)
    texture_norm = _normalize_feature(texture_row)
    blue_norm = _normalize_feature(blue_fraction)

    score = (
        config.membrane_brightness_weight * brightness_norm
        - config.membrane_texture_weight * texture_norm
        - config.membrane_blue_weight * blue_norm
    )

    window = max(3, int(height * config.membrane_window_ratio))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    score_smooth = np.convolve(score, kernel, mode="same")

    search_start = int(height * config.membrane_search_start_ratio)
    search_end = int(height * config.membrane_search_end_ratio)
    search_end = max(search_start + 1, min(search_end, height))

    window_scores = score_smooth.copy()
    if search_start > 0:
        window_scores[:search_start] = -np.inf
    if search_end < height:
        window_scores[search_end:] = -np.inf

    center = int(np.argmax(window_scores))
    half = window // 2
    start = max(0, center - half)
    end = min(height - 1, center + half)
    return start, end


def _find_handle_end(blue_fraction: np.ndarray, config: RuleBasedConfig) -> int | None:
    if blue_fraction.size == 0:
        return None
    rows = np.where(blue_fraction > config.blue_row_threshold)[0]
    if rows.size == 0:
        return None
    # Identify contiguous runs.
    runs = []
    start = rows[0]
    prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        runs.append((start, prev))
        start = row
        prev = row
    runs.append((start, prev))

    height = len(blue_fraction)
    search_end = int(height * config.blue_search_ratio)
    candidates = []
    for run_start, run_end in runs:
        if run_start > search_end:
            continue
        length = run_end - run_start + 1
        if length < config.blue_min_run:
            continue
        mean_fraction = float(np.mean(blue_fraction[run_start : run_end + 1]))
        candidates.append((length, mean_fraction, run_start, run_end))
    if not candidates:
        return None
    # Prefer longest run; tie-break on mean_fraction then earliest start.
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return int(candidates[0][3])


def _ensure_upright(
    image: np.ndarray, mask: np.ndarray, config: RuleBasedConfig
) -> tuple[np.ndarray, np.ndarray]:
    height = image.shape[0]
    if height < 10:
        return image, mask
    band = max(1, int(0.2 * height))
    top_score = _blue_score(image[:band], config)
    bottom_score = _blue_score(image[-band:], config)
    if bottom_score > top_score * 1.2:
        return (
            cv2.rotate(image, cv2.ROTATE_180),
            cv2.rotate(mask, cv2.ROTATE_180),
        )
    return image, mask


def _row_profile(image: np.ndarray, mask: np.ndarray, config: RuleBasedConfig) -> np.ndarray:
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    a_channel = lab[:, :, 1].astype(np.float32)
    a_channel = (a_channel - 128.0) / 128.0
    score = a_channel

    blue_mask = np.zeros((height, width), dtype=bool)
    if config.exclude_blue:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        blue_mask = (
            (h >= config.blue_hue_low)
            & (h <= config.blue_hue_high)
            & (s >= config.blue_saturation_min)
            & (v >= config.blue_value_min)
        )

    ys, xs = np.where(mask > 0)
    if xs.size > 0:
        x0, x1 = xs.min(), xs.max()
    else:
        x0, x1 = 0, width - 1
    inner_width = max(1, int((x1 - x0 + 1) * config.inner_width_ratio))
    margin = max(0, int(((x1 - x0 + 1) - inner_width) / 2))
    inner_x0 = x0 + margin
    inner_x1 = x1 - margin

    profile = np.full(image.shape[0], np.nan, dtype=np.float32)
    for row in range(image.shape[0]):
        row_mask = mask[row] > 0
        row_mask[:inner_x0] = False
        row_mask[inner_x1 + 1 :] = False
        if config.exclude_blue:
            row_mask = row_mask & (~blue_mask[row])
        if row_mask.any():
            row_vals = score[row, row_mask]
            profile[row] = float(np.percentile(row_vals, config.line_percentile))
    nan_mask = np.isnan(profile)
    if nan_mask.all():
        return np.zeros_like(profile)
    profile[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), profile[~nan_mask])

    if len(profile) >= 5:
        kernel_size = int(max(3, round(config.blur_sigma * 3)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel_size), config.blur_sigma).ravel()
    return profile


def _local_background(profile: np.ndarray, config: RuleBasedConfig) -> np.ndarray:
    if profile.size == 0:
        return profile
    window = max(3, int(len(profile) * config.background_window_ratio))
    if window % 2 == 0:
        window += 1
    half = window // 2
    background = np.empty_like(profile)
    for i in range(len(profile)):
        start = max(0, i - half)
        end = min(len(profile), i + half + 1)
        background[i] = np.percentile(profile[start:end], config.background_percentile)
    return background


def _segment_strengths(
    profile: np.ndarray, start: int, end: int
) -> tuple[float, int | None, float]:
    segment = profile[start:end]
    if segment.size == 0:
        return 0.0, None, 0.0
    baseline = float(np.median(segment))
    peak_idx = int(np.argmax(segment))
    peak_val = float(segment[peak_idx])
    strength = max(peak_val - baseline, 0.0)
    return strength, start + peak_idx, baseline


def _find_peaks(
    profile: np.ndarray, start: int, end: int, config: RuleBasedConfig
) -> list[int]:
    segment = profile[start:end]
    if segment.size < 3:
        return []
    baseline = float(np.median(segment))
    threshold = baseline + config.peak_prominence
    candidates = []
    max_width = max(1, int((end - start) * config.max_peak_width_ratio))
    for idx in range(1, len(segment) - 1):
        if segment[idx] > segment[idx - 1] and segment[idx] > segment[idx + 1]:
            if segment[idx] >= threshold:
                # Measure width at half-prominence to reject broad peaks.
                half_thresh = baseline + config.peak_prominence * 0.5
                left = idx
                while left > 0 and segment[left] >= half_thresh:
                    left -= 1
                right = idx
                while right < len(segment) - 1 and segment[right] >= half_thresh:
                    right += 1
                width = right - left + 1
                if width <= max_width:
                    candidates.append(idx)
    # sort by strength descending
    candidates.sort(key=lambda i: segment[i], reverse=True)
    selected: list[int] = []
    min_distance = max(1, (end - start) // 6)
    for idx in candidates:
        if all(abs(idx - chosen) >= min_distance for chosen in selected):
            selected.append(idx)
        if len(selected) >= config.max_peaks:
            break
    return [start + idx for idx in sorted(selected)]


def _find_peaks_in_window(
    profile: np.ndarray, start: int, end: int, config: RuleBasedConfig
) -> list[int]:
    return _find_peaks(profile, start, end, config)


def _strength_at(profile: np.ndarray, row: int, config: RuleBasedConfig) -> float:
    half = config.peak_window // 2
    start = max(0, row - half)
    end = min(len(profile), row + half + 1)
    window = profile[start:end]
    baseline = float(np.median(window)) if window.size else 0.0
    return max(float(profile[row] - baseline), 0.0)


def _template_response(profile: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if profile.size == 0:
        return profile
    return np.convolve(profile, kernel, mode="same")


def _line_score(profile: np.ndarray, config: RuleBasedConfig) -> np.ndarray:
    background = _local_background(profile, config)
    score = profile - background
    if config.line_kernel:
        kernel = np.asarray(config.line_kernel, dtype=np.float32)
        if kernel.size >= 3:
            score = _template_response(score, kernel)
            score = np.clip(score, 0.0, None)
    return score


def _window_bounds(start: int, end: int, window: tuple[float, float]) -> tuple[int, int]:
    band_len = end - start
    w_start = start + int(window[0] * band_len)
    w_end = start + int(window[1] * band_len)
    return w_start, w_end


def _best_peak(score: np.ndarray, start: int, end: int, min_ratio: float = 0.0) -> int | None:
    if end <= start or score.size == 0:
        return None
    segment = score[start:end]
    if segment.size == 0:
        return None
    max_val = float(np.max(segment))
    if min_ratio <= 0.0 or max_val <= 0.0:
        idx = int(np.argmax(segment))
        return start + idx
    threshold = max_val * min_ratio
    candidates = []
    for i in range(1, len(segment) - 1):
        if segment[i] >= threshold and segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            candidates.append(i)
    if not candidates:
        idx = int(np.argmax(segment))
        return start + idx
    return start + candidates[0]


def _peak_candidates(score: np.ndarray, start: int, end: int, min_ratio: float) -> list[int]:
    if end <= start or score.size == 0:
        return []
    segment = score[start:end]
    if segment.size == 0:
        return []
    max_val = float(np.max(segment))
    if max_val <= 0.0:
        idx = int(np.argmax(segment))
        return [start + idx]
    threshold = max_val * min_ratio
    candidates = []
    for i in range(1, len(segment) - 1):
        if segment[i] >= threshold and segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            candidates.append(start + i)
    if not candidates:
        idx = int(np.argmax(segment))
        return [start + idx]
    return candidates


def _best_peak_sequence(
    score: np.ndarray,
    c_range: tuple[int, int],
    t1_range: tuple[int, int],
    t2_range: tuple[int, int],
    min_ratio: float,
) -> tuple[int | None, int | None, int | None]:
    c_start, c_end = c_range
    t1_start, t1_end = t1_range
    t2_start, t2_end = t2_range
    if c_end <= c_start or t1_end <= t1_start or t2_end <= t2_start:
        return _best_peak(score, c_start, c_end), _best_peak(score, t1_start, t1_end), _best_peak(score, t2_start, t2_end)
    c_candidates = score[c_start:c_end]
    if c_candidates.size == 0:
        return None, None, None
    c_idx = _best_peak(score, c_start, c_end, min_ratio=min_ratio)
    if c_idx is None:
        return None, None, None
    t1_start = max(t1_start, c_idx + 1)
    t1_idx = _best_peak(score, t1_start, t1_end, min_ratio=min_ratio)
    if t1_idx is None:
        return c_idx, None, None
    t2_start = max(t2_start, t1_idx + 1)
    t2_idx = _best_peak(score, t2_start, t2_end, min_ratio=min_ratio)
    return c_idx, t1_idx, t2_idx


def _triplet_from_template(
    score: np.ndarray,
    start: int,
    end: int,
    config: RuleBasedConfig,
) -> tuple[int | None, int | None, int | None]:
    band_len = end - start
    if band_len <= 0:
        return None, None, None
    s1, s2 = config.template_spacing
    tol = config.template_tolerance
    c_start, c_end = _window_bounds(start, end, config.control_window)
    t1_start, t1_end = _window_bounds(start, end, config.test1_window)
    t2_start, t2_end = _window_bounds(start, end, config.test2_window)

    c_candidates = _peak_candidates(score, c_start, c_end, config.peak_min_ratio)
    t1_candidates = _peak_candidates(score, t1_start, t1_end, config.peak_min_ratio)
    t2_candidates = _peak_candidates(score, t2_start, t2_end, config.peak_min_ratio)

    best = None
    for c in c_candidates:
        for t1 in t1_candidates:
            if t1 <= c:
                continue
            d1 = (t1 - c) / max(band_len, 1)
            if abs(d1 - s1) > tol:
                continue
            for t2 in t2_candidates:
                if t2 <= t1:
                    continue
                d2 = (t2 - t1) / max(band_len, 1)
                if abs(d2 - s2) > tol:
                    continue
                return c, t1, t2
    # fallback: enforce ordering by chaining peaks
    return _best_peak_sequence(
        score, (c_start, c_end), (t1_start, t1_end), (t2_start, t2_end), config.peak_min_ratio
    )


def analyze_strip(image: np.ndarray, mask: np.ndarray, config: RuleBasedConfig) -> LineResult:
    height = image.shape[0]
    if height == 0:
        return LineResult(0.0, 0.0, 0.0, None, None, None, None, None, 0, 0, False)

    profile = _row_profile(image, mask, config)

    start = int(height * config.band_start_ratio)
    end = int(height * config.band_end_ratio)
    start = max(0, min(start, height - 1))
    end = max(start + 1, min(end, height))
    base_start = start

    membrane_start = None
    membrane_end = None
    if config.use_dynamic_band and config.use_membrane_anchor:
        region = _find_membrane_region(image, mask, config)
        if region is not None:
            membrane_start, membrane_end = region
            margin = int(height * config.band_margin_ratio)
            offset = int(height * config.membrane_offset_ratio)
            dynamic_start = min(height - 1, membrane_end + margin + offset)
            min_band_height = int(height * config.min_band_height_ratio)
            if end - dynamic_start >= min_band_height:
                start = dynamic_start
            else:
                start = min(base_start, max(0, end - min_band_height))

    if config.use_dynamic_band and (membrane_end is None):
        blue_fraction = _blue_row_fraction(image, config)
        handle_end = _find_handle_end(blue_fraction, config)
        if handle_end is not None:
            margin = int(height * config.band_margin_ratio)
            dynamic_start = min(height - 1, handle_end + margin)
            min_band_height = int(height * config.min_band_height_ratio)
            if end - dynamic_start >= min_band_height:
                start = max(start, dynamic_start)

    band_len = end - start
    if band_len > 0:
        if config.use_template_matching:
            score = _line_score(profile, config)
            control_row, test1_row, test2_row = _triplet_from_template(
                score, start, end, config
            )
            control_strength = score[control_row] if control_row is not None else 0.0
            test1_strength = score[test1_row] if test1_row is not None else 0.0
            test2_strength = score[test2_row] if test2_row is not None else 0.0
        else:
            control_start = start + int(config.control_band[0] * band_len)
            control_end = start + int(config.control_band[1] * band_len)
            remaining_start = control_end
            remaining_end = end

            control_peaks = _find_peaks_in_window(profile, control_start, control_end, config)
            if not control_peaks:
                control_peaks = _find_peaks_in_window(profile, start, end, config)
            if len(control_peaks) >= 1:
                control_row = max(control_peaks, key=lambda r: _strength_at(profile, r, config))
                control_strength = _strength_at(profile, control_row, config)

                test_peaks = _find_peaks_in_window(profile, remaining_start, remaining_end, config)
                test_peaks_sorted = sorted(test_peaks)
                if len(test_peaks_sorted) >= 2:
                    test1_row, test2_row = test_peaks_sorted[:2]
                elif len(test_peaks_sorted) == 1:
                    test1_row = test_peaks_sorted[0]
                    test2_row = None
                else:
                    test1_row = None
                    test2_row = None
                test1_strength = _strength_at(profile, test1_row, config) if test1_row is not None else 0.0
                test2_strength = _strength_at(profile, test2_row, config) if test2_row is not None else 0.0
            else:
                band_len = end - start
                seg_len = max(1, band_len // 3)
                control_strength, control_row, _ = _segment_strengths(profile, start, start + seg_len)
                test1_strength, test1_row, _ = _segment_strengths(profile, start + seg_len, start + 2 * seg_len)
                test2_strength, test2_row, _ = _segment_strengths(profile, start + 2 * seg_len, end)
    else:
        band_len = end - start
        seg_len = max(1, band_len // 3)
        control_strength, control_row, _ = _segment_strengths(profile, start, start + seg_len)
        test1_strength, test1_row, _ = _segment_strengths(profile, start + seg_len, start + 2 * seg_len)
        test2_strength, test2_row, _ = _segment_strengths(profile, start + 2 * seg_len, end)

    control_strength = max(control_strength, 0.0)
    test1_strength = max(test1_strength, 0.0)
    test2_strength = max(test2_strength, 0.0)
    valid = control_strength >= config.control_threshold
    return LineResult(
        control_strength=control_strength,
        test1_strength=test1_strength,
        test2_strength=test2_strength,
        control_row=control_row,
        test1_row=test1_row,
        test2_row=test2_row,
        membrane_start=membrane_start,
        membrane_end=membrane_end,
        band_start=start,
        band_end=end,
        valid=valid,
    )


def predict_from_strengths(result: LineResult, config: RuleBasedConfig) -> tuple[int, int]:
    if not result.valid:
        return 0, 0
    if result.control_strength <= 0:
        return 0, 0
    test1 = 1 if config.test_threshold <= result.test1_strength < result.control_strength else 0
    test2 = 1 if config.test_threshold <= result.test2_strength < result.control_strength else 0
    return test1, test2


def analyze_image(path: Path, config: RuleBasedConfig | None = None) -> tuple[LineResult, tuple[int, int]]:
    config = config or RuleBasedConfig()
    image = _load_image(path)
    mask = segment_strip_mask(image)
    if mask.mean() < config.min_foreground_fraction:
        mask = np.ones(image.shape[:2], dtype=np.uint8)
    rectified_img, rectified_mask = rectify_strip(image, mask, config)
    result = analyze_strip(rectified_img, rectified_mask, config)
    prediction = predict_from_strengths(result, config)
    return result, prediction
