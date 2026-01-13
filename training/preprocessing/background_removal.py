"""Background removal using transparent-background (InSPyReNet)."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from training.preprocessing.labels import LabelConfig, infer_delimiter, read_labels_csv


class BackgroundRemovalError(RuntimeError):
    pass


def _load_remover(mode: str = "base", device: str | None = "cpu"):
    try:
        from transparent_background import Remover
    except ImportError as exc:  # pragma: no cover - handled by caller
        raise BackgroundRemovalError(
            "transparent-background is not installed. Install with `pip install "
            "transparent-background` or via the project extras."
        ) from exc

    if device is not None:
        try:
            return Remover(mode=mode, device=device)
        except TypeError:
            pass
    try:
        return Remover(mode=mode)
    except TypeError:
        return Remover()


def _ensure_rgba(output: object) -> Image.Image:
    if isinstance(output, Image.Image):
        return output.convert("RGBA")
    if isinstance(output, np.ndarray):
        return Image.fromarray(output).convert("RGBA")
    raise BackgroundRemovalError("Unexpected output type from transparent-background")


def _composite_white(rgba: Image.Image) -> Image.Image:
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.split()[3])
    return white


def _save_output(output: object, output_path: Path, output_type: str = "white") -> Path:
    rgba = _ensure_rgba(output)
    output_type = output_type.lower()
    if output_type == "rgba":
        final_path = output_path.with_suffix(".png")
        rgba.save(final_path)
        return final_path
    if output_type == "white":
        rgb = _composite_white(rgba)
        rgb.save(output_path)
        return output_path
    raise BackgroundRemovalError(f"Unsupported output_type: {output_type}")


def _save_mask(rgba: Image.Image, mask_path: Path) -> None:
    mask = np.array(rgba)[:, :, 3]
    Image.fromarray(mask).save(mask_path)


def remove_background(image: Image.Image, remover) -> object:
    return remover.process(image)


def process_split(
    split_csv: Path,
    images_root: Path,
    output_dir: Path,
    device: str = "cpu",
    mode: str = "base",
    output_type: str = "white",
    skip_existing: bool = True,
    save_mask: bool = False,
    fallback_to_original: bool = True,
    progress_every: int = 50,
    delimiter: str | None = None,
    config: LabelConfig | None = None,
) -> dict[str, list[str]]:
    config = config or LabelConfig()
    if delimiter is None:
        delimiter = infer_delimiter(split_csv)
    df = read_labels_csv(split_csv, delimiter=delimiter)

    remover = _load_remover(mode=mode, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks" if save_mask else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    processed: list[str] = []
    fallback: list[str] = []
    rel_paths = df[config.image_path_col].tolist()
    total = len(rel_paths)
    if progress_every > 0:
        print(f"Processing {total} images in {output_dir}")
    for idx, rel_path in enumerate(rel_paths, start=1):
        src_path = images_root / rel_path
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out_path.exists():
            continue
        try:
            with Image.open(src_path) as img:
                image = img.convert("RGB")
            result = remove_background(image, remover)
            _save_output(result, out_path, output_type=output_type)
            if save_mask:
                rgba = _ensure_rgba(result)
                mask_path = (mask_dir / rel_path).with_suffix(".png")
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                _save_mask(rgba, mask_path)
            processed.append(rel_path)
        except Exception as exc:
            failures.append(f"{rel_path}: {exc}")
            if fallback_to_original:
                try:
                    with Image.open(src_path) as img:
                        image = img.convert("RGB")
                    image.save(out_path)
                    fallback.append(rel_path)
                except Exception as fallback_exc:
                    failures.append(f"{rel_path}: fallback_failed: {fallback_exc}")
        if progress_every > 0 and idx % progress_every == 0:
            print(
                f"Processed {idx}/{total} | ok {len(processed)} | "
                f"fallback {len(fallback)} | failed {len(failures)}"
            )

    return {"processed": processed, "failed": failures, "fallback": fallback}


def sample_qc_images(
    output_dir: Path,
    qc_dir: Path,
    sample_count: int = 20,
    seed: int = 42,
) -> list[Path]:
    output_paths = [p for p in output_dir.rglob("*") if p.is_file()]
    if not output_paths:
        return []
    rng = random.Random(seed)
    sample_paths = rng.sample(output_paths, k=min(sample_count, len(output_paths)))
    qc_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in sample_paths:
        dest = qc_dir / path.name
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def run_background_removal(
    splits: Iterable[str],
    splits_dir: Path,
    images_root: Path,
    output_root: Path,
    device: str = "cpu",
    mode: str = "base",
    output_type: str = "white",
    skip_existing: bool = True,
    save_mask: bool = False,
    fallback_to_original: bool = True,
    progress_every: int = 50,
    qc_samples: int = 20,
    seed: int = 42,
) -> None:
    for split in splits:
        split_csv = splits_dir / f"{split}.csv"
        output_dir = output_root / split
        split_images_root = images_root / split if (images_root / split).exists() else images_root
        if progress_every > 0:
            print(f"Starting split '{split}' -> {output_dir}")
        result = process_split(
            split_csv=split_csv,
            images_root=split_images_root,
            output_dir=output_dir,
            device=device,
            mode=mode,
            output_type=output_type,
            skip_existing=skip_existing,
            save_mask=save_mask,
            fallback_to_original=fallback_to_original,
            progress_every=progress_every,
        )
        report_path = output_dir / "background_removal_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        if qc_samples > 0:
            qc_dir = output_root / "qc" / split
            sample_qc_images(output_dir, qc_dir, sample_count=qc_samples, seed=seed)
