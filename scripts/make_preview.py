"""Render one LR/HR pair as a PNG montage: HR | LR (pixelated) | bicubic upsample.

Reads the output of `tileprep pairs`. Uses the first three bands, which are
B, G, R in OpenCV order, so cv2.imwrite shows natural colour.

Usage:
    python scripts/make_preview.py data/pairs --out examples/preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tileprep.degrade import bicubic_upsample

PANEL = 256  # display size of each panel in pixels


def to_uint8(tile: np.ndarray) -> np.ndarray:
    rgb = np.clip(tile[..., :3], 0, 1) if tile.shape[2] >= 3 else np.repeat(tile, 3, axis=2)
    return (rgb * 255).round().astype(np.uint8)


def sharpness(tile: np.ndarray) -> float:
    """Edge content of a tile; tiles touching nodata or cloud score 0."""
    if np.all(tile >= 0.95, axis=2).any() or np.all(tile == 0, axis=2).any():
        return 0.0
    gray = tile[..., :3].mean(axis=2).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def panel(image: np.ndarray, label: str) -> np.ndarray:
    resized = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    header = np.full((28, PANEL, 3), 255, np.uint8)
    cv2.putText(header, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([header, resized])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pairs", help="output folder of `tileprep pairs`")
    parser.add_argument("--out", default="examples/preview.png")
    parser.add_argument("--pair-id", help="pair to show (default: sharpest HR tile)")
    args = parser.parse_args()

    root = Path(args.pairs)
    pairs = pd.read_csv(root / "pairs.csv")
    pairs = pairs[pairs["dihedral_k"] == 0]  # originals only, not augmented copies
    if args.pair_id:
        row = pairs[pairs["pair_id"] == args.pair_id].iloc[0]
    else:
        # Variance of the Laplacian rewards sharp edges (roads, buildings) and
        # ignores smooth bright areas such as cloud.
        detail = [sharpness(np.load(root / p)) for p in pairs["hr_path"]]
        row = pairs.iloc[int(np.argmax(detail))]

    hr = np.load(root / row["hr_path"])
    lr = np.load(root / row["lr_path"])
    scale = int(row["scale"])
    bicubic = np.clip(bicubic_upsample(lr, scale), 0, 1)

    gap = np.full((PANEL + 28, 8, 3), 255, np.uint8)
    montage = np.hstack([
        panel(to_uint8(hr), f"HR {hr.shape[0]}px"),
        gap,
        panel(to_uint8(lr), f"LR {lr.shape[0]}px (x{scale} down)"),
        gap,
        panel(to_uint8(bicubic), "Bicubic upsample of LR"),
    ])  # fmt: skip
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from pair {row['pair_id']}")


if __name__ == "__main__":
    main()
