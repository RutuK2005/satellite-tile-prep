"""Generate synthetic 4-band (B, G, R, NIR) uint16 scenes for demos and smoke tests.

The scenes are not physically accurate, but they have the structures that
matter for tiling and super-resolution: field boundaries, thin roads, a
winding river, forest texture, sensor noise, a few clouds and a nodata wedge
(like the edge of a satellite swath). Values mimic Sentinel-2 L2A surface
reflectance scaled by 10,000.

Usage:
    python scripts/make_synthetic_scene.py --out data/scenes --count 3 --size 1536
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Rough surface reflectance per class, in OpenCV band order: B, G, R, NIR.
CLASSES = {
    "crop": (0.04, 0.09, 0.06, 0.42),
    "soil": (0.14, 0.18, 0.24, 0.30),
    "dry_grass": (0.10, 0.14, 0.18, 0.32),
    "forest": (0.03, 0.06, 0.04, 0.32),
    "water": (0.07, 0.07, 0.05, 0.02),
    "road": (0.18, 0.19, 0.20, 0.22),
    "urban": (0.22, 0.22, 0.24, 0.26),
}
CLOUD = np.array([0.75, 0.76, 0.77, 0.78], dtype=np.float32)


def smooth_noise(height: int, width: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [0, 1]: random grid upsampled with bicubic interpolation."""
    small = rng.random((height // cell + 2, width // cell + 2)).astype(np.float32)
    big = cv2.resize(small, (width + 2 * cell, height + 2 * cell), interpolation=cv2.INTER_CUBIC)
    big = big[cell : cell + height, cell : cell + width]
    return (big - big.min()) / (big.max() - big.min() + 1e-6)


def paint(scene: np.ndarray, mask: np.ndarray, name: str, jitter: float = 0.0) -> None:
    scene[mask > 0] = np.array(CLASSES[name], dtype=np.float32) * (1 + jitter)


def make_fields(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Irregular grid of agricultural fields, slightly rotated."""
    pad = max(height, width) // 2
    big_h, big_w = height + 2 * pad, width + 2 * pad
    scene = np.zeros((big_h, big_w, 4), np.float32)
    ys = np.cumsum(rng.integers(60, 220, size=big_h // 60))
    xs = np.cumsum(rng.integers(60, 220, size=big_w // 60))
    for y0, y1 in zip(np.r_[0, ys], ys, strict=False):
        for x0, x1 in zip(np.r_[0, xs], xs, strict=False):
            name = rng.choice(["crop", "crop", "soil", "dry_grass"])
            scene[y0:y1, x0:x1] = np.array(CLASSES[name]) * rng.uniform(0.8, 1.2)
            # crop rows: faint stripes inside the field
            if name == "crop":
                stripes = 1 + 0.06 * np.sin(np.arange(y0, min(y1, big_h)) * rng.uniform(0.5, 1.5))
                scene[y0:y1, x0:x1] *= stripes[:, None, None]
    angle = rng.uniform(-25, 25)
    rot = cv2.getRotationMatrix2D((big_w / 2, big_h / 2), angle, 1.0)
    scene = cv2.warpAffine(scene, rot, (big_w, big_h), flags=cv2.INTER_NEAREST)
    return scene[pad : pad + height, pad : pad + width].copy()


def make_scene(height: int, width: int, seed: int) -> np.ndarray:
    """Build one synthetic scene as uint16 reflectance x 10,000, shape (H, W, 4)."""
    rng = np.random.default_rng(seed)
    scene = make_fields(height, width, rng)

    # Forest patches where low-frequency noise is high, with leafy texture.
    forest = smooth_noise(height, width, 160, rng) > 0.72
    texture = 1 + 0.15 * (smooth_noise(height, width, 4, rng) - 0.5)
    scene[forest] = np.array(CLASSES["forest"]) * texture[forest, None]

    # Small town: cluster of buildings.
    cy, cx = rng.integers(height // 4, 3 * height // 4), rng.integers(width // 4, 3 * width // 4)
    town = np.zeros((height, width), np.uint8)
    for _ in range(120):
        y, x = rng.normal(cy, 80), rng.normal(cx, 80)
        w, h = rng.integers(6, 20, size=2)
        cv2.rectangle(town, (int(x), int(y)), (int(x + w), int(y + h)), 1, -1)
    paint(scene, town, "urban")

    # Roads: straight lines across the scene, one passing through the town.
    roads = np.zeros((height, width), np.uint8)
    for _ in range(3):
        p0 = (int(rng.integers(0, width)), 0)
        p1 = (int(rng.integers(0, width)), height - 1)
        cv2.line(roads, p0, p1, 1, int(rng.integers(3, 6)))
    cv2.line(roads, (0, int(cy)), (width - 1, int(cy + rng.integers(-200, 200))), 1, 4)
    paint(scene, roads, "road")

    # River: a sine-shaped polyline with a random phase.
    ys = np.arange(0, height, 4)
    xs = width * 0.3 + 120 * np.sin(ys / 180 + rng.uniform(0, 6)) + 40 * np.sin(ys / 47)
    river = np.zeros((height, width), np.uint8)
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(river, [points], False, 1, int(rng.integers(14, 24)))
    paint(scene, river, "water")

    # Soften class edges (mixed pixels) and add sensor noise.
    scene = cv2.GaussianBlur(scene, (0, 0), 0.8)
    scene += rng.normal(0, 0.004, scene.shape).astype(np.float32)

    # Clouds: blobs of smooth noise blended towards bright white.
    cloud_alpha = np.clip((smooth_noise(height, width, 220, rng) - 0.9) * 20, 0, 1)
    cloud_alpha = cv2.GaussianBlur(cloud_alpha, (0, 0), 3)[..., None]
    scene = scene * (1 - cloud_alpha) + CLOUD * cloud_alpha

    out = np.clip(scene * 10_000, 1, 65_535).astype(np.uint16)
    # Nodata wedge in a random corner, like the edge of a satellite swath.
    depth = rng.uniform(0.25, 0.45)
    wedge = np.array(
        [[0, height - 1], [0, int(height * (1 - depth))], [int(width * depth), height - 1]],
        np.int32,
    )
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [wedge], 1)
    mask = np.rot90(mask, int(rng.integers(0, 4)))  # scenes are square, so shape is unchanged
    out[mask > 0] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/scenes", help="output folder")
    parser.add_argument("--count", type=int, default=3, help="number of scenes")
    parser.add_argument("--size", type=int, default=1536, help="scene height and width")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    for i in range(args.count):
        scene = make_scene(args.size, args.size, seed=args.seed + i)
        path = out / f"synthetic_{i + 1:02d}.tif"
        cv2.imwrite(str(path), scene)
        print(f"wrote {path} {scene.shape} {scene.dtype}")


if __name__ == "__main__":
    main()
