"""Command-line interface: tileprep {stats, tile, pairs, stitch}."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tileprep import __version__
from tileprep.filters import FilterConfig
from tileprep.io import list_images, load_image, save_image
from tileprep.normalize import METHODS, NormStats, fit_stats
from tileprep.pipeline import TileSettings, make_pairs, run_tiling, stitch_scene
from tileprep.stitch import BLEND_MODES


def _nodata(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def _add_norm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHODS, default="percentile")
    parser.add_argument("--low", type=float, default=2.0, help="lower percentile (default 2)")
    parser.add_argument("--high", type=float, default=98.0, help="upper percentile (default 98)")
    parser.add_argument(
        "--nodata", type=_nodata, default=0.0, help="nodata pixel value, or 'none' (default 0)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tileprep", description="Prepare satellite scenes for super-resolution training."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="fit normalisation stats over one image or a folder")
    p.add_argument("input", help="image file or folder of images")
    p.add_argument("-o", "--output", required=True, help="stats JSON to write")
    _add_norm_args(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("tile", help="normalise and tile scenes, writing manifest.csv")
    p.add_argument("input", help="image file or folder of images")
    p.add_argument("-o", "--output", required=True, help="output folder")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    _add_norm_args(p)
    p.add_argument("--stats", help="reuse stats JSON (e.g. dataset-wide) instead of per-scene")
    p.add_argument("--no-filter", action="store_true", help="keep every tile")
    p.add_argument("--max-nodata", type=float, default=0.2, help="max nodata fraction")
    p.add_argument("--max-cloud", type=float, default=0.3, help="max bright/cloud fraction")
    p.add_argument("--cloud-threshold", type=float, default=0.9, help="'bright' in norm units")
    p.add_argument("--min-std", type=float, default=0.02, help="min std in norm units")
    p.add_argument("--save-skipped", action="store_true", help="also save skipped tiles")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.set_defaults(func=cmd_tile)

    p = sub.add_parser("pairs", help="make LR/HR pairs from a tile output folder")
    p.add_argument("tiles", help="output folder of 'tileprep tile'")
    p.add_argument("-o", "--output", required=True, help="pairs output folder")
    p.add_argument("--scale", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--blur-sigma", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std (norm units)")
    p.add_argument("--augment", type=int, default=0, help="extra augmented copies per tile")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("stitch", help="rebuild a scene from (possibly processed) tiles")
    p.add_argument("tiles", help="output folder of 'tileprep tile'")
    p.add_argument("--scene", required=True, help="scene name (input file name without suffix)")
    p.add_argument("-o", "--output", required=True, help=".tif/.png/.npy file to write")
    p.add_argument("--tiles-from", help="folder of processed tiles with the same file names")
    p.add_argument("--blend", choices=BLEND_MODES, default="feather")
    p.add_argument(
        "--denormalize", action="store_true", help="convert back to original units/dtype"
    )
    p.set_defaults(func=cmd_stitch)
    return parser


def cmd_stats(args: argparse.Namespace) -> None:
    paths = list_images(args.input)
    images = (load_image(p) for p in paths)  # generator: one scene in memory at a time
    stats = fit_stats(images, args.method, args.low, args.high, args.nodata)
    stats.save(args.output)
    table = pd.DataFrame({"offset": stats.offset, "scale": stats.scale})
    table.index.name = "band"
    print(f"{stats.method} stats from {len(paths)} image(s) -> {args.output}")
    print(table.round(3).to_string())


def cmd_tile(args: argparse.Namespace) -> None:
    filters = None
    if not args.no_filter:
        filters = FilterConfig(
            nodata_value=args.nodata,
            max_nodata_frac=args.max_nodata,
            cloud_threshold=args.cloud_threshold,
            max_cloud_frac=args.max_cloud,
            min_std=args.min_std,
        )
    settings = TileSettings(
        tile_size=args.tile_size,
        overlap=args.overlap,
        method=args.method,
        low=args.low,
        high=args.high,
        nodata_value=args.nodata,
        filters=filters,
        save_skipped=args.save_skipped,
    )
    stats = NormStats.load(args.stats) if args.stats else None
    paths = list_images(args.input)
    manifest = run_tiling(paths, args.output, settings, stats=stats, workers=args.workers)

    print(f"\n{manifest['kept'].sum()} of {len(manifest)} tiles kept")
    print(manifest["reason"].value_counts().to_string())
    print(f"manifest: {Path(args.output) / 'manifest.csv'}")


def cmd_pairs(args: argparse.Namespace) -> None:
    make_pairs(
        args.tiles, args.output, args.scale, args.blur_sigma, args.noise, args.augment, args.seed
    )


def cmd_stitch(args: argparse.Namespace) -> None:
    image = stitch_scene(args.tiles, args.scene, args.tiles_from, args.blend, args.denormalize)
    output = Path(args.output)
    if output.suffix.lower() == ".npy":
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, image)
    elif image.dtype == np.float32 and output.suffix.lower() not in {".tif", ".tiff"}:
        raise SystemExit("float32 output needs .tif or .npy (or pass --denormalize)")
    else:
        save_image(output, image)
    print(f"stitched {args.scene}: {image.shape} {image.dtype} -> {output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `tileprep` console script."""
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
