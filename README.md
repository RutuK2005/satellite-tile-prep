# satellite-tile-prep

[![CI](https://github.com/RutuK2005/satellite-tile-prep/actions/workflows/ci.yml/badge.svg)](https://github.com/RutuK2005/satellite-tile-prep/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tools for turning large satellite images into training data for super-resolution models.

At an ISRO hackathon our model needed high-resolution satellite imagery cut into tiles, normalised and paired with low-resolution versions. That work was a pile of one-off scripts. This repo is a cleaner rewrite of the same steps as a small CLI.

![HR tile, 4x downscaled LR tile, bicubic upsample](examples/preview.png)

*Left to right: the high-res target tile, the low-res input made from it, and a plain bicubic upscale of that input (the baseline a super-resolution model has to beat).*

## What it does

1. **stats**: works out per-band normalisation (2nd–98th percentile clipping by default, or min-max / z-score) and saves it to JSON, so inference uses exactly the same transform.
2. **tile**: normalises each scene and cuts it into overlapping tiles. It pads the edges by mirroring, and skips tiles that are mostly empty, cloud or featureless. Every tile, kept or skipped, goes into `manifest.csv` with the reason.
3. **pairs**: makes low-res/high-res training pairs by blurring and downscaling (2x or 4x), with optional noise and flips/rotations applied identically to both.
4. **stitch**: puts tiles (or a model's output tiles) back together into a full image, blending the overlaps so seams don't show.

It reads 8- or 16-bit images with up to 4 bands through OpenCV, and processes multiple scenes in parallel.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# no satellite data handy? generate a few synthetic scenes
python scripts/make_synthetic_scene.py --out data/scenes --count 3 --size 1536

tileprep stats data/scenes -o data/stats.json
tileprep tile data/scenes -o data/tiles --tile-size 256 --overlap 32 --stats data/stats.json --workers 3
tileprep pairs data/tiles -o data/pairs --scale 4 --noise 0.01 --augment 1
tileprep stitch data/tiles --scene synthetic_01 -o data/stitched/synthetic_01.tif --denormalize
```

On the synthetic scenes the tile step keeps 127 of 147 tiles (18 skipped as empty, 2 as cloud).

To stitch a model's outputs, save them with the same file names in a folder and add `--tiles-from <folder>`. If they're 2x or 4x bigger than the input tiles, that's detected automatically.

With min-max scaling and no skip rules, tiling then stitching gives back the original image exactly (max pixel difference 0). Percentile clipping is lossy by design.

### Real data

Free Sentinel-2 scenes can be downloaded from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/). A true-colour 8-bit export works directly. For reflectance data, download bands B02, B03, B04 and B08 as 16-bit TIFFs and stack them:

```python
import numpy as np
from tileprep.io import load_image, save_image

bands = [load_image(f"download/{b}.tiff")[..., 0] for b in ["B02", "B03", "B04", "B08"]]
save_image("data/scenes/sentinel2.tif", np.dstack(bands))
```

For files with more than 4 bands, install the optional rasterio reader with `pip install -e ".[geo]"`.

## Notes

- OpenCV instead of GDAL keeps installation simple, but it doesn't keep georeferencing (coordinates/CRS).
- Percentiles are estimated from a random sample of up to 500k pixels per scene, which is much lighter on memory and gives practically the same result.
- Clouds push the 98th percentile up, so land can look dark after normalisation. Real cloud masks would fix that (see TODO).
- Parallelism is per scene, so one huge scene still runs on a single core.
- Synthetic degradation (blur + downscale) is the usual approach when there are no real matched pairs, but real sensor blur looks different, so it's worth varying the blur and noise settings.

## Tests

```bash
pytest
ruff check .
```

## TODO

- Keep georeferencing when rasterio is available
- Use the Sentinel-2 cloud mask (SCL band) instead of the brightness rule
- PyTorch `Dataset` that reads `pairs.csv` directly
