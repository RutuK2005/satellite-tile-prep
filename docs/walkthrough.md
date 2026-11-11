# Walkthrough: how satellite-tile-prep works

A study guide to the codebase. Read it next to the source. Each section says what a
module does, the main idea behind it, and what to look at in the code.

Suggested reading order: `tiling.py` → `filters.py` → `normalize.py` → `degrade.py` →
`stitch.py` → `pipeline.py` → `cli.py`. Then run the quickstart in the README and open
`data/tiles/manifest.csv` in Pandas.

---

## 1. The problem

A satellite scene is large (a Sentinel-2 tile is about 10,980 x 10,980 pixels per band)
and its values are 16-bit reflectance, not 0-255. A super-resolution network wants:

- small, fixed-size inputs (e.g. 256 x 256) that fit in GPU memory,
- float values on a consistent scale,
- pairs of (low-res input, high-res target) that line up pixel for pixel,
- no useless examples (black nodata borders, clouds, empty water).

After the model has run, the processed tiles must be put back together into one
scene without visible seams. This toolkit covers all of those steps.

---

## 2. Module by module

### `io.py`: loading images

- `cv2.imread(path, cv2.IMREAD_UNCHANGED)` is the key call. The default flag
  (`IMREAD_COLOR`) would convert a 16-bit, 4-band image into 8-bit, 3-band BGR, which
  throws away most of the radiometric precision.
- Everything is returned as `(H, W, C)`, even single-band images (`image[:, :, None]`),
  so later code never needs `if image.ndim == 2` branches.
- If OpenCV returns `None` (e.g. a GeoTIFF with 13 bands), the code tries rasterio.
  It's imported inside the function, so the package works without it.
- OpenCV uses **BGR** channel order. The synthetic scenes store bands as B, G, R, NIR,
  so `cv2.imwrite` of the first three bands shows natural colour.

### `tiling.py`: the tile grid

- `make_grid(height, width, tile_size, overlap)` works out how many rows and columns of
  tiles are needed: `n = ceil((length - tile) / stride) + 1`, where
  `stride = tile_size - overlap`.
- `padded_shape` is the size the image must be for the last tile to fit:
  `(n - 1) * stride + tile_size`.
- `pad_to_grid` pads only the **bottom and right**, so a tile's `(y, x)` offset in the
  padded image is also its offset in the original image. That keeps stitching simple.
- `np.pad(mode="reflect")` mirrors the image at the border instead of adding zeros.
- `iter_tiles` is a generator that yields **views** (slices), not copies, so tiling a
  big image doesn't double its memory use.

### `filters.py`: skip rules

`evaluate_tile(raw, norm, config)` returns a `TileDecision` with `keep`, a `reason` and
the stats behind the decision. Checks in order:

1. **nodata**: fraction of pixels equal to the nodata value in *every* band (checked on
   the raw tile, because normalisation changes the values).
2. **cloud**: fraction of pixels whose normalised value is at least `cloud_threshold` in
   *every* band. Bright in all bands means white, which is typical of clouds.
3. **low_variance**: mean of the per-band standard deviations is below `min_std`. Flat
   tiles teach an SR model nothing.

The stats are recorded even for kept tiles, so the manifest can be used later to
re-filter without re-tiling (e.g. `df[df.cloud_frac < 0.05]`).

### `normalize.py`: per-band normalisation

Every method is the same formula with different parameters:
`norm = (raw - offset) / scale`.

| method     | offset     | scale             | clip to [0, 1] |
|------------|------------|-------------------|----------------|
| percentile | p_low (2%) | p_high - p_low    | yes            |
| minmax     | min        | max - min         | yes            |
| zscore     | mean       | std               | no             |

- `fit_stats` samples up to 500k valid pixels per image (nodata excluded) and computes
  the parameters **per band**. Bands have very different ranges (NIR is much brighter
  than blue over vegetation), so one global scale would squash some bands.
- `NormStats.save/load` writes the parameters to JSON. Training and inference must use
  the **same** numbers, otherwise the model sees inputs from a different distribution.
- `invert` maps model outputs back to reflectance and the original dtype (rounding and
  clipping to the uint16 range).
- A constant band would give `scale = 0`. The code replaces that with 1 to avoid
  dividing by zero.

### `degrade.py`: making LR from HR

`degrade(hr, scale, blur_sigma, noise_std)`:

1. **Gaussian blur.** This approximates the sensor's point spread function and also
   acts as an anti-aliasing filter before downsampling.
2. **Bicubic downscale** by `scale` with `cv2.resize(..., INTER_CUBIC)`.
3. **Optional Gaussian noise** with a seeded RNG, so results are reproducible.

`dihedral(image, k)` implements the 8 symmetries of a square (4 rotations x optional
flip). `augment_pair` picks one `k` and applies it to **both** LR and HR. A test checks
that `degrade(augment(hr)) == augment(degrade(hr))`, i.e. the pair stays aligned.

### `stitch.py`: putting tiles back together

```
output = sum(weight_i * tile_i) / sum(weight_i)
```

- `blend_weights` builds a 2-D window: 1.0 in the centre, falling linearly to near 0
  across the overlap band at each edge (`np.outer(ramp, ramp)`).
- Two accumulators, `accum` (weighted sum) and `weight_sum`, are filled tile by tile,
  then divided. `np.divide(..., where=weight_sum > 0)` leaves uncovered pixels (from
  skipped tiles) at 0.
- If every tile is unmodified, every term in the weighted average is the same value,
  so the result is exact. The round-trip tests rely on this.

### `pipeline.py`: workflows and multiprocessing

- `process_scene` does the whole job for **one** scene: load → fit or reuse stats →
  normalise → pad → tile → filter → save `.npy` → return manifest rows (dicts).
- `run_tiling` runs `process_scene` for many scenes. With `workers > 1` it uses
  `ProcessPoolExecutor` + `as_completed`, so progress is printed as soon as any scene
  finishes. Failures are caught per scene and logged, so one bad file doesn't stop the
  batch. The rows are collected into a Pandas DataFrame and written to `manifest.csv`.
- `TileSettings.__post_init__` validates settings in the parent process, so a bad
  `--overlap` fails immediately instead of failing once per worker.
- `make_pairs` reads the manifest, degrades every kept tile, and writes `hr/`, `lr/` and
  `pairs.csv`. Each tile uses `np.random.default_rng([seed, i])`, so the noise doesn't
  depend on processing order.
- `stitch_scene` reads the offsets from the manifest. If the tiles in `--tiles-from` are
  k times bigger (a x4 SR model's output), it multiplies offsets, size and overlap by k.

### `cli.py`

argparse with subcommands (`stats`, `tile`, `pairs`, `stitch`). Each subcommand sets
`func=cmd_*`, so `main()` just calls `args.func(args)`. The console script is declared
in `pyproject.toml` under `[project.scripts]`.

---

## 3. Key concepts

### Tiling, stride and overlap

With `tile_size=256, overlap=32` the window moves 224 px each step. Overlap is useful
because:

- a model's output is less reliable near tile borders (it can't see past the edge),
  so with overlap every border region also appears nearer the centre of a neighbour,
- objects cut by one tile boundary appear whole in the neighbouring tile.

The cost is more tiles: about `(256/224)^2 ≈ 1.3x` as many as with no overlap.

### Percentile normalisation

16-bit reflectance has a few extreme values (specular glints, saturated clouds, hot
pixels). Min-max scaling lets one outlier squash all real data into a narrow range.
Clipping at the 2nd and 98th percentiles ignores the extreme 2% at each end and spreads
the rest over [0, 1]. The trade-off: values beyond the clip points are lost, so
`invert` can't restore them exactly.

You can see this in the synthetic demo: clouds make up a small but noticeable share of
the bright pixels, so they pull the 98th percentile up and the land looks darker after
normalisation. On real data you would compute stats on cloud-free scenes, or use a
cloud mask when sampling.

### Why degrade HR to make LR?

Supervised SR needs pairs of the *same place* at two resolutions, perfectly
co-registered and taken at the same time. Real pairs from two sensors differ in
acquisition time, viewing angle, atmosphere and spectral response. Downsampling the HR
image yourself gives exact alignment and unlimited pairs. The catch is the *domain
gap*: the model learns to undo *your* degradation, so the degradation should be close
to the real one (blur + downsample + noise is the standard baseline, randomising it
makes the model more robust).

Why blur before downsampling? Plain bicubic subsampling of a sharp image aliases:
high-frequency detail (thin roads) turns into jagged patterns that a real
low-resolution sensor would never produce.

### Blending seams

Stitching with no overlap, or with overlap but a hard cut, shows visible lines where
neighbouring model outputs disagree slightly. Plain averaging (`--blend mean`) halves
the step but still has an abrupt change where the overlap starts. Feathering ramps
each tile's weight from 1 in the centre down to near 0 at its edge, so the transition
is gradual and border pixels (the least reliable ones) count least.

### Multiprocessing

- **Why processes, not threads?** CPython's GIL lets only one thread run Python
  bytecode at a time. The per-tile loop (slicing, filter stats, `np.save`, building
  dicts) is Python-level work, so threads wouldn't speed it up much. Separate
  processes each have their own interpreter.
- **Why per scene?** Scenes are independent, so there's no shared state. Each worker
  writes its own files and returns plain dicts. Only small objects (paths, settings,
  stats) are pickled to the workers, not big arrays.
- **Windows note.** Windows starts workers with "spawn" (a fresh interpreter that
  imports the module), so the worker function must live at module level
  (`process_scene`), and scripts that start pools need `if __name__ == "__main__":`.

---

## 4. Likely interview questions

**1. Why do you need overlap between tiles? How did you choose it?**
Models are less accurate near borders because they lack context, and objects get cut
at tile edges. Overlap lets the stitcher favour each tile's centre. Make it at least
the model's useful receptive field near the edge. 32 px on 256 px tiles (12.5%) is a
common middle ground between seam quality and the extra compute.

**2. What happens at the image edges when the size isn't a multiple of the stride?**
The image is reflect-padded on the bottom/right so the last tiles are complete. The
manifest stores the original height/width, and stitching crops the padding off.
Reflection avoids black borders that the model would treat as real content.

**3. Why percentile clipping instead of min-max?**
Min-max is controlled by the single most extreme pixel. One saturated cloud or hot
pixel compresses all real values into a narrow range. Percentiles ignore the tails.
The cost is that clipped values can't be recovered.

**4. Why save normalisation stats to JSON?**
The model learns a mapping in normalised space. At inference the input must be
normalised with the same offsets and scales as training, and the output must be
de-normalised with them too. Saving them makes the transform reproducible and lets
you compare datasets.

**5. Why create LR images by degrading HR instead of using a real low-res sensor?**
It gives perfect pixel alignment and unlimited pairs. The risk is a domain gap if the
synthetic degradation differs from the real sensor. You reduce it by using a realistic
PSF blur, adding noise, randomising the degradation, and validating on real LR imagery.

**6. Why blur before bicubic downsampling?**
Downsampling without low-pass filtering causes aliasing: fine detail turns into false
patterns. The Gaussian blur works as both an anti-aliasing filter and a stand-in for
the sensor's optical blur.

**7. How do you check that tiling and stitching are correct?**
A round-trip test: tile an image with overlap, stitch the untouched tiles, and compare
with the original. With min-max normalisation and no filtering the difference is 0
(tested in `tests/test_pipeline.py`, and checked on a 1536 x 1536 synthetic scene).
There is also a test for upscaled tiles (as if from a x2 model) and for holes left by
skipped tiles.

**8. Why ProcessPoolExecutor and not threads or asyncio?**
The work is CPU-bound, and the per-tile loop is Python code that holds the GIL, so
threads would mostly run one at a time. asyncio is for waiting on I/O, not CPU work.
Processes give real parallelism. Scenes are independent, so the only communication is
sending a path in and getting manifest rows back.

**9. How do augmentations work for LR/HR pairs, and what can go wrong?**
Draw one random dihedral transform (flip/rot90) and apply it to both images.
Transforming them independently misaligns the pair and the model learns nonsense.
Flips and 90-degree rotations are safe because they move whole pixels and commute with
downsampling. Arbitrary-angle rotations would re-interpolate (and blur) the HR target.
Colour jitter would also have to be applied identically to both images.

**10. How would you scale this to thousands of full-size Sentinel-2 scenes?**
Read scenes in windows (rasterio) instead of loading them whole, and split large
scenes across workers. Write tiles to a chunked format (Zarr/HDF5) or a sharded
dataset instead of many small files. Compute dataset stats once from a sample of
scenes. Use the manifest to resume interrupted runs (skip scenes already listed). On a
cluster, run the same per-scene function under a job scheduler.

---

## 5. Things to try yourself

- Change `--overlap` to 0 and stitch real model outputs (or tiles with a little noise
  added) with `--blend mean` and `--blend feather`, then compare the seams.
- Run `tileprep tile` with `--method zscore` and see why the cloud threshold needs to
  change.
- Load `manifest.csv` and plot `std` against `cloud_frac` to choose better thresholds.
- Time `--workers 1` against `--workers 4` on more, larger scenes.
