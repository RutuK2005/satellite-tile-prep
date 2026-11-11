import json

import numpy as np
import pandas as pd

from tileprep.cli import main
from tileprep.io import load_image, save_image


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(0)
    scene = rng.integers(100, 4000, size=(80, 96, 4), dtype=np.uint16)
    save_image(tmp_path / "scenes" / "demo.tif", scene)
    scenes, out = str(tmp_path / "scenes"), str(tmp_path / "out")

    assert main(["stats", scenes, "-o", str(tmp_path / "stats.json")]) == 0
    assert json.loads((tmp_path / "stats.json").read_text())["method"] == "percentile"

    args = ["tile", scenes, "-o", out, "--tile-size", "32", "--overlap", "8", "--workers", "1"]
    assert main([*args, "--stats", str(tmp_path / "stats.json")]) == 0
    manifest = pd.read_csv(tmp_path / "out" / "manifest.csv")
    assert len(manifest) == 3 * 4  # 80x96 with stride 24 -> 3 rows x 4 cols

    assert main(["pairs", out, "-o", str(tmp_path / "pairs"), "--scale", "2"]) == 0
    assert (tmp_path / "pairs" / "pairs.csv").exists()

    stitched = tmp_path / "stitched.tif"
    assert main(["stitch", out, "--scene", "demo", "-o", str(stitched), "--denormalize"]) == 0
    restored = load_image(stitched)
    assert restored.shape == scene.shape
    assert restored.dtype == np.uint16
    assert "stitched demo" in capsys.readouterr().out


def test_cli_no_filter_keeps_all_tiles(tmp_path):
    save_image(tmp_path / "blank.tif", np.zeros((64, 64, 1), np.uint16))
    out = tmp_path / "out"
    main(["tile", str(tmp_path / "blank.tif"), "-o", str(out), "--tile-size", "32", "--no-filter",
          "--nodata", "none", "--overlap", "0"])  # fmt: skip
    manifest = pd.read_csv(out / "manifest.csv")
    assert manifest["kept"].all()
    assert len(manifest) == 4
