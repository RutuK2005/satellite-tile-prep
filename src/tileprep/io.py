"""Reading and writing raster images.

OpenCV is the default reader because it is a light dependency and handles the
common cases (8/16-bit or float32, 1-4 bands). rasterio is only used as an
optional fallback for files OpenCV cannot open, such as GeoTIFFs with >4 bands.

All arrays inside the toolkit use (height, width, bands) layout, even for
single-band images, so downstream code never has to special-case 2-D arrays.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jp2", ".jpg", ".jpeg"}

# libtiff prints harmless warnings for 4-band TIFFs written by OpenCV.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an (H, W, C) array, preserving its dtype and bit depth."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # IMREAD_UNCHANGED keeps 16-bit data and the alpha/4th band; the default
    # flag would silently convert everything to 8-bit, 3-channel BGR.
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = _load_with_rasterio(path)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    return image


def _load_with_rasterio(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise ValueError(
            f"OpenCV could not read {path}. For multi-band GeoTIFFs install the "
            "optional dependency: pip install 'satellite-tile-prep[geo]'"
        ) from exc
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
    return np.transpose(data, (1, 2, 0))


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an (H, W, C) array with OpenCV, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV failed to write {path}")


def list_images(source: str | Path) -> list[Path]:
    """Return a sorted list of image files from a single file or a folder."""
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No such file or folder: {source}")
