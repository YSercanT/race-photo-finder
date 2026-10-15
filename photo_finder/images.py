"""Fotograflari listeleme ve EXIF yonu duzeltilmis sekilde okuma."""

import sys
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from photo_finder import config


def list_images(folder: Path) -> list[Path]:
    """Klasordeki (alt klasorler dahil) tum fotograflari sirali dondurur."""
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def load_image(source: Path | BinaryIO, max_side: int = config.MAX_IMAGE_SIDE) -> np.ndarray:
    """Resmi okuyup BGR numpy dizisi olarak dondurur (OpenCV / InsightFace formati).

    `source` dosya yolu da olabilir, bellekteki bir dosya da (Streamlit upload, Drive indirmesi).
    """
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)    # kameranin "90 derece dondur" notunu uygula
        img = img.convert("RGB")              # PNG alfa kanali, gri tonlama vb. -> 3 kanal
        img.thumbnail((max_side, max_side))   # en-boy oranini koruyarak sadece kucultur
        rgb = np.asarray(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # Hizli kontrol: python -m photo_finder.images data/races/<yaris>
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RACES_DIR
    paths = list_images(folder)
    print(f"{len(paths)} fotograf bulundu: {folder}")
    for path in paths[:5]:
        print(f"  {path.name}: {load_image(path).shape}")
