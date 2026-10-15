"""EasyOCR ile fotograftaki gogus numaralarini okuma."""

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from photo_finder import config


@dataclass(frozen=True)
class BibReading:
    number: str        # "1234" (bastaki sifirlar korunur: "0042" ile "42" farkli numaralar olabilir)
    confidence: float  # 0-1 arasi EasyOCR guveni


@lru_cache(maxsize=1)
def get_reader():
    """EasyOCR modelini bir kez yukler, sonraki cagrilarda ayni nesneyi dondurur.

    Model yuklemek birkac saniye surer ve GPU bellegi kaplar; her fotografta yeniden yuklemek
    3.000 fotografta saatler kaybettirir. Import da burada: easyocr (ve PyTorch) agir oldugu icin
    sadece gercekten OCR yapilacaksa yuklensin.
    """
    import easyocr

    # gpu=True: CUDA yoksa EasyOCR uyari verip CPU'ya gecer
    return easyocr.Reader(["en"], gpu=True, verbose=False)


def parse_bibs(
    raw_results: list[tuple],
    min_confidence: float = config.OCR_MIN_CONFIDENCE,
    min_digits: int = config.BIB_MIN_DIGITS,
    max_digits: int = config.BIB_MAX_DIGITS,
) -> list[BibReading]:
    """EasyOCR'in ham ciktisini [(bbox, text, confidence), ...] numara listesine cevirir.

    Modelden bagimsiz saf bir fonksiyon: model olmadan test edilebilir.
    """
    best: dict[str, float] = {}
    for _bbox, text, confidence in raw_results:
        if confidence < min_confidence:
            continue
        number = re.sub(r"\D", "", text)  # "12 34" -> "1234"
        if not (min_digits <= len(number) <= max_digits):
            continue
        # Ayni numara birden fazla kez okunduysa en emin okumayi tut
        best[number] = max(confidence, best.get(number, 0.0))

    readings = [BibReading(number, conf) for number, conf in best.items()]
    return sorted(readings, key=lambda r: r.confidence, reverse=True)


def read_bibs(image: np.ndarray, min_confidence: float = config.OCR_MIN_CONFIDENCE) -> list[BibReading]:
    """BGR resimdeki gogus numaralarini okur."""
    # allowlist: modele "sadece rakam var" diyoruz; O->0, I->1, S->5 karisikliklarini azaltir
    raw = get_reader().readtext(image, allowlist="0123456789")
    return parse_bibs(raw, min_confidence=min_confidence)


if __name__ == "__main__":
    # Hizli kontrol: python -m photo_finder.bibs data/races/<yaris>
    from photo_finder.images import list_images, load_image

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RACES_DIR
    for path in list_images(folder):
        readings = read_bibs(load_image(path))
        found = ", ".join(f"{r.number} ({r.confidence:.2f})" for r in readings) or "-"
        print(f"{path.name}: {found}")
