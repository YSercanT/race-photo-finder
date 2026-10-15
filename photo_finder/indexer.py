"""Drive'daki yaris fotograflarini tarayip indekse yazma (yavas, fotograf basina bir kez)."""

import sqlite3
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from itertools import islice

import requests

from photo_finder import config, database
from photo_finder.bibs import get_reader, read_bibs
from photo_finder.download import DriveFile, fetch_image_bytes
from photo_finder.faces import detect_faces, get_app
from photo_finder.images import load_image


@dataclass(frozen=True)
class ScanProgress:
    done: int
    total: int
    file: DriveFile
    error: str | None = None


def prefetch(items: Iterable, fn: Callable, workers: int) -> Iterator[tuple]:
    """`fn(item)`'i arka planda paralel calistirir, sonuclari SIRAYLA (item, sonuc) olarak verir.

    Indirme agi bekler (I/O), isleme GPU/CPU'yu; ikisini ust uste bindirerek toplam sure kisalir.
    Pencere `workers * 2` ile sinirli: islemeden hizli indirip bellegi doldurmayiz.
    Hata olursa sonuc yerine Exception nesnesi doner; tek bozuk dosya taramayi durdurmaz.
    """
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        window = deque((item, pool.submit(fn, item)) for item in islice(iterator, workers * 2))
        while window:
            item, future = window.popleft()
            if (next_item := next(iterator, None)) is not None:
                window.append((next_item, pool.submit(fn, next_item)))
            try:
                yield item, future.result()
            except Exception as exc:  # noqa: BLE001 - hatayi cagirana raporluyoruz
                yield item, exc


_local = threading.local()


def _session() -> requests.Session:
    """Her thread kendi Session'ini kullanir (requests.Session thread-safe garanti etmez)."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def index_drive_files(conn: sqlite3.Connection, files: list[DriveFile], api_key: str) -> Iterator[ScanProgress]:
    """Henuz islenmemis fotograflari indir -> OCR + yuz -> indekse yaz. Ilerlemeyi adim adim bildirir."""
    done_ids = database.processed_ids(conn, (f.id for f in files))
    todo = [f for f in files if f.id not in done_ids]

    def fetch(file: DriveFile) -> bytes:
        return fetch_image_bytes(file, api_key, session=_session())

    for i, (file, data) in enumerate(prefetch(todo, fetch, config.DOWNLOAD_WORKERS), start=1):
        error = None
        try:
            if isinstance(data, Exception):
                raise data
            image = load_image(BytesIO(data))
            # Esik 0: ham okumalari sakla, filtreyi arama aninda uygula (bkz. database.py)
            database.save_photo(conn, file, read_bibs(image, min_confidence=0.0), detect_faces(image))
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        yield ScanProgress(i, len(todo), file, error)


def load_models() -> None:
    """OCR ve yuz modellerini onceden yukler (ikisi de lru_cache'li; tekrar cagirmak bedava)."""
    get_reader()
    get_app()
