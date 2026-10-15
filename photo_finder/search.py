"""Indeks uzerinde numara/yuz ile arama (hizli: model calistirmaz, sadece okur)."""

import re
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from photo_finder import config, database
from photo_finder.faces import cosine_similarity


@dataclass(frozen=True)
class Match:
    file_id: str
    bib_match: bool
    face_score: float | None = None  # fotograftaki en benzer yuzun skoru; yuz aranmadiysa None
    face_size: int | None = None     # o yuzun boyutu (piksel); kucukse arka planda yakalanmissin


def best_faces(file_ids: list[str], sizes: list[int], scores: np.ndarray) -> dict[str, tuple[float, int]]:
    """Yuz bazli skorlari fotograf bazina indirger: bir fotografta 5 yuz varsa en benzeri.

    Fotografin odaginda kim oldugu onemsiz; arka plandaki yuz de en benzer olabilir.
    """
    best: dict[str, tuple[float, int]] = {}
    for file_id, size, score in zip(file_ids, sizes, scores):
        if file_id not in best or score > best[file_id][0]:
            best[file_id] = (float(score), size)
    return best


def search(conn: sqlite3.Connection, file_ids: Collection[str], bib: str | None = None,
           reference: np.ndarray | None = None,
           face_threshold: float = config.FACE_SIMILARITY_THRESHOLD,
           ocr_min_confidence: float = config.OCR_MIN_CONFIDENCE,
           min_face_size: int = config.FACE_MIN_SIZE) -> list[Match]:
    """Secili fotograflar icinde numarasi VEYA yuzu eslesenler.

    Siralama: once ikisi de tutanlar, sonra numara, sonra yuz skoru.
    """
    bib_ids: set[str] = set()
    if bib and (number := re.sub(r"\D", "", bib)):
        bib_ids = database.photos_with_bib(conn, file_ids, number, ocr_min_confidence)

    faces: dict[str, tuple[float, int]] = {}
    if reference is not None:
        ids, sizes, embeddings = database.load_faces(conn, file_ids, min_size=min_face_size)
        faces = best_faces(ids, sizes, cosine_similarity(reference, embeddings))

    face_ids = {fid for fid, (score, _) in faces.items() if score >= face_threshold}
    matches = [
        Match(fid, fid in bib_ids, *faces.get(fid, (None, None)))
        for fid in bib_ids | face_ids
    ]
    return sorted(matches, key=lambda m: (
        not (m.bib_match and (m.face_score or 0) >= face_threshold),
        not m.bib_match,
        -(m.face_score or 0),
    ))
