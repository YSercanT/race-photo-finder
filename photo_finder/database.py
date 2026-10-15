"""SQLite indeksine sonuclari yazma ve okuma.

Ilkeler:
- Indeks fotograf bazli: hangi linkten/yaristan geldigi onemsiz, Drive dosya ID'si anahtar.
  Ayni fotograf iki kez taranmaz; secim degisse de eski taramalar bosa gitmez.
- Ham sonuclari sakla, filtreyi arama aninda uygula: esikler degisince yeniden tarama gerekmez.
"""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from photo_finder import config
from photo_finder.bibs import BibReading
from photo_finder.download import DriveFile
from photo_finder.faces import Face

EMBEDDING_DIM = 512

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    file_id TEXT PRIMARY KEY,       -- Drive dosya ID'si
    name    TEXT NOT NULL,
    folder  TEXT NOT NULL           -- alt klasor: CP / fotografci
);
CREATE TABLE IF NOT EXISTS bibs (
    file_id    TEXT NOT NULL REFERENCES photos(file_id),
    number     TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS faces (
    file_id   TEXT NOT NULL REFERENCES photos(file_id),
    det_score REAL NOT NULL,
    size      INTEGER NOT NULL,     -- yuzun kisa kenari (piksel); kucukse arka planda
    embedding BLOB NOT NULL         -- 512 x float32 = 2048 bayt
);
CREATE INDEX IF NOT EXISTS idx_bibs_file ON bibs(file_id);
CREATE INDEX IF NOT EXISTS idx_faces_file ON faces(file_id);
"""


def connect(path: Path | str = config.INDEX_DB, timeout: float = 5.0) -> sqlite3.Connection:
    """Veritabanina baglanir.

    isolation_level=None (autocommit): Python'un sqlite3 modulu varsayilan olarak her yazmadan once
    sessizce transaction acar ve commit edilene kadar kilit tutar. Streamlit her rerun'da yeni
    baglanti acip eskisini acik biraktigi icin, eski baglantinin unutulmus transaction'i yeni
    baglantinin yazmasini kilitliyordu ('database is locked'). Artik transaction'lari sadece
    save_photo'da, elle aciyoruz.
    WAL modu: okuyanlar yazani, yazan okuyanlari beklemez.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _set_scope(conn: sqlite3.Connection, file_ids: Iterable[str]) -> None:
    """Sorgulari secili fotograflarla sinirlamak icin gecici tablo.

    `WHERE file_id IN (?, ?, ...)` 20.000 parametreyle calismaz; gecici tablo + JOIN calisir
    ve indeks sayesinde hizlidir. TEMP tablo sadece bu baglantida yasar, diske yazilmaz.
    """
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS scope (file_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM scope")
    conn.executemany("INSERT OR IGNORE INTO scope VALUES (?)", ((fid,) for fid in file_ids))


def processed_ids(conn: sqlite3.Connection, file_ids: Iterable[str]) -> set[str]:
    """Verilen fotograflardan hangileri daha once taranmis?"""
    _set_scope(conn, file_ids)
    return {fid for (fid,) in conn.execute("SELECT file_id FROM photos JOIN scope USING (file_id)")}


def save_photo(conn: sqlite3.Connection, file: DriveFile, bibs: list[BibReading], faces: list[Face]) -> None:
    """Bir fotografin tum sonuclarini tek transaction'da yazar: ya hepsi ya hicbiri.

    Tarama yarida kesilirse yarim kayit kalmaz; fotograf "islenmemis" sayilip tekrar islenir.
    """
    conn.execute("BEGIN IMMEDIATE")  # yazma kilidini en basta al
    try:
        conn.execute("DELETE FROM bibs WHERE file_id = ?", (file.id,))
        conn.execute("DELETE FROM faces WHERE file_id = ?", (file.id,))
        conn.execute("INSERT OR REPLACE INTO photos VALUES (?, ?, ?)", (file.id, file.name, file.folder))
        conn.executemany("INSERT INTO bibs VALUES (?, ?, ?)",
                         [(file.id, b.number, b.confidence) for b in bibs])
        conn.executemany("INSERT INTO faces VALUES (?, ?, ?, ?)",
                         [(file.id, f.det_score, f.size, f.embedding.astype(np.float32).tobytes()) for f in faces])
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def photos_with_bib(conn: sqlite3.Connection, file_ids: Iterable[str], number: str,
                    min_confidence: float) -> set[str]:
    _set_scope(conn, file_ids)
    rows = conn.execute(
        """SELECT DISTINCT file_id FROM bibs JOIN scope USING (file_id)
           WHERE number = ? AND confidence >= ?""",
        (number, min_confidence),
    )
    return {fid for (fid,) in rows}


def load_faces(conn: sqlite3.Connection, file_ids: Iterable[str],
               min_size: int = 0) -> tuple[list[str], list[int], np.ndarray]:
    """Secili fotograflardaki yuzleri tek matris olarak dondurur: (file_id'ler, boyutlar, (N, 512))."""
    _set_scope(conn, file_ids)
    rows = conn.execute(
        "SELECT file_id, size, embedding FROM faces JOIN scope USING (file_id) WHERE size >= ?", (min_size,)
    ).fetchall()
    if not rows:
        return [], [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    ids = [fid for fid, _, _ in rows]
    sizes = [size for _, size, _ in rows]
    embeddings = np.stack([np.frombuffer(blob, dtype=np.float32) for _, _, blob in rows])
    return ids, sizes, embeddings
