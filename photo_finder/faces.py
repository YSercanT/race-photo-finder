"""InsightFace ile yuz tespiti ve embedding cikarma."""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from photo_finder import config


@dataclass(frozen=True)
class Face:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) piksel
    det_score: float                 # tespit guveni
    embedding: np.ndarray            # 512 boyutlu, L2-normalize (uzunlugu 1)

    @property
    def size(self) -> int:
        """Yuzun kisa kenari (piksel)."""
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


@lru_cache(maxsize=1)
def get_app():
    """InsightFace modelini bir kez yukler (bkz. bibs.get_reader ile ayni mantik)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=config.FACE_MODEL,
        # Sadece ihtiyacimiz olan iki model: yas/cinsiyet/landmark modellerini atlamak hiz kazandirir
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.FACE_DET_SIZES)
    return app


def detect_faces(image: np.ndarray) -> list[Face]:
    """BGR resimdeki yuzleri bulur (arka plandakiler dahil), cok kucuk ve belirsiz olanlari eler."""
    faces = []
    for f in get_app().get(image):
        face = Face(
            bbox=tuple(int(v) for v in f.bbox),
            det_score=float(f.det_score),
            embedding=f.normed_embedding.astype(np.float32),
        )
        if face.det_score >= config.FACE_MIN_DET_SCORE and face.size >= config.FACE_MIN_SIZE_INDEX:
            faces.append(face)
    return faces


def cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Normalize vektorlerde cosine similarity = nokta carpimi.

    query: (512,), embeddings: (N, 512) -> (N,) skor, -1 ile 1 arasi.
    Tek bir matris carpimi: 15.000 yuz icin milisaniyeler.
    """
    return embeddings @ query


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Birden fazla referans fotografin embedding'ini tek vektore indirger.

    Ortalama, tek fotograftaki isik/aci tesadufunu yumusatir. Ortalama alininca uzunluk 1'den
    kuculur; tekrar normalize etmezsek cosine_similarity yanlis sonuc verir.
    """
    mean = np.mean(embeddings, axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def build_reference(images: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Referans fotograflardan (selfie vb.) kisinin yuz vektorunu cikarir.

    Her fotografta en buyuk yuz alinir: referans fotografta arkada baska biri de olabilir.
    Doner: (yuz vektoru, yuz bulunan fotograf sayisi).
    """
    embeddings = []
    for image in images:
        faces = detect_faces(image)
        if faces:
            embeddings.append(max(faces, key=lambda f: f.size).embedding)
    if not embeddings:
        raise ValueError("Yukledigin fotograflarda yuz bulunamadi. Yuzunun net ve tam gorundugu, "
                         "cok karanlik olmayan bir fotograf dene.")
    return average_embedding(embeddings), len(embeddings)


if __name__ == "__main__":
    # Hizli kontrol: python -m photo_finder.faces data/races/<yaris> [data/reference]
    from photo_finder.images import list_images, load_image

    race_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RACES_DIR
    ref_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else config.REFERENCE_DIR

    reference, _ = build_reference([load_image(p) for p in list_images(ref_dir)])
    for path in list_images(race_dir):
        faces = detect_faces(load_image(path))
        if not faces:
            print(f"{path.name}: yuz yok")
            continue
        scores = cosine_similarity(reference, np.stack([f.embedding for f in faces]))
        best = float(scores.max())
        mark = "  <-- SEN" if best >= config.FACE_SIMILARITY_THRESHOLD else ""
        print(f"{path.name}: {len(faces)} yuz, en yuksek benzerlik {best:.2f}{mark}")
