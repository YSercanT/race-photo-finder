"""Gercek InsightFace modeliyle testler (model indirilmemisse atlanir)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("insightface")
if not (Path.home() / ".insightface/models/buffalo_l").exists():
    pytest.skip("buffalo_l modeli indirilmemis", allow_module_level=True)

from insightface.data import get_image  # noqa: E402

from photo_finder.faces import build_reference, cosine_similarity, detect_faces  # noqa: E402


@pytest.fixture(scope="module")
def group_and_selfie():
    group = get_image("t1")  # kutuphanenin ornek grup fotografi
    face = max(detect_faces(group), key=lambda f: f.size)
    x1, y1, x2, y2 = face.bbox
    w = x2 - x1
    crop = group[max(0, y1 - w // 3):y2 + w // 3, max(0, x1 - w // 4):x2 + w // 4]
    selfie = cv2.resize(crop, (1536, 2048))  # yuz ekrani dolduruyor, telefon selfie'si gibi
    return group, face, selfie


def test_close_up_selfie_face_is_detected(group_and_selfie):
    """Regresyon: tek olcekli (1024) tespit ekrani dolduran yuzleri kaciriyordu."""
    _, _, selfie = group_and_selfie
    assert len(detect_faces(selfie)) == 1


def test_selfie_matches_same_person_in_group(group_and_selfie):
    group, face, selfie = group_and_selfie
    reference, n_used = build_reference([selfie])
    assert n_used == 1
    faces = detect_faces(group)
    scores = cosine_similarity(reference, np.stack([f.embedding for f in faces]))
    best = faces[int(scores.argmax())]
    assert best.bbox == face.bbox   # en benzer yuz, selfie'yi kestigimiz kisi
    assert scores.max() > 0.6
