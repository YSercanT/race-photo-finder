import numpy as np
import pytest

from photo_finder.faces import Face, average_embedding, cosine_similarity


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cosine_similarity_identical_orthogonal_opposite():
    q = unit([1, 0, 0])
    embeddings = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([-1, 0, 0])])
    assert cosine_similarity(q, embeddings) == pytest.approx([1.0, 0.0, -1.0])


def test_average_embedding_is_unit_length():
    avg = average_embedding([unit([1, 0]), unit([0, 1])])
    assert np.linalg.norm(avg) == pytest.approx(1.0)
    assert avg == pytest.approx(unit([1, 1]))


def test_face_size_is_short_side():
    face = Face(bbox=(10, 20, 110, 70), det_score=0.9, embedding=unit([1, 0]))
    assert face.size == 50
