import numpy as np
import pytest

from photo_finder import database, search
from photo_finder.bibs import BibReading
from photo_finder.download import DriveFile
from photo_finder.faces import Face


def unit(v):
    v = np.zeros(512, dtype=np.float32) + np.asarray(v + [0] * (512 - len(v)), dtype=np.float32)
    return v / np.linalg.norm(v)


ME, OTHER = unit([1, 0]), unit([0, 1])


def face(embedding, size=100):
    return Face(bbox=(0, 0, size, size), det_score=0.9, embedding=embedding)


@pytest.fixture
def conn():
    conn = database.connect(":memory:")
    # p1: numaram + yuzum, p2: sadece numaram (dusuk guven), p3: sadece yuzum, p4: baska biri
    # x: secili olmayan (baska yaris/klasor) fotograf; aramaya hic girmemeli
    database.save_photo(conn, DriveFile("p1", "1.jpg", "CP1"), [BibReading("1234", 0.9)], [face(ME)])
    database.save_photo(conn, DriveFile("p2", "2.jpg", "CP1"), [BibReading("1234", 0.3)], [])
    database.save_photo(conn, DriveFile("p3", "3.jpg", "CP2"), [], [face(OTHER), face(ME)])
    database.save_photo(conn, DriveFile("p4", "4.jpg", "CP2"), [BibReading("999", 0.9)], [face(OTHER)])
    database.save_photo(conn, DriveFile("x", "x.jpg", ""), [BibReading("1234", 0.9)], [face(ME)])
    # p5: onde baskasi (buyuk yuz), ben arkada kucuk yuzum
    database.save_photo(conn, DriveFile("p5", "5.jpg", "Start"), [], [face(OTHER, 300), face(ME, 25)])
    return conn


SELECTED = ["p1", "p2", "p3", "p4", "p5"]


def ids(matches):
    return [m.file_id for m in matches]


def test_embedding_roundtrip(conn):
    file_ids, sizes, embeddings = database.load_faces(conn, SELECTED)
    assert sorted(file_ids) == ["p1", "p3", "p3", "p4", "p5", "p5"]
    assert sorted(sizes) == [25, 100, 100, 100, 100, 300]
    assert embeddings.shape == (6, 512) and embeddings.dtype == np.float32


def test_resave_replaces_old_results(conn):
    database.save_photo(conn, DriveFile("p1", "1.jpg", "CP1"), [], [])
    assert search.search(conn, SELECTED, bib="1234", ocr_min_confidence=0.5) == []


def test_bib_search_filters_confidence_at_query_time(conn):
    assert ids(search.search(conn, SELECTED, bib="1234", ocr_min_confidence=0.5)) == ["p1"]
    assert sorted(ids(search.search(conn, SELECTED, bib="1234", ocr_min_confidence=0.2))) == ["p1", "p2"]


def test_face_search_uses_best_face_in_photo(conn):
    assert sorted(ids(search.search(conn, SELECTED, reference=ME, face_threshold=0.5, min_face_size=40))) == ["p1", "p3"]


def test_combined_ranking_both_first(conn):
    result = search.search(conn, SELECTED, bib="1234", reference=ME, face_threshold=0.5, ocr_min_confidence=0.2,
                           min_face_size=40)
    assert ids(result)[0] == "p1"
    assert set(ids(result)) == {"p1", "p2", "p3"}


def test_processed_ids_only_reports_given_photos(conn):
    assert database.processed_ids(conn, ["p1", "x", "never_scanned"]) == {"p1", "x"}


def test_scope_handles_many_ids(conn):
    many = SELECTED + [f"fake{i}" for i in range(40_000)]  # SQLite parametre sinirini asar
    assert ids(search.search(conn, many, bib="1234", ocr_min_confidence=0.5)) == ["p1"]


def test_background_face_found_only_when_small_faces_included(conn):
    without = search.search(conn, SELECTED, reference=ME, face_threshold=0.5, min_face_size=40)
    assert "p5" not in ids(without)
    with_bg = search.search(conn, SELECTED, reference=ME, face_threshold=0.5, min_face_size=20)
    p5 = next(m for m in with_bg if m.file_id == "p5")
    assert p5.face_score == pytest.approx(1.0)  # ondeki yabanci degil, arkadaki ben
    assert p5.face_size == 25


def test_bib_only_match_has_no_face_info(conn):
    (match,) = search.search(conn, SELECTED, bib="1234", ocr_min_confidence=0.5)
    assert match.face_score is None and match.face_size is None


def test_old_connection_does_not_block_writes(tmp_path):
    """Streamlit her rerun'da yeni baglanti acar; eskisi acik kalir. Okuma yapmis eski bir
    baglanti, yenisinin yazmasini kilitlememeli (eskiden: 'database is locked')."""
    path = tmp_path / "index.db"
    old = database.connect(path)
    database.processed_ids(old, ["p1"])  # eski rerun: okudu, baglanti acik kaldi
    new = database.connect(path, timeout=0.5)
    database.save_photo(new, DriveFile("p1", "1.jpg", ""), [BibReading("1234", 0.9)], [])
    assert database.processed_ids(old, ["p1"]) == {"p1"}
