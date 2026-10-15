from collections import Counter

from photo_finder.folders import group_folders, label, matches_any, normalize, parse_keywords


def test_normalize_turkish():
    assert normalize("ÖDÜL TÖRENİ") == "odul toreni"
    assert normalize("Çocuk Koşusu") == "cocuk kosusu"
    assert normalize("CUMARTESI") == "cumartesi"


def test_keywords_match_without_turkish_chars():
    keywords = parse_keywords("odul, COCUK ,")
    assert keywords == ["odul", "cocuk"]
    assert matches_any("GAMZE NUR PULAŞ/ÖDÜL TÖRENİ", keywords)
    assert matches_any("OKAN YILMAZ/Çocuk Koşu", keywords)
    assert not matches_any("OKAN YILMAZ/Pazar", keywords)


def test_keywords_ignore_spaces_and_dashes():
    keywords = parse_keywords("21k")
    assert matches_any("21 K", keywords)
    assert matches_any("10K - 21K MIX", keywords)
    assert not matches_any("5-10K FINISH", keywords)


def test_keywords_match_parent_folders_too():
    # "burak" yazinca fotografcinin tum alt klasorleri eslesmeli
    assert matches_any("BURAK BAŞTÜRK/SATURDAY/3-PAMUCAK", parse_keywords("burak"))


def test_group_folders_by_top_level():
    groups = group_folders(["B/x", "A", "B/y/z", ""])
    assert groups == {"(ana klasor)": [""], "A": ["A"], "B": ["B/x", "B/y/z"]}


def test_label_strips_group_name():
    counts = Counter({"B/y/z": 5, "A": 3})
    assert label("B/y/z", "B", counts) == "y/z — 5 foto"
    assert label("A", "A", counts) == "(dogrudan bu klasor) — 3 foto"
