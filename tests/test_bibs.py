from photo_finder.bibs import BibReading, parse_bibs

BOX = [[0, 0], [1, 0], [1, 1], [0, 1]]  # konum testlerde onemsiz


def test_keeps_confident_numbers():
    raw = [(BOX, "1234", 0.9)]
    assert parse_bibs(raw, min_confidence=0.5) == [BibReading("1234", 0.9)]


def test_drops_low_confidence():
    raw = [(BOX, "1234", 0.2)]
    assert parse_bibs(raw, min_confidence=0.5) == []


def test_strips_spaces_and_non_digits():
    raw = [(BOX, "12 34", 0.9)]
    assert parse_bibs(raw, min_confidence=0.5)[0].number == "1234"


def test_filters_by_length():
    raw = [(BOX, "7", 0.9), (BOX, "123456", 0.9), (BOX, "42", 0.9)]
    numbers = [r.number for r in parse_bibs(raw, min_confidence=0.5, min_digits=2, max_digits=5)]
    assert numbers == ["42"]


def test_keeps_leading_zeros():
    raw = [(BOX, "0042", 0.9)]
    assert parse_bibs(raw, min_confidence=0.5)[0].number == "0042"


def test_duplicates_keep_best_confidence_and_sorted():
    raw = [(BOX, "1234", 0.6), (BOX, "555", 0.95), (BOX, "1234", 0.8)]
    assert parse_bibs(raw, min_confidence=0.5) == [BibReading("555", 0.95), BibReading("1234", 0.8)]
