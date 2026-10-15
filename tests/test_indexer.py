import time

from photo_finder.indexer import prefetch


def test_prefetch_keeps_order_even_if_finishing_out_of_order():
    def slow_for_small(x):
        time.sleep(0.05 if x < 3 else 0)
        return x * 10

    assert list(prefetch(range(6), slow_for_small, workers=3)) == [(i, i * 10) for i in range(6)]


def test_prefetch_returns_errors_instead_of_crashing():
    def fail_on_two(x):
        if x == 2:
            raise ValueError("bozuk")
        return x

    results = dict(prefetch(range(4), fail_on_two, workers=2))
    assert isinstance(results[2], ValueError)
    assert results[3] == 3
