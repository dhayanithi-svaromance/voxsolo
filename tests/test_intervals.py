"""Tests for the interval algebra. Run with: pytest -q  (or: python tests/test_intervals.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxsolo.intervals import (  # noqa: E402
    complement,
    coverage_at_least,
    exclusive_by_label,
    intersect,
    merge,
    pad_into_free_space,
    subtract,
    total_duration,
    union,
)


def test_merge():
    assert merge([[0, 1], [1, 2]]) == [[0, 2]]
    assert merge([[0, 3], [1, 2], [5, 6]]) == [[0, 3], [5, 6]]
    assert merge([[1, 1], [2, 3]]) == [[2, 3]]
    assert merge([]) == []


def test_intersect():
    assert intersect([[0, 5]], [[3, 8]]) == [[3, 5]]
    assert intersect([[0, 2]], [[5, 8]]) == []
    assert intersect([[0, 10]], [[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_subtract():
    assert subtract([[0, 10]], [[3, 5]]) == [[0, 3], [5, 10]]
    assert subtract([[0, 10]], [[1, 2], [3, 4], [9, 20]]) == [[0, 1], [2, 3], [4, 9]]
    assert subtract([[2, 4]], [[0, 10]]) == []
    assert subtract([[0, 5]], []) == [[0, 5]]


def test_union_and_complement():
    assert union([[0, 2]], [[1, 4]], [[7, 8]]) == [[0, 4], [7, 8]]
    assert complement([[2, 4]], 0, 10) == [[0, 2], [4, 10]]


def test_overlap_generalises_beyond_two_speakers():
    spk = {"A": [[0, 10]], "B": [[4, 12]], "C": [[8, 20]]}
    assert coverage_at_least(spk, 2) == [[4, 12]]
    assert coverage_at_least(spk, 3) == [[8, 10]]


def test_one_speaker_overlapping_himself_is_not_overlap():
    assert coverage_at_least({"A": [[0, 5], [2, 7]]}, 2) == []


def test_touching_segments_are_not_overlap():
    assert coverage_at_least({"A": [[0, 5]], "B": [[5, 9]]}, 2) == []


def test_exclusive_by_label():
    spk = {"A": [[0, 10]], "B": [[4, 12]], "C": [[8, 20]]}
    ex = exclusive_by_label(spk)
    assert ex["A"] == [[0, 4]]
    assert ex["B"] == []          # B is fully overlapped by A and/or C
    assert ex["C"] == [[12, 20]]


def test_total_duration_counts_overlap_once():
    assert total_duration([[0, 2], [1, 5], [9, 10]]) == 6.0


def test_padding_never_enters_blocked_space():
    assert pad_into_free_space([[5, 6]], [[0, 1]], 0.5, 0, 10) == [[4.5, 6.5]]
    assert pad_into_free_space([[5, 6]], [[6.2, 9]], 0.5, 0, 10) == [[4.5, 6.2]]
    assert pad_into_free_space([[0, 1]], [], 0.5, 0, 10) == [[0, 1.5]]


def test_padding_only_adds_never_removes():
    # even if the kept region itself intersects 'blocked', it survives intact
    assert pad_into_free_space([[5, 6]], [[5.5, 5.7]], 0.5, 0, 10) == [[4.5, 6.5]]


def test_padding_drops_disconnected_islands():
    # a blocked interval strictly inside the pad zone must not leave an
    # isolated pad sliver on its far side
    assert pad_into_free_space([[5, 6]], [[4.6, 4.8]], 0.5, 0, 10) == [[4.8, 6.5]]
    # blocked hard against the region: no leading pad at all, no island
    assert pad_into_free_space([[5, 6]], [[4.7, 5.0]], 0.5, 0, 10) == [[5, 6.5]]


def test_merge_rejects_reversed_intervals():
    try:
        merge([[5, 2]])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for reversed interval")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
