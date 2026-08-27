"""Interval algebra for diarization timelines.

An "interval set" is a list of ``[start, end]`` pairs in seconds. Functions that
return interval sets always return them **merged and sorted**: non-overlapping,
non-touching, ascending. Zero-length intervals are dropped.

These operations are deliberately independent of any diarization backend so
they can be unit-tested on their own.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

Interval = List[float]
IntervalSet = List[Interval]

# Intervals closer than this are treated as touching and get merged. Also the
# threshold below which an interval is considered empty.
EPS = 1e-9


def merge(intervals: Iterable[Sequence[float]], eps: float = EPS) -> IntervalSet:
    """Merge overlapping/adjacent intervals into a canonical interval set.

    Zero-length intervals are dropped; a reversed interval (``end < start``)
    raises ``ValueError`` -- it always indicates corrupted input (e.g. a
    hand-edited timeline JSON), and silently discarding it would silently
    discard speech.
    """
    items = sorted(([float(s), float(e)] for s, e in intervals), key=lambda iv: iv[0])
    out: IntervalSet = []
    for start, end in items:
        if end < start - eps:
            raise ValueError(f"reversed interval [{start}, {end}]")
        if end - start <= eps:
            continue
        if out and start <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def union(*interval_sets: Iterable[Sequence[float]]) -> IntervalSet:
    """Union of any number of interval sets."""
    combined: List[Sequence[float]] = []
    for iv_set in interval_sets:
        combined.extend(iv_set)
    return merge(combined)


def intersect(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> IntervalSet:
    """Intersection of two interval sets."""
    xs, ys = merge(a), merge(b)
    out: IntervalSet = []
    i = j = 0
    while i < len(xs) and j < len(ys):
        start = max(xs[i][0], ys[j][0])
        end = min(xs[i][1], ys[j][1])
        if end - start > EPS:
            out.append([start, end])
        # Advance whichever interval ends first.
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> IntervalSet:
    """Everything in ``a`` that is not in ``b``."""
    minuend, subtrahend = merge(a), merge(b)
    if not subtrahend:
        return minuend

    out: IntervalSet = []
    for start, end in minuend:
        cursor = start
        for bs, be in subtrahend:
            if be <= cursor:          # entirely before the remaining piece
                continue
            if bs >= end:             # past the end; nothing left to cut
                break
            if bs > cursor:
                out.append([cursor, bs])
            cursor = max(cursor, be)
            if cursor >= end:
                break
        if end - cursor > EPS:
            out.append([cursor, end])
    return merge(out)


def complement(a: Iterable[Sequence[float]], start: float, end: float) -> IntervalSet:
    """Everything within ``[start, end]`` that is not in ``a``."""
    return subtract([[start, end]], a)


def coverage_at_least(
    interval_sets: Mapping[str, Iterable[Sequence[float]]], k: int
) -> IntervalSet:
    """Regions covered by at least ``k`` of the given labels.

    Each label's own intervals are merged first, so a single speaker with two
    overlapping turns still counts as depth 1 -- this is what makes the result
    mean "k distinct speakers" rather than "k segments".

    With ``k=2`` this is the general N-speaker definition of overlapped speech.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    events: List[tuple] = []
    for ivs in interval_sets.values():
        for start, end in merge(ivs):
            events.append((start, 1))
            events.append((end, -1))
    if not events:
        return []

    # Sort by time; process exits (-1) before entries (+1) at equal timestamps so
    # that intervals which merely touch are not reported as overlapping.
    events.sort(key=lambda ev: (ev[0], ev[1]))

    out: IntervalSet = []
    depth = 0
    region_start = None
    for time, delta in events:
        was_deep = depth >= k
        depth += delta
        is_deep = depth >= k
        if not was_deep and is_deep:
            region_start = time
        elif was_deep and not is_deep and region_start is not None:
            if time - region_start > EPS:
                out.append([region_start, time])
            region_start = None
    return merge(out)


def total_duration(intervals: Iterable[Sequence[float]]) -> float:
    """Total covered time of an interval set."""
    return sum(e - s for s, e in merge(intervals))


def exclusive_by_label(
    interval_sets: Mapping[str, Iterable[Sequence[float]]]
) -> Dict[str, IntervalSet]:
    """Per-label speech with all overlapped (>=2 speaker) regions removed."""
    overlap = coverage_at_least(interval_sets, 2)
    return {
        label: subtract(ivs, overlap) for label, ivs in interval_sets.items()
    }


def pad_into_free_space(
    keep: Iterable[Sequence[float]],
    blocked: Iterable[Sequence[float]],
    pad: float,
    lower: float,
    upper: float,
) -> IntervalSet:
    """Grow ``keep`` by ``pad`` seconds on each side without entering ``blocked``.

    Diarization boundaries tend to clip word onsets and offsets slightly. This
    recovers a little lead-in/out, but only into genuinely free space: the result
    never intrudes on ``blocked`` (typically other speakers' speech plus the
    overlapped regions), and never leaves ``[lower, upper]``.

    The original ``keep`` is always fully contained in the result, even if it
    intersects ``blocked`` -- padding may only add time, never remove it.

    Pad fragments that a blocked interval disconnects from their region are
    dropped: padding must stay contiguous with the speech it cushions, never
    reappear as an isolated sliver on the far side of another speaker's turn.
    """
    base = merge(keep)
    if pad <= 0:
        return base
    grown = [
        [max(lower, s - pad), min(upper, e + pad)] for s, e in base
    ]
    free = subtract(grown, blocked)
    attached = [
        frag for frag in free
        if any(frag[0] <= b_end + EPS and b_start <= frag[1] + EPS
               for b_start, b_end in base)
    ]
    return union(attached, base)
