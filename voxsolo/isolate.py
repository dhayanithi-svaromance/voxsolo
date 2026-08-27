"""Render an isolated speaker track.

Keeps one speaker's speech and silences everything else -- other speakers,
overlapped speech, and (optionally) non-speech gaps. Kept samples are copied
verbatim from the source, so background noise inside kept regions is untouched:
no denoising, no normalisation, no gain change. Only the short fade ramps at
region edges (half-cosine, ``fade`` seconds, for click-free cuts) modify any
sample; set ``fade=0`` for bit-exact edges at the cost of possible clicks.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf

from . import intervals as iv
from .diarize import Timeline


@dataclass
class IsolateResult:
    out_path: str
    speaker: str
    kept_regions: List[List[float]]
    kept_seconds: float
    total_seconds: float
    clips_path: Optional[str] = None

    @property
    def kept_fraction(self) -> float:
        return self.kept_seconds / self.total_seconds if self.total_seconds else 0.0

    def summary(self) -> str:
        return (
            f"{self.speaker}: kept {self.kept_seconds:.2f}s of "
            f"{self.total_seconds:.2f}s ({self.kept_fraction * 100:.1f}%) "
            f"in {len(self.kept_regions)} region(s)"
        )


def compute_keep_regions(
    timeline: Timeline,
    speaker: str,
    pad: float = 0.05,
    keep_overlap: bool = False,
    keep_background: bool = False,
) -> List[List[float]]:
    """Decide which parts of the timeline survive for ``speaker``.

    pad
        Seconds of lead-in/out recovered around each region, but only into space
        no other speaker occupies -- diarization boundaries tend to clip onsets.
    keep_overlap
        Keep regions where this speaker talks over someone else, instead of
        silencing them.
    keep_background
        Keep non-speech gaps (room tone) instead of replacing them with digital
        silence. Other speakers' speech is still removed.
    """
    if speaker not in timeline.speakers:
        raise KeyError(
            f"unknown speaker {speaker!r}; timeline has: "
            f"{', '.join(timeline.labels) or '(none)'}"
        )

    base = (timeline.speakers[speaker] if keep_overlap
            else timeline.exclusive.get(speaker, []))

    # Never bleed into anyone else's audio: everything spoken by someone other
    # than the target, which for the default path also covers the overlaps.
    others = [label for label in timeline.speakers if label != speaker]
    blocked = iv.subtract(iv.union(*[timeline.speakers[o] for o in others]) if others
                          else [], base)

    keep = iv.pad_into_free_space(base, blocked, pad, 0.0, timeline.duration)

    if keep_background:
        non_speech = iv.subtract([[0.0, timeline.duration]], timeline.speech())
        keep = iv.union(keep, non_speech)
    return keep


def _cosine_ramp(n: int) -> np.ndarray:
    """Half-cosine 0->1 ramp: smoother spectral roll-off than a linear fade."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n, endpoint=False)))


def render(
    master_wav: str,
    timeline: Timeline,
    speaker: str,
    out_path: str,
    pad: float = 0.05,
    fade: float = 0.010,
    keep_overlap: bool = False,
    keep_background: bool = False,
    subtype: Optional[str] = None,
    export_clips: bool = False,
) -> IsolateResult:
    """Write ``out_path`` containing only ``speaker`` from ``master_wav``.

    ``fade`` applies a short half-cosine ramp at each region edge to avoid
    clicks; interior samples are copied bit-exactly. ``subtype`` defaults to the
    master's own subtype so bit depth is preserved end to end. With
    ``export_clips=True`` a second file (``*_clips.wav``) holds the kept
    regions concatenated back to back -- handy for auditioning a speaker.
    """
    data, sample_rate = sf.read(master_wav, always_2d=True)
    n_samples = len(data)
    total_seconds = n_samples / sample_rate

    keep = compute_keep_regions(
        timeline, speaker, pad=pad,
        keep_overlap=keep_overlap, keep_background=keep_background,
    )

    mask = np.zeros(n_samples, dtype=np.float64)
    fade_len = max(0, int(round(fade * sample_rate)))
    concat_pieces: List[np.ndarray] = []
    for start, end in keep:
        a = max(0, int(round(start * sample_rate)))
        b = min(n_samples, int(round(end * sample_rate)))
        if b <= a:
            continue
        mask[a:b] = 1.0
        f = min(fade_len, (b - a) // 2)
        if f > 0:
            ramp = _cosine_ramp(f)
            # min() keeps an already-faded neighbour from being re-raised.
            mask[a:a + f] = np.minimum(mask[a:a + f], ramp)
            mask[b - f:b] = np.minimum(mask[b - f:b], ramp[::-1])

        if export_clips:
            piece = data[a:b].copy()
            if f > 0:
                ramp_col = ramp[:, None]
                piece[:f] *= ramp_col
                piece[-f:] *= ramp_col[::-1]
            concat_pieces.append(piece)

    out = data * mask[:, None]
    if subtype is None:
        info = sf.info(master_wav)
        subtype = info.subtype if info.subtype else "PCM_24"
    if subtype not in sf.available_subtypes("WAV"):
        subtype = "PCM_24"
    sf.write(out_path, out, sample_rate, subtype=subtype)

    clips_path = None
    if export_clips and concat_pieces:
        stem, _ = os.path.splitext(out_path)
        clips_path = f"{stem}_clips.wav"
        sf.write(clips_path, np.concatenate(concat_pieces, axis=0),
                 sample_rate, subtype=subtype)

    return IsolateResult(
        out_path=out_path,
        speaker=speaker,
        kept_regions=keep,
        kept_seconds=iv.total_duration(keep),
        total_seconds=total_seconds,
        clips_path=clips_path,
    )


def verify(master_wav: str, rendered_wav: str, timeline: Timeline,
           speaker: str, fade: float = 0.010) -> dict:
    """Check the render did what it claims.

    Confirms kept audio is bit-identical to the source away from fade ramps, and
    that every other speaker's exclusive speech is digitally silent. Pass the
    same ``fade`` used for the render -- the comparison guard scales with it, so
    a large fade cannot cause a false failure and a zero fade is checked
    bit-exactly to the region edge.
    """
    src, sr = sf.read(master_wav, always_2d=True)
    out, sr2 = sf.read(rendered_wav, always_2d=True)
    report = {
        "sample_rate_match": sr == sr2,
        "length_match": len(src) == len(out),
        "bit_identical_in_kept": None,
        "max_abs_diff_in_kept": None,
        "other_speakers_silent": None,
        "max_leak_from_others": 0.0,
    }
    if not (report["sample_rate_match"] and report["length_match"]):
        return report

    # Guard only what the fade ramp can actually touch (plus a couple of
    # samples of rounding slack); everything further inside a kept region must
    # match the source exactly.
    guard = int(round(fade * sr)) + 2
    worst = 0.0
    compared = 0
    for start, end in timeline.exclusive.get(speaker, []):
        a = int(round(start * sr)) + guard
        b = int(round(end * sr)) - guard
        if b - a <= 0:
            continue
        compared += b - a
        worst = max(worst, float(np.abs(out[a:b] - src[a:b]).max()))
    report["max_abs_diff_in_kept"] = worst
    report["bit_identical_in_kept"] = worst == 0.0
    report["samples_compared"] = compared

    leak = 0.0
    for label, regions in timeline.exclusive.items():
        if label == speaker:
            continue
        for start, end in regions:
            a, b = int(round(start * sr)), int(round(end * sr))
            if b > a:
                leak = max(leak, float(np.abs(out[a:b]).max()))
    report["max_leak_from_others"] = leak
    report["other_speakers_silent"] = leak == 0.0
    return report
