"""Render/verify roundtrip on synthetic audio -- guards the core guarantees:
kept audio bit-identical to the source, other speakers digitally silent."""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxsolo.diarize import Segment, build_timeline  # noqa: E402
from voxsolo.isolate import compute_keep_regions, render, verify  # noqa: E402

SR = 16000
DUR = 12.0


def _make_master(tmp, subtype="PCM_24"):
    rng = np.random.default_rng(seed=7)
    data = (rng.standard_normal((int(DUR * SR), 2)) * 0.1).astype(np.float64)
    path = os.path.join(tmp, "master.wav")
    sf.write(path, data, SR, subtype=subtype)
    return path


def _timeline():
    # A [1,4] and [8,10]; B [3.5,6]; overlap [3.5,4]
    return build_timeline(
        [Segment(1.0, 4.0, "A"), Segment(8.0, 10.0, "B"),
         Segment(3.5, 6.0, "B")],
        duration=DUR,
    )


def test_render_verify_roundtrip_with_fade():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "A.wav")
    result = render(master, tl, "A", out, pad=0.05, fade=0.010)
    assert result.kept_seconds > 0
    report = verify(master, out, tl, "A", fade=0.010)
    assert report["length_match"]
    assert report["bit_identical_in_kept"], report
    assert report["other_speakers_silent"], report
    assert report["samples_compared"] > 0


def test_render_verify_roundtrip_fade_zero():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "A0.wav")
    render(master, tl, "A", out, pad=0.05, fade=0.0)
    report = verify(master, out, tl, "A", fade=0.0)
    assert report["bit_identical_in_kept"], report
    assert report["other_speakers_silent"], report


def test_large_fade_does_not_false_fail_verify():
    # the historical bug: fade > the old fixed 50ms guard caused a false FAIL
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "Abig.wav")
    render(master, tl, "A", out, pad=0.0, fade=0.2)
    report = verify(master, out, tl, "A", fade=0.2)
    assert report["bit_identical_in_kept"], report


def test_verify_catches_corruption():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "Abad.wav")
    render(master, tl, "A", out, pad=0.05, fade=0.010)
    data, sr = sf.read(out, always_2d=True)
    data[int(2.0 * SR)] *= 0.5  # corrupt one sample well inside kept region
    sf.write(out, data, sr, subtype="PCM_24")
    report = verify(master, out, tl, "A", fade=0.010)
    assert not report["bit_identical_in_kept"]


def test_verify_catches_leakage():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "Aleak.wav")
    render(master, tl, "A", out, pad=0.05, fade=0.010)
    data, sr = sf.read(out, always_2d=True)
    data[int(5.0 * SR)] = 0.5  # inject audio inside B's exclusive region
    sf.write(out, data, sr, subtype="PCM_24")
    report = verify(master, out, tl, "A", fade=0.010)
    assert not report["other_speakers_silent"]


def test_overlap_is_silenced_by_default():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "Aov.wav")
    render(master, tl, "A", out, pad=0.05, fade=0.010)
    data, _ = sf.read(out, always_2d=True)
    ov = data[int(3.6 * SR):int(3.9 * SR)]  # inside the 3.5-4.0 overlap
    assert np.abs(ov).max() == 0.0


def test_subtype_preserved():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp, subtype="FLOAT")
    tl = _timeline()
    out = os.path.join(tmp, "Afl.wav")
    render(master, tl, "A", out)
    assert sf.info(out).subtype == "FLOAT"


def test_clips_export():
    tmp = tempfile.mkdtemp(prefix="voxsolo_iso_")
    master = _make_master(tmp)
    tl = _timeline()
    out = os.path.join(tmp, "Ac.wav")
    result = render(master, tl, "A", out, export_clips=True)
    assert result.clips_path and os.path.exists(result.clips_path)
    clip_len = len(sf.read(result.clips_path, always_2d=True)[0]) / SR
    assert abs(clip_len - result.kept_seconds) < 0.01


def test_keep_regions_never_touch_other_speakers():
    tl = _timeline()
    keep = compute_keep_regions(tl, "A", pad=0.5)
    for start, end in keep:
        for b_start, b_end in tl.exclusive["B"]:
            assert end <= b_start + 1e-9 or start >= b_end - 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
