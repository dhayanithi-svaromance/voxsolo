"""Tests for timecode conversion and NLE/subtitle export formats."""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxsolo.audio import (  # noqa: E402
    _sec_to_edl,
    _sec_to_srt,
    export_audacity_labels,
    export_davinci_csv,
    export_edl,
    export_srt,
    export_vtt,
)
from voxsolo.diarize import Segment, build_timeline  # noqa: E402

SEGS = [
    {"segment_id": "SEG_001", "speaker": "SPEAKER_00", "start": 0.0,
     "end": 6.04, "duration": 6.04, "transcript": "hello there"},
    {"segment_id": "SEG_002", "speaker": "SPEAKER_01", "start": 6.5,
     "end": 9.999, "duration": 3.499, "transcript": ""},
]


def test_srt_timecode_rollover():
    # the classic bugs: ms rounding to 1000, seconds not carrying
    assert _sec_to_srt(1.9996) == "00:00:02,000"
    assert _sec_to_srt(59.9996) == "00:01:00,000"
    assert _sec_to_srt(3599.9995) == "01:00:00,000"
    assert _sec_to_srt(0.0) == "00:00:00,000"


def test_edl_frame_rollover():
    # frames must never equal fps
    assert _sec_to_edl(0.999, 24) == "00:00:01:00"
    assert _sec_to_edl(10.9584, 24) == "00:00:10:23"
    assert _sec_to_edl(59.999, 25) == "00:01:00:00"
    for sec in [0.0, 0.5, 1.0, 59.99, 3600.02, 7261.7]:
        for fps in [23.976, 24, 25, 29.97, 30]:
            frames = int(_sec_to_edl(sec, fps).rsplit(":", 1)[1])
            assert frames < int(round(fps))


def _tmp(name):
    return os.path.join(tempfile.mkdtemp(prefix="voxsolo_test_"), name)


def test_srt_structure():
    path = export_srt(SEGS, _tmp("t.srt"))
    blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    assert len(blocks) == 2
    for i, block in enumerate(blocks, 1):
        lines = block.split("\n")
        assert lines[0] == str(i)
        assert re.match(r"^\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d$",
                        lines[1])
    assert "[SPEAKER_00] hello there" in blocks[0]
    assert "[SPEAKER_01]" in blocks[1]  # empty transcript -> label only


def test_vtt_structure():
    path = export_vtt(SEGS, _tmp("t.vtt"))
    content = open(path, encoding="utf-8").read()
    assert content.startswith("WEBVTT\n")
    assert "<v SPEAKER_00>hello there" in content


def test_edl_structure():
    path = export_edl(SEGS, _tmp("t.edl"), fps=24.0)
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0].startswith("TITLE:")
    assert lines[1] == "FCM: NON-DROP FRAME"
    events = [ln for ln in lines if re.match(r"^\d{3}  ", ln)]
    assert len(events) == 2
    tc = r"\d\d:\d\d:\d\d:\d\d"
    for ev in events:
        assert re.match(rf"^\d{{3}}  AX\s+AA/V  C\s+{tc} {tc} {tc} {tc}$", ev)
    assert any("FROM CLIP NAME: SPEAKER_00_SEG_001" in ln for ln in lines)


def test_marker_csv_distinct_colors():
    path = export_davinci_csv(SEGS, _tmp("t.csv"), fps=24.0)
    rows = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(rows) == 3  # header + 2
    colors = {row.rsplit(",", 1)[1] for row in rows[1:]}
    assert len(colors) == 2, "each speaker must get its own marker colour"


def test_audacity_labels():
    path = export_audacity_labels(SEGS, _tmp("t.txt"))
    for line in open(path, encoding="utf-8").read().splitlines():
        parts = line.split("\t")
        assert len(parts) == 3
        assert float(parts[1]) > float(parts[0])


def test_rttm_uri_sanitized():
    tl = build_timeline([Segment(0.0, 1.0, "A")], duration=2.0)
    path = tl.write_rttm(_tmp("t.rttm"), uri="my file (1).mov")
    for line in open(path).read().splitlines():
        fields = line.split()
        assert len(fields) == 10, f"RTTM line must have 10 fields: {line}"
        assert fields[1] == "my_file_(1).mov"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
