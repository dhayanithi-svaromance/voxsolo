"""Audio extraction, muxing, and NLE/subtitle export helpers (ffmpeg + soundfile)."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import soundfile as sf

# Containers we can mux a lossless PCM track back into.
_PCM_FRIENDLY = {".mov", ".mkv", ".avi", ".wav"}


class FFmpegMissingError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegMissingError(
            "ffmpeg and ffprobe must be installed and on PATH.\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg"
        )


def _run(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd[:6])}...\n{proc.stderr.strip()[-2000:]}"
        )
    return proc.stdout


@dataclass
class MediaInfo:
    path: str
    duration: float
    sample_rate: int
    channels: int
    has_video: bool
    subtype: str  # soundfile subtype to render with (e.g. PCM_16/PCM_24/FLOAT)


def probe(path: str) -> MediaInfo:
    """Read stream metadata, raising if there is no audio track.

    Reports the FIRST audio stream (``0:a:0``) -- the same stream the extract
    functions are pinned to, so what probe describes is what gets processed.
    Video streams that are just attached pictures (album/podcast cover art) do
    not count as video: muxing an isolated track into a JPEG would fail.
    """
    require_ffmpeg()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = _run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path]
    )
    data = json.loads(raw)
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise ValueError(f"no audio stream found in {path}")
    a = audio[0]
    duration = float(
        a.get("duration") or data.get("format", {}).get("duration") or 0.0
    )
    has_video = any(
        s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )

    # Pick the render bit depth. For plain audio files soundfile reports the
    # exact stored subtype; for compressed/container sources decode to 24-bit
    # PCM (comfortably above any lossy codec's real resolution).
    subtype = "PCM_24"
    try:
        info = sf.info(path)
        if info.subtype:
            subtype = info.subtype
    except Exception:
        codec = a.get("codec_name", "")
        if codec in ("pcm_f32le", "pcm_f32be", "pcm_f64le", "pcm_f64be"):
            subtype = "FLOAT"
        elif codec == "pcm_s16le":
            subtype = "PCM_16"

    return MediaInfo(
        path=path,
        duration=duration,
        sample_rate=int(a.get("sample_rate", 0)),
        channels=int(a.get("channels", 0)),
        has_video=has_video,
        subtype=subtype,
    )


def extract_master(path: str, out_wav: str, subtype: str = "PCM_24") -> str:
    """Extract audio losslessly at native rate/layout -- the render source.

    Pinned to stream ``0:a:0`` (the stream ``probe`` reports). Nothing is
    resampled, downmixed, filtered or normalised, so the source's noise floor
    is preserved exactly.
    """
    require_ffmpeg()
    codec = {"FLOAT": "pcm_f32le", "PCM_16": "pcm_s16le"}.get(subtype, "pcm_s24le")
    _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
          "-c:a", codec, out_wav])
    return out_wav


def extract_analysis(path: str, out_wav: str, sample_rate: int = 16000) -> str:
    """Extract 16 kHz mono from stream ``0:a:0`` for the models. Never rendered."""
    require_ffmpeg()
    _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
          "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", out_wav])
    return out_wav


def mux_audio(source_media: str, audio_wav: str, out_path: str,
              audio_codec: Optional[str] = None) -> str:
    """Copy the source's video stream and attach ``audio_wav``.

    Video is stream-copied, never re-encoded. Audio stays lossless PCM where the
    container supports it, otherwise falls back to high-bitrate AAC. Refuses to
    overwrite the source media itself.
    """
    require_ffmpeg()
    if os.path.realpath(out_path) == os.path.realpath(source_media):
        raise ValueError(
            f"refusing to overwrite the source media: {source_media}"
        )
    ext = os.path.splitext(out_path)[1].lower()
    if audio_codec is None:
        audio_codec = "pcm_s24le" if ext in _PCM_FRIENDLY else "aac"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", source_media, "-i", audio_wav,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", audio_codec]
    if audio_codec == "aac":
        cmd += ["-b:a", "320k"]
    cmd.append(out_path)
    try:
        _run(cmd)
    except RuntimeError:
        if audio_codec == "pcm_s24le":  # container refused PCM
            return mux_audio(source_media, audio_wav, out_path, audio_codec="aac")
        raise
    return out_path


class TempDir:
    """Self-cleaning temp directory, or a caller-supplied directory kept as-is."""

    def __init__(self, keep_dir: Optional[str] = None):
        self.keep_dir = keep_dir
        self._tmp: Optional[str] = None

    def __enter__(self) -> str:
        if self.keep_dir:
            os.makedirs(self.keep_dir, exist_ok=True)
            return self.keep_dir
        self._tmp = tempfile.mkdtemp(prefix="voxsolo_")
        return self._tmp

    def __exit__(self, *exc) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)


# --- Filmmaker NLE, DAW, and subtitle exports -------------------------------
#
# Segment dicts carry: segment_id, speaker, start, end, and optionally
# transcript. Timecode conversion decomposes from total units so seconds,
# minutes, and hours always agree with the rounded remainder (no ",1000"
# milliseconds, no frame count equal to fps).


def _sec_to_srt(sec: float) -> str:
    """SRT timestamp HH:MM:SS,mmm with correct rollover."""
    total_ms = int(round(max(0.0, float(sec)) * 1000.0))
    h, total_ms = divmod(total_ms, 3_600_000)
    m, total_ms = divmod(total_ms, 60_000)
    s, ms = divmod(total_ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sec_to_edl(sec: float, fps: float = 24.0) -> str:
    """Non-drop-frame timecode HH:MM:SS:FF with correct frame rollover.

    Non-integer rates (23.976, 29.97) use the standard NDF convention: frames
    are counted at the nominal integer rate, so timecode drifts ~0.1% from
    wall-clock -- exactly how NLEs label those rates in NDF mode.
    """
    frames_per_sec = int(round(fps))
    total_frames = int(round(max(0.0, float(sec)) * fps))
    h, total_frames = divmod(total_frames, frames_per_sec * 3600)
    m, total_frames = divmod(total_frames, frames_per_sec * 60)
    s, frames = divmod(total_frames, frames_per_sec)
    return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"


def _cue_text(seg: dict) -> str:
    text = (seg.get("transcript") or "").strip()
    return f"[{seg['speaker']}] {text}" if text else f"[{seg['speaker']}]"


def export_srt(segments: List[dict], out_path: str) -> str:
    """SRT subtitles with speaker attribution."""
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            f.write(f"{idx}\n{_sec_to_srt(seg['start'])} --> "
                    f"{_sec_to_srt(seg['end'])}\n{_cue_text(seg)}\n\n")
    return out_path


def export_vtt(segments: List[dict], out_path: str) -> str:
    """WebVTT subtitles with ``<v Speaker>`` voice tags."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, 1):
            start = _sec_to_srt(seg["start"]).replace(",", ".")
            end = _sec_to_srt(seg["end"]).replace(",", ".")
            text = (seg.get("transcript") or "").strip() or seg["speaker"]
            f.write(f"{idx}\n{start} --> {end}\n<v {seg['speaker']}>{text}\n\n")
    return out_path


def export_edl(segments: List[dict], out_path: str, fps: float = 24.0) -> str:
    """CMX 3600 Edit Decision List (one event per speaker turn).

    Imports into DaVinci Resolve / Premiere Pro / Final Cut as a cut list;
    source and record timecode are identical because the events describe the
    original timeline itself.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("TITLE: SPEAKER_DIARIZATION_CUTS\nFCM: NON-DROP FRAME\n\n")
        for idx, seg in enumerate(segments, 1):
            s_tc = _sec_to_edl(seg["start"], fps)
            e_tc = _sec_to_edl(seg["end"], fps)
            f.write(f"{idx:03d}  AX       AA/V  C        "
                    f"{s_tc} {e_tc} {s_tc} {e_tc}\n")
            f.write(f"* FROM CLIP NAME: "
                    f"{seg['speaker']}_{seg.get('segment_id', idx)}\n\n")
    return out_path


# Distinct marker colours cycled per speaker (DaVinci's marker palette).
_MARKER_COLORS = ["Cyan", "Purple", "Yellow", "Green", "Red", "Blue",
                  "Pink", "Fuchsia"]


def export_davinci_csv(segments: List[dict], out_path: str,
                       fps: float = 24.0) -> str:
    """Speaker-turn markers as CSV (index, timecode in/out, name, colour).

    One colour per speaker, cycling through DaVinci's marker palette. Note:
    Resolve reads timeline markers from an EDL, not a CSV -- this file is for
    spreadsheets, logging, and scripting; use the EDL for direct NLE import.
    """
    speakers: List[str] = []
    for seg in segments:
        if seg["speaker"] not in speakers:
            speakers.append(seg["speaker"])
    color_of = {spk: _MARKER_COLORS[i % len(_MARKER_COLORS)]
                for i, spk in enumerate(speakers)}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Start", "End", "Duration", "Name",
                         "Description", "Color"])
        for idx, seg in enumerate(segments, 1):
            dur = seg["end"] - seg["start"]
            writer.writerow([
                idx,
                _sec_to_edl(seg["start"], fps),
                _sec_to_edl(seg["end"], fps),
                _sec_to_edl(dur, fps),
                seg["speaker"],
                (seg.get("transcript") or "").strip(),
                color_of[seg["speaker"]],
            ])
    return out_path


def export_audacity_labels(segments: List[dict], out_path: str) -> str:
    """Audacity label track (tab-separated ``start\\tend\\tlabel``).

    Import via Audacity: File > Import > Labels. The de-facto exchange format
    for podcast/DAW workflows.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("transcript") or "").strip()
            label = f"{seg['speaker']}: {text}" if text else seg["speaker"]
            f.write(f"{seg['start']:.6f}\t{seg['end']:.6f}\t{label}\n")
    return out_path
