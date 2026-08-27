"""Command-line interface.

    python -m voxsolo diarize  INPUT [-o DIR]        # who speaks when + NLE exports
    python -m voxsolo isolate  INPUT --speaker S     # render isolated speaker stems
    python -m voxsolo isolate  INPUT --all --verify
    python -m voxsolo full     INPUT [-o DIR]        # everything + HTML player
    python -m voxsolo player   INPUT --timeline T    # rebuild the HTML player only

Exit codes: 0 success; 1 error; 2 bad usage (unknown speaker, conflicting
flags); 3 verification failed; 4 gated model not accessible; 5 ffmpeg missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import soundfile as sf

# Import names directly: the package __init__ re-exports a `diarize` function,
# which shadows the `diarize` submodule under `from . import diarize`.
from .audio import (
    FFmpegMissingError,
    MediaInfo,
    TempDir,
    export_audacity_labels,
    export_davinci_csv,
    export_edl,
    export_srt,
    export_vtt,
    extract_analysis,
    extract_master,
    mux_audio,
    probe,
)
from .diarize import Timeline, diarize
from .isolate import render, verify
from .pipeline import DEFAULT_PIPELINE, GatedModelError, load_pipeline

_VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".avi", ".webm"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="audio or video file")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="directory for generated files "
                             "(default: alongside the input)")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                        help=f"pyannote pipeline (default: {DEFAULT_PIPELINE}, "
                             "falling back to speaker-diarization-3.1)")
    parser.add_argument("--token", default=None,
                        help="HF token (else $HF_TOKEN or the huggingface-cli "
                             "login)")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: cuda when available)")
    parser.add_argument("--allow-mirrors", action="store_true",
                        help="if the gated pipelines are unavailable, rebuild "
                             "the 3.1 recipe from checksum-pinned community "
                             "mirrors (no HF token needed)")
    parser.add_argument("--trust-mirrors", action="store_true",
                        help="with --allow-mirrors: skip the SHA-256 pin check "
                             "on mirror weights (not recommended)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="exact speaker count; omit to auto-detect (pass "
                             "it whenever you know it -- it prevents cluster "
                             "collapse on noisy audio)")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--timeline", default=None,
                        help="reuse a previously saved timeline JSON "
                             "(skips the model entirely)")
    parser.add_argument("--save-timeline", default=None,
                        help="write the timeline JSON here (default: "
                             "<out-dir>/<name>_timeline.json)")
    parser.add_argument("--rttm", default=None,
                        help="write RTTM here (default: <out-dir>/<name>.rttm)")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="frame rate for EDL / timecode exports "
                             "(default 24)")
    parser.add_argument("--workdir", default=None,
                        help="keep intermediate files here instead of a "
                             "self-cleaning temp dir")
    parser.add_argument("--quiet", action="store_true")


def _add_transcribe(parser: argparse.ArgumentParser,
                    on_by_default: bool) -> None:
    if on_by_default:
        parser.add_argument("--no-transcribe", action="store_true",
                            help="skip Faster-Whisper transcription")
    else:
        parser.add_argument("--transcribe", action="store_true",
                            help="transcribe speaker turns with Faster-Whisper")
    parser.add_argument("--whisper-model", default="base",
                        help="tiny | base | small | medium | large-v3 "
                             "(default: base)")
    parser.add_argument("--language", default=None,
                        help="language code (e.g. 'ta', 'en'); auto-detected "
                             "if omitted")


def _add_render(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pad", type=float, default=0.05,
                        help="lead-in/out recovered around each region, only "
                             "into free space (seconds, default 0.05)")
    parser.add_argument("--fade", type=float, default=0.010,
                        help="half-cosine edge fade to prevent clicks "
                             "(seconds, default 0.010; 0 = bit-exact edges)")
    parser.add_argument("--keep-overlap", action="store_true",
                        help="keep this speaker's overlapped speech instead "
                             "of silencing it")
    parser.add_argument("--keep-background", action="store_true",
                        help="keep non-speech room tone instead of digital "
                             "silence")
    parser.add_argument("--clips", action="store_true",
                        help="also write <stem>_clips.wav with the kept "
                             "regions concatenated")
    parser.add_argument("--no-video", action="store_true",
                        help="skip muxing isolated audio back into the video")
    parser.add_argument("--verify", action="store_true",
                        help="assert kept audio is bit-identical and other "
                             "speakers are digitally silent (exit 3 on "
                             "failure)")


def _out_dir_and_base(args) -> tuple:
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    return out_dir, base


def _get_timeline(args, media: MediaInfo, workdir: str) -> Timeline:
    if args.timeline:
        if not args.quiet:
            print(f"[1/2] loading timeline from {args.timeline}")
        timeline = Timeline.load(args.timeline)
        if media.duration and abs(timeline.duration - media.duration) > 0.5:
            print(f"warning: timeline duration ({timeline.duration:.2f}s) "
                  f"does not match this file ({media.duration:.2f}s) -- was it "
                  f"saved from a different recording?", file=sys.stderr)
        return timeline

    analysis = extract_analysis(
        args.input, os.path.join(workdir, "analysis_16k.wav")
    )
    if not args.quiet:
        print(f"[1/2] loading pipeline ({args.pipeline}) ...")
    loaded = load_pipeline(
        pipeline_name=args.pipeline, token=args.token, device=args.device,
        allow_mirrors=args.allow_mirrors or args.trust_mirrors,
        verify_checksum=not args.trust_mirrors,
    )
    if loaded.used_mirrors and not args.quiet:
        print(f"      NOTE: using mirrored weights -> {loaded.source}")
    if not args.quiet:
        print("      diarizing ...")
    return diarize(
        analysis, duration=media.duration, pipeline=loaded.pipeline,
        num_speakers=args.num_speakers, min_speakers=args.min_speakers,
        max_speakers=args.max_speakers, source=loaded.source,
    )


def _segment_dicts(timeline: Timeline) -> List[dict]:
    return [
        {"segment_id": f"SEG_{i + 1:03d}", "speaker": s.speaker,
         "start": s.start, "end": s.end,
         "duration": round(s.end - s.start, 3), "transcript": ""}
        for i, s in enumerate(timeline.segments)
    ]


def _transcribe(args, timeline: Timeline, workdir: str,
                quiet: bool) -> List[dict]:
    from .transcribe import SpeechTranscriber, TranscriptionUnavailableError

    segments = _segment_dicts(timeline)
    try:
        transcriber = SpeechTranscriber(model_size=args.whisper_model,
                                        language=args.language,
                                        device=args.device)
    except TranscriptionUnavailableError as exc:
        print(f"warning: {exc} -- continuing without transcripts",
              file=sys.stderr)
        return segments
    analysis = os.path.join(workdir, "analysis_16k.wav")
    if not os.path.exists(analysis):
        extract_analysis(args.input, analysis)
    wav_16k, _ = sf.read(analysis)
    if not quiet:
        print(f"      transcribing {len(segments)} turns "
              f"({args.whisper_model}) ...")
    return transcriber.transcribe_segments(wav_16k, segments)


def _emit_exports(args, timeline: Timeline, out_dir: str, base: str,
                  segments: Optional[List[dict]] = None) -> None:
    segments = segments or _segment_dicts(timeline)

    json_path = args.save_timeline or os.path.join(out_dir,
                                                   f"{base}_timeline.json")
    timeline.save(json_path)
    rttm_path = args.rttm or os.path.join(out_dir, f"{base}.rttm")
    timeline.write_rttm(rttm_path, uri=base)

    written = {
        "timeline JSON": json_path,
        "RTTM": rttm_path,
        "SRT subtitles": export_srt(segments,
                                    os.path.join(out_dir, f"{base}.srt")),
        "WebVTT": export_vtt(segments, os.path.join(out_dir, f"{base}.vtt")),
        "CMX 3600 EDL": export_edl(segments,
                                   os.path.join(out_dir, f"{base}.edl"),
                                   fps=args.fps),
        "marker CSV": export_davinci_csv(
            segments, os.path.join(out_dir, f"{base}_markers.csv"),
            fps=args.fps),
        "Audacity labels": export_audacity_labels(
            segments, os.path.join(out_dir, f"{base}_labels.txt")),
    }
    if not args.quiet:
        for kind, path in written.items():
            print(f"      {kind:15s} -> {path}")


def _print_timeline(timeline: Timeline) -> None:
    print("\n" + "=" * 60)
    print(timeline.summary())
    if timeline.overlap:
        print("\noverlapped speech regions (>=2 speakers):")
        for start, end in timeline.overlap:
            print(f"  {start:8.3f} -> {end:8.3f}  ({end - start:.2f}s)")
    print("=" * 60)


def _render_speakers(args, media: MediaInfo, timeline: Timeline,
                     targets: List[str], out_dir: str, base: str,
                     workdir: str) -> Dict[str, str]:
    """Render stems for ``targets``; returns {speaker: wav_path}. Exits via
    SystemExit(3) on a failed --verify."""
    master = extract_master(args.input,
                            os.path.join(workdir, "master.wav"),
                            subtype=media.subtype)
    input_real = os.path.realpath(args.input)
    stems: Dict[str, str] = {}
    if not args.quiet:
        print(f"[2/2] rendering ({media.subtype}, {media.sample_rate} Hz, "
              f"{media.channels} ch) ...")
    for spk in targets:
        out_wav = os.path.join(out_dir, f"{base}_{spk}_timeline.wav")
        if os.path.realpath(out_wav) == input_real:
            raise ValueError(
                f"output would overwrite the input file: {out_wav}")
        result = render(
            master, timeline, spk, out_wav,
            pad=args.pad, fade=args.fade,
            keep_overlap=args.keep_overlap,
            keep_background=args.keep_background,
            subtype=media.subtype,
            export_clips=args.clips,
        )
        stems[spk] = out_wav
        print(f"      {result.summary()}")
        if result.clips_path and not args.quiet:
            print(f"        clips  -> {result.clips_path}")

        if args.verify:
            report = verify(master, out_wav, timeline, spk, fade=args.fade)
            ok = (report["bit_identical_in_kept"] and
                  report["other_speakers_silent"] and
                  report["length_match"])
            print(f"        verify: "
                  f"bit-identical={report['bit_identical_in_kept']} "
                  f"others-silent={report['other_speakers_silent']} "
                  f"length={report['length_match']} "
                  f"{'[OK]' if ok else '[FAILED]'}")
            if not ok:
                raise SystemExit(3)

        if media.has_video and not args.no_video:
            ext = os.path.splitext(args.input)[1]
            if ext.lower() in _VIDEO_EXTS:
                out_video = os.path.join(out_dir, f"{base}_{spk}_only{ext}")
                if os.path.realpath(out_video) == input_real:
                    raise ValueError(
                        f"output would overwrite the input file: {out_video}")
                try:
                    mux_audio(args.input, out_wav, out_video)
                    if not args.quiet:
                        print(f"        video  -> {out_video}")
                except RuntimeError as exc:
                    print(f"warning: video mux failed for {spk}: {exc}",
                          file=sys.stderr)
    return stems


def _resolve_targets(args, timeline: Timeline) -> List[str]:
    if args.all:
        if not timeline.labels:
            print("error: no speakers detected in this file "
                  "(silence or music only?)", file=sys.stderr)
            raise SystemExit(2)
        return timeline.labels
    if args.speaker not in timeline.speakers:
        print(f"error: no speaker {args.speaker!r} in this file.\n"
              f"available: {', '.join(timeline.labels) or '(none)'}",
              file=sys.stderr)
        raise SystemExit(2)
    return [args.speaker]


def _player_sources(out_dir: str, input_path: str,
                    stems: Dict[str, str]) -> Dict[str, str]:
    """Track hrefs relative to the HTML file (which lives in ``out_dir``)."""
    sources = {"full": os.path.relpath(os.path.abspath(input_path), out_dir)}
    for spk, wav in stems.items():
        sources[spk] = os.path.relpath(os.path.abspath(wav), out_dir)
    return sources


def cmd_diarize(args) -> int:
    media = probe(args.input)
    out_dir, base = _out_dir_and_base(args)
    with TempDir(args.workdir) as workdir:
        timeline = _get_timeline(args, media, workdir)
        _print_timeline(timeline)
        segments = None
        if args.transcribe:
            segments = _transcribe(args, timeline, workdir, args.quiet)
        _emit_exports(args, timeline, out_dir, base, segments)
    return 0


def cmd_isolate(args) -> int:
    media = probe(args.input)
    out_dir, base = _out_dir_and_base(args)
    with TempDir(args.workdir) as workdir:
        timeline = _get_timeline(args, media, workdir)
        if not args.quiet:
            _print_timeline(timeline)
        targets = _resolve_targets(args, timeline)
        _render_speakers(args, media, timeline, targets, out_dir, base,
                         workdir)
        if args.save_timeline:
            timeline.save(args.save_timeline)
        if args.rttm:
            timeline.write_rttm(args.rttm, uri=base)
    return 0


def cmd_full(args) -> int:
    from .html_player import generate_html_player

    media = probe(args.input)
    out_dir, base = _out_dir_and_base(args)
    with TempDir(args.workdir) as workdir:
        timeline = _get_timeline(args, media, workdir)
        _print_timeline(timeline)

        segments = None
        if not args.no_transcribe:
            segments = _transcribe(args, timeline, workdir, args.quiet)

        args.all, args.speaker = True, None  # full always renders everyone
        targets = _resolve_targets(args, timeline)
        stems = _render_speakers(args, media, timeline, targets, out_dir,
                                 base, workdir)

        _emit_exports(args, timeline, out_dir, base, segments)

        html_path = os.path.join(out_dir, f"{base}_player.html")
        generate_html_player(
            os.path.basename(args.input), media.duration,
            segments or _segment_dicts(timeline), html_path,
            _player_sources(out_dir, args.input, stems),
        )
        if not args.quiet:
            print(f"      HTML player     -> {html_path}")
    return 0


def cmd_player(args) -> int:
    """Rebuild the HTML player from a saved timeline -- no models involved."""
    from .html_player import generate_html_player

    if not args.timeline:
        print("error: player requires --timeline TIMELINE_JSON",
              file=sys.stderr)
        return 2
    media = probe(args.input)
    out_dir, base = _out_dir_and_base(args)
    timeline = Timeline.load(args.timeline)

    # Pick up any stems a previous isolate/full run left in the output dir.
    stems = {}
    for spk in timeline.labels:
        wav = os.path.join(out_dir, f"{base}_{spk}_timeline.wav")
        if os.path.exists(wav):
            stems[spk] = wav

    html_path = os.path.join(out_dir, f"{base}_player.html")
    generate_html_player(
        os.path.basename(args.input), media.duration,
        _segment_dicts(timeline), html_path,
        _player_sources(out_dir, args.input, stems),
    )
    print(f"HTML player -> {html_path}"
          + (f" ({len(stems)} solo track(s))" if stems else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxsolo",
        description="Speaker diarization, verbatim speaker isolation with "
                    "overlap silencing, and NLE/DAW timeline exports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diar = sub.add_parser("diarize",
                            help="who speaks when + timeline exports "
                                 "(RTTM/SRT/VTT/EDL/CSV/labels)")
    _add_common(p_diar)
    _add_transcribe(p_diar, on_by_default=False)
    p_diar.set_defaults(func=cmd_diarize)

    p_iso = sub.add_parser("isolate",
                           help="render isolated speaker stems, silencing "
                                "everyone else and all overlaps")
    _add_common(p_iso)
    group = p_iso.add_mutually_exclusive_group(required=True)
    group.add_argument("--speaker", help="label to keep, e.g. SPEAKER_01")
    group.add_argument("--all", action="store_true",
                       help="render one stem per detected speaker")
    _add_render(p_iso)
    p_iso.set_defaults(func=cmd_isolate)

    p_full = sub.add_parser("full",
                            help="diarize + transcribe + isolate everyone + "
                                 "exports + HTML player")
    _add_common(p_full)
    _add_transcribe(p_full, on_by_default=True)
    _add_render(p_full)
    p_full.set_defaults(func=cmd_full)

    p_play = sub.add_parser("player",
                            help="rebuild the HTML player from a saved "
                                 "timeline (no model run)")
    _add_common(p_play)
    p_play.set_defaults(func=cmd_player)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except GatedModelError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 4
    except FFmpegMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
