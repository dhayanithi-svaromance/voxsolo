# voxsolo 🎚️

> **The solo button for any voice in a recording.**
> Diarize → keep one speaker bit-exact, mute everyone else (overlaps included) → NLE/DAW timeline exports → interactive HTML review player.

On a mixing console, *solo* plays one channel and silences the rest. `voxsolo` does that to people: pick a speaker, get a full-length stem where only they exist — every kept sample copied verbatim from your source.

**🎧 Hear it now — no install:** [dhayanithi-svaromance.github.io/voxsolo](https://dhayanithi-svaromance.github.io/voxsolo/) — a scene from *His Girl Friday* (1940, public domain, famous for its overlapping rapid-fire dialogue), soloed by this tool. Switch tracks to hear each actor alone; 12.9s of cross-talk silenced. The page itself is voxsolo's own HTML output.

[![PyPI](https://img.shields.io/pypi/v/voxsolo)](https://pypi.org/project/voxsolo/)
[![CI](https://github.com/dhayanithi-svaromance/voxsolo/actions/workflows/test.yml/badge.svg)](https://github.com/dhayanithi-svaromance/voxsolo/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

No established open-source tool does the middle step. Source-separation models (Demucs, SepFormer, ClearVoice) *resynthesize* audio — fine for karaoke, wrong for film dialogue where you must keep the original noise floor, room tone, and every bit of the recording exactly as shot. `voxsolo` instead **copies the target speaker's samples verbatim** and silences everyone else, including every region where two people talk at once. What you keep is bit-identical to the source — and the tool proves it.

```
┌─────────────┐   ┌──────────────────┐   ┌───────────────────────────────┐
│ any media   │ → │ pyannote         │ → │ per-speaker stems (verbatim)  │
│ .mov .wav … │   │ diarization      │   │ + overlaps silenced            │
└─────────────┘   └──────────────────┘   │ + SRT/VTT/EDL/CSV/labels      │
                                         │ + HTML review player          │
                                         │ + video with isolated audio   │
                                         └───────────────────────────────┘
```

## Why this exists

Diarization tools (pyannote, NeMo, WhisperX) output *labels* — RTTM files and timestamps. Editors need *audio* and *timelines*. The gap between them is real: pyannote's own issue tracker tells people asking for per-speaker audio to script it themselves, and the only repos that do exactly this are single-file utilities. `voxsolo` closes that gap properly:

- **Verbatim isolation.** Kept samples are copied untouched — no denoising, no normalization, no resampling, no gain change. Bit depth is preserved end to end (PCM_16/PCM_24/FLOAT in → same out).
- **Overlap handling that is actually correct.** Overlap is computed with an exact sweep-line algorithm over *N* speakers — nested overlaps, triple-talk, self-overlapping turns, touching boundaries. Simultaneous speech is silenced by default (`--keep-overlap` to keep it).
- **Provable output.** `--verify` asserts, sample by sample, that kept audio is bit-identical to the source and other speakers are digitally silent — and fails the exit code if not.
- **Click-free cuts.** Half-cosine micro-fades (default 10 ms) at region edges only; `--fade 0` gives bit-exact edges. Pre/post-roll handles (`--pad`) recover word onsets without ever bleeding into another speaker's turn.
- **Filmmaker handoff.** CMX 3600 EDL, SRT/WebVTT subtitles, marker CSV, Audacity labels, RTTM — plus the isolated audio muxed back into your video with the video stream **copied, not re-encoded**.
- **Visual review.** A self-contained HTML dashboard: color-coded speaker timeline, click-to-seek, live active-speaker highlight, and a track switcher between the full mix and each isolated stem.

## Install

```bash
pip install voxsolo              # or: pip install "voxsolo[transcribe]" for subtitles with text
# ffmpeg must be on PATH:  sudo apt install ffmpeg  /  brew install ffmpeg
```

<details><summary>Install from source instead</summary>

```bash
git clone https://github.com/dhayanithi-svaromance/voxsolo
cd voxsolo && pip install .
```
</details>

**Models.** The default engine is pyannote `speaker-diarization-community-1` (the current standard, best accuracy), falling back to `speaker-diarization-3.1`. Both are gated on Hugging Face — a free, one-time terms acceptance:

1. Accept conditions at [hf.co/pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) (and/or [3.1](https://hf.co/pyannote/speaker-diarization-3.1) + [segmentation-3.0](https://hf.co/pyannote/segmentation-3.0))
2. `huggingface-cli login` (or `export HF_TOKEN=...`)

**No token?** `--allow-mirrors` rebuilds the 3.1 recipe from community re-uploads of the weights, pinned by SHA-256 so a tampered mirror is rejected, not trusted. It works fully offline once cached. This is opt-in by design — you're choosing to trust mirror repos instead of the official gated ones.

## Use

```bash
# 1. See who speaks when (+ all timeline exports)
voxsolo diarize film_scene.mov

# 2. Isolate one speaker — everyone else and all cross-talk silenced
voxsolo isolate film_scene.mov --speaker SPEAKER_01 --verify

# 3. Everything: stems for all speakers, subtitles, EDL, HTML player
voxsolo full film_scene.mov -o scene_output/

# Diarize once, render many times (no second model run)
voxsolo diarize film_scene.mov --save-timeline tl.json
voxsolo isolate film_scene.mov --timeline tl.json --all --clips
```

The flags that matter most:

| Flag | Why you'd use it |
|---|---|
| `--num-speakers N` | **Pass it whenever you know the count.** On noisy material, auto-detection can collapse two similar voices into one; pinning N prevents it. |
| `--verify` | Proof, not vibes: bit-identical check + leakage check, exit 3 on failure. |
| `--keep-overlap` | Keep the target's overlapped speech instead of silencing it. |
| `--keep-background` | Keep room tone between turns instead of digital silence. |
| `--fps 25` | Timecode rate for EDL/marker exports. |
| `--transcribe` / `--no-transcribe` | Faster-Whisper text in SRT/VTT/labels (`pip install ".[transcribe]"`). |
| `--fade 0` | Bit-exact region edges (accepting possible clicks). |

Outputs land next to your input (or in `-o DIR`): `<name>_<SPEAKER>_timeline.wav` stems aligned to the original timeline, `<name>_<SPEAKER>_only.mov` video with isolated audio, `<name>.srt/.vtt/.edl`, `<name>_markers.csv`, `<name>_labels.txt` (Audacity), `<name>.rttm`, `<name>_timeline.json`, `<name>_player.html`.

## Measured, not promised

Real clips, real numbers (RTX 3050 Laptop 4 GB unless noted):

| Clip | Length | Overlap silenced | Stems | Diarization time | `--verify` |
|---|---|---|---|---|---|
| *His Girl Friday* (1940) office scene — [hear it](https://dhayanithi-svaromance.github.io/voxsolo/) | 75 s | 12.9 s in 15 regions | 30.0 s + 29.9 s | 11.0 s (~7× realtime) | ✅ bit-identical |
| 1980s Tamil film scene, 44.1 kHz stereo, heavy tape hiss | 68 s | 6.4 s in 4 regions | 26.2 s + 18.3 s | — | ✅ bit-identical |

What `--verify` prints — for every stem, every render:

```
SPEAKER_00: kept 30.02s of 75.00s (40.0%) in 15 region(s)
  verify: bit-identical=True others-silent=True length=True [OK]
```

`bit-identical` compares every kept sample against the source; `others-silent` asserts digital zero wherever any other speaker talks. If either fails, the exit code fails. That's the whole fidelity claim, and it's machine-checked on your file, not ours.

## How it compares

| | voxsolo | pyannote.audio | WhisperX | Auto-Editor | ClearVoice / Demucs | AudioShake (commercial) |
|---|---|---|---|---|---|---|
| Diarization | ✅ (pyannote engine) | ✅ | ✅ (via pyannote) | ❌ (loudness only) | ❌ | ✅ |
| Per-speaker **audio** out | ✅ verbatim | ❌ labels only | ❌ labels only | ❌ | ✅ resynthesized | ✅ resynthesized |
| Original noise/fidelity kept | ✅ bit-identical, proven | — | — | ✅ | ❌ neural output | ❌ neural output |
| Overlap silencing (N-speaker exact) | ✅ | detects, doesn't render | ❌ | ❌ | — | — |
| NLE/DAW exports | EDL, SRT, VTT, CSV, Audacity, RTTM | RTTM | SRT, VTT, JSON, labels | FCP7 XML, FCPXML, MLT | ❌ | — |
| Video mux (stream-copy) | ✅ | ❌ | ❌ | ✅ | ❌ | — |
| Visual review UI | ✅ HTML player | ❌ | ❌ | ❌ | ❌ | ✅ |

`voxsolo` does **not** separate mixed voices: silencing an overlap removes the target's words there too. That is the intended trade — verbatim fidelity over reconstruction. When you need the words back from a mix, use a source-separation tool (ClearVoice, TIGER, Bandit-v2) and accept resynthesized audio.

## Honest limitations

Everything downstream depends on diarization quality, and **diarization is content-dependent**. Expect degradation with heavy noise/music/reverb, band-limited or archival sources, similar-sounding voices, many speakers, or sub-second turns. Concretely, on a noisy 1980s film clip during development: auto-detect merged two speakers into one; `--num-speakers 2` produced the correct split. The interval algebra and rendering are exact and tested — the model in front of them is not magic. Run `diarize` first, sanity-check the turn structure, listen to the result, and score against a reference (`--rttm` + `pyannote.metrics`) when you need a number.

Boundaries are also approximate: `--pad` (default 50 ms) recovers clipped word onsets, but only into space no other speaker occupies.

## Library API

```python
from voxsolo import load_pipeline, diarize, render, verify

loaded = load_pipeline(allow_mirrors=True)          # or token="hf_..."
timeline = diarize("analysis_16k.wav", duration=68.0, pipeline=loaded.pipeline,
                   num_speakers=2)
result = render("master.wav", timeline, "SPEAKER_01", "out.wav")
report = verify("master.wav", "out.wav", timeline, "SPEAKER_01")
assert report["bit_identical_in_kept"] and report["other_speakers_silent"]
```

`Timeline` serializes to JSON and RTTM; the interval algebra (`voxsolo.intervals`) is dependency-free and fully unit-tested (sweep-line overlap for any N, exact subtraction, contiguity-safe padding).

## Project layout

```
voxsolo/
  intervals.py    exact interval algebra (merge/subtract/overlap-N) — fully tested
  pipeline.py     pyannote loading: community-1 → 3.1 → SHA-256-pinned mirrors
  diarize.py      Timeline model, JSON round-trip, RTTM export
  isolate.py      keep-region computation, verbatim rendering, verification
  audio.py        ffmpeg extraction/muxing + EDL/SRT/VTT/CSV/Audacity exports
  transcribe.py   optional Faster-Whisper per-turn transcription
  html_player.py  self-contained review dashboard
  cli.py          diarize / isolate / full / player commands
tests/            30 tests, model-free, run in CI
```

## License

MIT (see [LICENSE](LICENSE)). The pyannote **models** carry their own terms — accept them on their Hugging Face pages before commercial use; `pyannote.audio` itself is MIT, `speaker-diarization-community-1` weights are CC-BY-4.0.

---

If voxsolo saved you a re-record, a paid tool, or an afternoon of manual muting — [a star](https://github.com/dhayanithi-svaromance/voxsolo) helps the next editor find it.
