"""Optional Faster-Whisper transcription of diarized speaker turns.

Transcribing turn by turn (rather than the whole file) keeps each speaker's
text attributed to them and avoids hallucination loops on noisy inter-speech
gaps. faster-whisper is an optional dependency: install with
``pip install voxsolo[transcribe]``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class TranscriptionUnavailableError(RuntimeError):
    pass


def _load_model(model_size: str, device: Optional[str]):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionUnavailableError(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        ) from exc

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # faster-whisper/ctranslate2 rejects "cuda:0"-style strings: it takes the
    # device kind plus a separate integer index.
    device_index = 0
    if ":" in device:
        device, idx = device.split(":", 1)
        device_index = int(idx)

    if device == "cpu":
        return WhisperModel(model_size, device="cpu", compute_type="int8")

    # Larger models do not fit in fp16 on small GPUs (e.g. large-v3 on 4 GB);
    # step down through quantised variants before giving up on the GPU.
    for compute_type in ("float16", "int8_float16"):
        try:
            return WhisperModel(model_size, device=device,
                                device_index=device_index,
                                compute_type=compute_type)
        except Exception:
            continue
    return WhisperModel(model_size, device="cpu", compute_type="int8")


class SpeechTranscriber:
    """Transcribes each speaker turn of 16 kHz mono audio."""

    def __init__(self, model_size: str = "base",
                 language: Optional[str] = None,
                 device: Optional[str] = None):
        self.language = language
        self.model = _load_model(model_size, device)

    def transcribe_segments(self, wav_16k_mono: np.ndarray,
                            segments: List[dict],
                            sr: int = 16000) -> List[dict]:
        """Return copies of ``segments`` with ``transcript`` (and language) set.

        Turns too short to transcribe, and failed transcriptions, get an empty
        transcript -- downstream exports fall back to the speaker label rather
        than carrying fabricated placeholder text.
        """
        updated = []
        for seg in segments:
            a = int(seg["start"] * sr)
            b = min(int(seg["end"] * sr), len(wav_16k_mono))
            chunk = np.asarray(wav_16k_mono[a:b], dtype=np.float32)

            out = dict(seg)
            out["transcript"] = ""
            if seg["end"] - seg["start"] >= 0.35 and len(chunk):
                try:
                    whisper_segs, info = self.model.transcribe(
                        chunk, language=self.language, beam_size=5,
                        vad_filter=False,
                    )
                    out["transcript"] = " ".join(
                        s.text.strip() for s in whisper_segs).strip()
                    out["detected_language"] = info.language
                except Exception:
                    pass
            updated.append(out)
        return updated
