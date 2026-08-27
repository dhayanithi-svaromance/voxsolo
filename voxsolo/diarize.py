"""Run diarization and turn the result into a plain, serialisable timeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import intervals as iv


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


@dataclass
class Timeline:
    """Diarization result, independent of pyannote's object model.

    ``speakers`` maps each label to its merged speech regions; ``overlap`` is
    every region with >=2 distinct speakers active; ``exclusive`` is per-speaker
    speech with those overlapped regions removed.
    """

    duration: float
    segments: List[Segment] = field(default_factory=list)
    speakers: Dict[str, List[List[float]]] = field(default_factory=dict)
    overlap: List[List[float]] = field(default_factory=list)
    exclusive: Dict[str, List[List[float]]] = field(default_factory=dict)
    source: str = ""

    @property
    def labels(self) -> List[str]:
        return sorted(self.speakers)

    def speech(self) -> List[List[float]]:
        """All speech from all speakers, merged."""
        return iv.union(*self.speakers.values()) if self.speakers else []

    def summary(self) -> str:
        lines = [
            f"duration       : {self.duration:.2f}s",
            f"speakers found : {len(self.speakers)}  ({', '.join(self.labels) or '-'})",
            f"total speech   : {iv.total_duration(self.speech()):.2f}s",
            f"overlapped     : {iv.total_duration(self.overlap):.2f}s "
            f"in {len(self.overlap)} region(s)",
        ]
        for label in self.labels:
            total = iv.total_duration(self.speakers[label])
            solo = iv.total_duration(self.exclusive.get(label, []))
            lines.append(
                f"  {label:<14} {total:7.2f}s total | {solo:7.2f}s solo | "
                f"{total - solo:6.2f}s overlapped"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["segments"] = [asdict(s) if not isinstance(s, dict) else s
                            for s in self.segments]
        return data

    def save(self, path: str) -> str:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Timeline":
        with open(path) as handle:
            data = json.load(handle)
        data["segments"] = [Segment(**s) for s in data.get("segments", [])]
        return cls(**data)

    def write_rttm(self, path: str, uri: str = "audio") -> str:
        """Write the standard RTTM format used by diarization scoring tools.

        RTTM is whitespace-delimited, so any whitespace in ``uri`` (e.g. a
        filename with spaces) is replaced with underscores to keep the file
        parseable by scoring tools.
        """
        uri = "_".join(str(uri).split()) or "audio"
        with open(path, "w") as handle:
            for seg in sorted(self.segments, key=lambda s: (s.start, s.end)):
                handle.write(
                    f"SPEAKER {uri} 1 {seg.start:.3f} {seg.end - seg.start:.3f} "
                    f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
                )
        return path


def build_timeline(segments: List[Segment], duration: float,
                   source: str = "") -> Timeline:
    """Derive speaker / overlap / exclusive interval sets from raw segments."""
    speakers: Dict[str, List[List[float]]] = {}
    for seg in segments:
        speakers.setdefault(seg.speaker, []).append([seg.start, seg.end])
    speakers = {label: iv.merge(ivs) for label, ivs in speakers.items()}

    overlap = iv.coverage_at_least(speakers, 2)
    exclusive = iv.exclusive_by_label(speakers)
    return Timeline(
        duration=duration,
        segments=sorted(segments, key=lambda s: (s.start, s.end)),
        speakers=speakers,
        overlap=overlap,
        exclusive=exclusive,
        source=source,
    )


def diarize(
    audio_path: str,
    duration: float,
    pipeline,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    source: str = "",
) -> Timeline:
    """Run a loaded pyannote pipeline over ``audio_path``.

    Leave the speaker-count arguments as ``None`` to let the pipeline decide.
    Give ``num_speakers`` only when you actually know the count -- forcing a wrong
    value is a common cause of bad diarization.
    """
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    result = pipeline(audio_path, **kwargs)
    # pyannote 4.x returns a wrapper object; 3.x returns an Annotation directly.
    annotation = getattr(result, "speaker_diarization", result)

    segments = [
        Segment(start=round(float(turn.start), 3),
                end=round(float(turn.end), 3),
                speaker=str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    return build_timeline(segments, duration=duration, source=source)
