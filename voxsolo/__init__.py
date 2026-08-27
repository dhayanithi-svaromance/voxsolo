"""voxsolo: speaker diarization, verbatim speaker isolation, NLE/DAW exports."""

from .audio import (
    MediaInfo,
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
from .diarize import Segment, Timeline, build_timeline, diarize
from .intervals import coverage_at_least, exclusive_by_label, merge, subtract
from .isolate import IsolateResult, compute_keep_regions, render, verify
from .pipeline import DEFAULT_PIPELINE, GatedModelError, load_pipeline

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PIPELINE",
    "GatedModelError",
    "IsolateResult",
    "MediaInfo",
    "Segment",
    "Timeline",
    "build_timeline",
    "compute_keep_regions",
    "coverage_at_least",
    "diarize",
    "exclusive_by_label",
    "export_audacity_labels",
    "export_davinci_csv",
    "export_edl",
    "export_srt",
    "export_vtt",
    "extract_analysis",
    "extract_master",
    "load_pipeline",
    "merge",
    "mux_audio",
    "probe",
    "render",
    "subtract",
    "verify",
]
