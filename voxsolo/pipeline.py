"""Loading a pyannote speaker-diarization pipeline.

The official pyannote pipelines live in **gated** Hugging Face repos: you must be
logged in *and* have accepted the model's conditions on the model page. When that
has not happened you get a 403 ``GatedRepoError`` at download time -- note that
``HfApi.model_info()`` still succeeds on a gated repo, so a metadata call is not
a valid access check.

This module fails with an actionable message in that case, and offers an opt-in
fallback that reconstructs the same pipeline from non-gated component repos.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import threading
import warnings
from dataclasses import dataclass
from typing import Optional

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"

# Tried in order when the caller does not name a pipeline explicitly: the
# pyannote 4.x community pipeline is the current standard and measurably more
# accurate than 3.1 (its predecessor), but both are gated -- the mirror
# fallback reconstructs the 3.1 recipe from non-gated component repos.
DEFAULT_PIPELINE_ORDER = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
)

# Hyper-parameters of the official speaker-diarization-3.1 recipe.
_V31_PARAMS = {
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.7045654963945799,
    },
    "segmentation": {"min_duration_off": 0.0},
}

# Community re-uploads of pyannote/segmentation-3.0 (MIT). Pinned by SHA-256 of
# pytorch_model.bin so a swapped mirror is detected rather than silently trusted.
# Verified identical across all three repos below.
_SEGMENTATION_SHA256 = (
    "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
)
_SEGMENTATION_MIRRORS = (
    "tensorlake/segmentation-3.0",
    "ivrit-ai/pyannote-segmentation-3.0",
    "it-just-works/pyannote-segmentation",
)
_EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"  # not gated

# The get_plda monkeypatch below mutates module-global state; serialise it.
_PATCH_LOCK = threading.Lock()

_GATED_HELP = """
Cannot download '{repo}': the repo is gated and this account has not been granted access.

To fix (one time, ~1 minute):
  1. Log in at https://huggingface.co and open:
       https://hf.co/pyannote/speaker-diarization-3.1
       https://hf.co/pyannote/segmentation-3.0
     Accept the conditions on BOTH pages.
  2. Make a token at https://hf.co/settings/tokens
  3. Expose it as HF_TOKEN, or run:  huggingface-cli login

Alternatively, pass --allow-mirrors (CLI) or allow_mirrors=True (API) to rebuild
the same pipeline from non-gated community mirrors of the model weights.
""".strip()


class GatedModelError(RuntimeError):
    """Raised when a gated repo cannot be downloaded, with instructions."""


@dataclass
class LoadedPipeline:
    """A diarization pipeline plus a note on where its weights came from."""

    pipeline: object
    source: str
    used_mirrors: bool


def _resolve_token(token: Optional[str]) -> Optional[str]:
    """Token from the argument, the environment, or the huggingface-cli login."""
    if token:
        return token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        # Respects HF_HOME / HF_TOKEN_PATH and the huggingface-cli login cache.
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official(token: Optional[str], pipeline_name: str):
    from huggingface_hub.errors import GatedRepoError
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(pipeline_name, token=token)
    except GatedRepoError as exc:
        raise GatedModelError(_GATED_HELP.format(repo=pipeline_name)) from exc
    if pipeline is None:
        # Older pyannote versions return None instead of raising on auth failure.
        raise GatedModelError(_GATED_HELP.format(repo=pipeline_name))
    return pipeline


def _load_from_mirrors(token: Optional[str], verify_checksum: bool):
    """Rebuild the 3.1 recipe from non-gated component repos.

    pyannote-audio 4.x's ``SpeakerDiarization.__init__`` eagerly fetches a PLDA
    from a gated repo even when the chosen clustering never uses it. ``_plda`` is
    only read when ``clustering == "VBxClustering"``, so for AgglomerativeClustering
    we can safely stub the getter out.
    """
    from huggingface_hub import hf_hub_download
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization
    import pyannote.audio.pipelines.speaker_diarization as sd_module

    last_error = None
    for repo in _SEGMENTATION_MIRRORS:
        try:
            weights = hf_hub_download(repo, "pytorch_model.bin", token=token)
        except Exception as exc:  # try the next mirror
            last_error = exc
            continue
        if verify_checksum:
            actual = _sha256(weights)
            if actual != _SEGMENTATION_SHA256:
                warnings.warn(
                    f"Mirror {repo} checksum {actual[:16]}... does not match the "
                    f"pinned value {_SEGMENTATION_SHA256[:16]}.... Skipping it. "
                    f"Pass verify_checksum=False to use it anyway.",
                    RuntimeWarning,
                )
                last_error = RuntimeError(f"checksum mismatch for {repo}")
                continue
        segmentation = Model.from_pretrained(weights)
        break
    else:
        raise GatedModelError(
            "No usable segmentation mirror. Last error: "
            f"{type(last_error).__name__}: {last_error}"
        )

    _PATCH_LOCK.acquire()
    original_get_plda = getattr(sd_module, "get_plda", None)
    if original_get_plda is not None:
        sd_module.get_plda = lambda *args, **kwargs: None
    try:
        kwargs = dict(
            segmentation=segmentation,
            embedding=_EMBEDDING_REPO,
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
        )
        try:
            # pyannote-audio >= 4 exposes legacy=True for the v3.1 behaviour.
            pipeline = SpeakerDiarization(legacy=True, plda=None, token=token, **kwargs)
        except TypeError:
            pipeline = SpeakerDiarization(**kwargs)
    finally:
        if original_get_plda is not None:
            sd_module.get_plda = original_get_plda
        _PATCH_LOCK.release()

    pipeline.instantiate(_V31_PARAMS)
    return pipeline, repo


def load_pipeline(
    pipeline_name: str = DEFAULT_PIPELINE,
    token: Optional[str] = None,
    device: Optional[str] = None,
    allow_mirrors: bool = False,
    verify_checksum: bool = True,
) -> LoadedPipeline:
    """Load a diarization pipeline, moving it to ``device`` if given.

    When ``pipeline_name`` is the default, each name in
    ``DEFAULT_PIPELINE_ORDER`` is tried in turn (community-1 first -- the
    current pyannote standard -- then legacy 3.1); an explicitly named
    pipeline is tried alone. With ``allow_mirrors=True``, ANY official-load
    failure (gated repo, network down, transient HTTP) falls back to
    reconstructing the 3.1 recipe from non-gated component repos -- mirror
    weights already in the local cache then work fully offline. Without it,
    a gated-repo failure raises :class:`GatedModelError` with instructions.
    """
    import torch

    token = _resolve_token(token)
    candidates = (
        DEFAULT_PIPELINE_ORDER if pipeline_name == DEFAULT_PIPELINE
        else (pipeline_name,)
    )

    pipeline = None
    source = ""
    used_mirrors = False
    first_error: Optional[Exception] = None
    for name in candidates:
        try:
            if allow_mirrors:
                # pyannote prints a multi-line "could not download" banner
                # straight to stdout/stderr. Swallow it while a fallback is
                # still possible, so a successful later load does not look
                # like a failure.
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    pipeline = _load_official(token, name)
            else:
                pipeline = _load_official(token, name)
            source = name
            break
        except Exception as exc:  # noqa: BLE001 - every load error is a fallback cue
            # Keep the most actionable error: a GatedModelError carries the
            # accept-the-terms instructions.
            if first_error is None or isinstance(exc, GatedModelError):
                first_error = exc

    if pipeline is None and not allow_mirrors:
        raise first_error

    if pipeline is None:
        # Official loads all failed and mirrors are allowed.
        warnings.warn(
            f"official pipeline load failed "
            f"({type(first_error).__name__}); using mirrored weights",
            RuntimeWarning,
        )
        pipeline, mirror = _load_from_mirrors(token, verify_checksum)
        source = f"{mirror} + {_EMBEDDING_REPO} (3.1 recipe, mirrored weights)"
        used_mirrors = True

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return LoadedPipeline(pipeline=pipeline, source=source, used_mirrors=used_mirrors)
