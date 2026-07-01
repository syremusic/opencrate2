"""
OpenCrate engine — shared by the indexer and the API server.

Handles three things:
  1. Loading the LAION-CLAP music model via Hugging Face transformers
     (lazily, once). Weights are fetched + cached by the HF hub.
  2. Turning text or audio into normalized 512-d embeddings.
  3. Loading a built FAISS index + paths list and searching it.

CLAP embeds text and audio into the *same* space, so a text query like
"dusty lo-fi snare with room ambience" can be matched directly against
audio embeddings via cosine similarity.
"""

from __future__ import annotations

# import torch before faiss to avoid segfaults on Apple Silicon
import torch
import faiss

import os
from pathlib import Path

import numpy as np

DIM = 512
SAMPLE_RATE = 48_000  # CLAP expects 48 kHz mono audio
# LAION's music+speech CLAP, hosted + cached by the HF hub. Override with
# OPENCRATE_MODEL (e.g. "laion/larger_clap_music" or "laion/clap-htsat-unfused").
MODEL_ID = os.environ.get("OPENCRATE_MODEL", "laion/larger_clap_music_and_speech")

# Paths / index filenames (shared between indexer and server)
INDEX_FILE = "index.faiss"
PATHS_FILE = "paths.txt"
EMB_FILE = "embeddings.f32"
BAD_FILE = "bad.txt"

AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_model = None


def load_clap():
    """Load the CLAP model + processor once and cache them for the process.

    Returns (model, processor, device). Weights are downloaded and cached by
    the HF hub on first use (~2 GB, one time).
    """
    global _model
    if _model is not None:
        return _model

    # imported lazily so the indexer/server start fast
    from transformers import ClapModel, ClapProcessor

    device = get_device()
    print(f"Loading CLAP ({MODEL_ID}) on {device}…")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    _model = (model, processor, device)
    return _model


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def embed_text(queries: list[str]) -> np.ndarray:
    """Embed one or more text queries → (n, 512) normalized float32."""
    model, processor, device = load_clap()
    inputs = processor(text=queries, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vecs = model.get_text_features(**inputs)
    return _normalize(vecs.cpu().numpy())


def embed_audio(paths: list[str]) -> np.ndarray:
    """Embed one or more audio files → (n, 512) normalized float32."""
    import librosa  # imported lazily so the indexer/server start fast

    model, processor, device = load_clap()
    audios = [librosa.load(p, sr=SAMPLE_RATE, mono=True)[0] for p in paths]
    inputs = processor(
        audio=audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        vecs = model.get_audio_features(**inputs)
    return _normalize(vecs.cpu().numpy())


class Library:
    """A loaded FAISS index plus the parallel list of audio file paths.

    Row i of the index corresponds to paths[i].
    """

    def __init__(self, index_dir: str | Path):
        d = Path(index_dir)
        idx_path, paths_path = d / INDEX_FILE, d / PATHS_FILE
        if not idx_path.exists() or not paths_path.exists():
            raise FileNotFoundError(
                f"Index not found in {d}. Run index_library.py first."
            )
        self.index = faiss.read_index(str(idx_path))
        self.paths = [line.strip() for line in paths_path.open() if line.strip()]
        if self.index.ntotal != len(self.paths):
            raise ValueError(
                f"Index/paths mismatch: {self.index.ntotal} vectors "
                f"vs {len(self.paths)} paths."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def search(self, vec: np.ndarray, k: int) -> list[dict]:
        """vec: (1, 512) normalized. Returns [{path, score}], ranked."""
        k = max(1, min(k, len(self.paths)))
        scores, ids = self.index.search(vec, k)
        out = []
        for sid, score in zip(ids[0], scores[0]):
            if sid < 0:
                continue
            out.append({"path": self.paths[int(sid)], "score": float(score)})
        return out
