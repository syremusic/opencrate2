"""
OpenCrate indexer — recursively scan a folder, embed every audio file with
CLAP, and build a FAISS index.

Usage:
    python index_library.py --root /Users/dmitriyburenok/Splice
    python index_library.py --root /Users/dmitriyburenok/Splice --out opencrate_index --batch 8

It's resumable: re-running picks up where it left off (already-indexed files
are skipped, unreadable ones are remembered in bad.txt). Ctrl-C finishes the
current batch then stops cleanly.

Outputs (in --out dir):
    embeddings.f32   raw normalized float32 vectors, appended as we go
    paths.txt        one audio file path per vector (parallel to embeddings)
    bad.txt          files that failed to read/embed
    index.faiss      the final searchable index (built at the end)
"""

from __future__ import annotations

# import torch before faiss to avoid segfaults on Apple Silicon
import torch
import faiss

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from engine import (
    DIM,
    EMB_FILE,
    PATHS_FILE,
    BAD_FILE,
    INDEX_FILE,
    AUDIO_EXTS,
    embed_audio,
)

MAX_SECONDS = 90  # skip very long files (loops/stems); CLAP only needs a clip

# ───────── graceful interrupt ─────────────────────────────────────────────
_stop = False


def _sigint(_sig, _frame):
    global _stop
    _stop = True
    print("\n⚠️  Interrupt received — finishing current batch, then stopping…\n")


signal.signal(signal.SIGINT, _sigint)


# ───────── filesystem scan ────────────────────────────────────────────────
def scan_audio(root: Path):
    """Yield every audio file under root, recursively. Skips mac junk."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("._"):
                continue
            if Path(name).suffix.lower() in AUDIO_EXTS:
                yield Path(dirpath) / name


def duration_of(path: Path) -> float | None:
    """Read duration from the header only (fast, no decode). None if unreadable."""
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.frames <= 0:
            return None
        return info.frames / info.samplerate
    except Exception:
        return None


# ───────── crash recovery ─────────────────────────────────────────────────
def _rows_in_bin(p: Path) -> int:
    return p.stat().st_size // (DIM * 4) if p.exists() else 0


def _count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def repair(emb_f: Path, paths_f: Path) -> None:
    """Trim embeddings.f32 / paths.txt to the same length after a crash."""
    n_emb, n_paths = _rows_in_bin(emb_f), _count_lines(paths_f)
    keep = min(n_emb, n_paths)
    if n_emb > keep:
        with emb_f.open("r+b") as fh:
            fh.truncate(keep * DIM * 4)
    if n_paths > keep and paths_f.exists():
        lines = paths_f.open().readlines()[:keep]
        with paths_f.open("w") as fh:
            fh.writelines(lines)


def load_set(p: Path) -> set[str]:
    if not p.exists():
        return set()
    return {line.strip() for line in p.open() if line.strip()}


# ───────── append ─────────────────────────────────────────────────────────
def append(emb_f: Path, paths_f: Path, vecs: np.ndarray, paths: list[str]) -> None:
    with emb_f.open("ab") as fh:
        fh.write(vecs.astype("float32", copy=False).tobytes(order="C"))
        fh.flush()
        os.fsync(fh.fileno())
    with paths_f.open("a") as fh:
        for path in paths:
            fh.write(path + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ───────── build ──────────────────────────────────────────────────────────
def build(root: Path, out: Path, batch: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    emb_f, paths_f, bad_f, idx_f = (
        out / EMB_FILE,
        out / PATHS_FILE,
        out / BAD_FILE,
        out / INDEX_FILE,
    )

    repair(emb_f, paths_f)
    done = load_set(paths_f)
    bad = load_set(bad_f)

    print("Scanning files…")
    todo = [p for p in scan_audio(root) if str(p) not in done and str(p) not in bad]
    print(f"{len(done):,} already indexed · {len(todo):,} new files to process")
    if not todo:
        print("Nothing new to do.")
    else:
        _process(todo, emb_f, paths_f, bad_f, batch)

    if _stop:
        print("🛑 Stopped by user. Re-run to resume.")
        return

    _build_faiss(emb_f, idx_f)


def _process(todo, emb_f, paths_f, bad_f, batch):
    pending_paths: list[Path] = []

    def flush():
        if not pending_paths:
            return
        try:
            vecs = embed_audio([str(p) for p in pending_paths])
        except Exception as e:
            # If a whole batch fails, fall back to one-by-one so one bad
            # file doesn't poison the rest.
            print(f"Batch embed failed ({e}); retrying individually…")
            for p in pending_paths:
                try:
                    v = embed_audio([str(p)])
                    append(emb_f, paths_f, v, [str(p)])
                except Exception:
                    with bad_f.open("a") as f:
                        f.write(str(p) + "\n")
        else:
            append(emb_f, paths_f, vecs, [str(p) for p in pending_paths])
        pending_paths.clear()

    for p in tqdm(todo, ncols=88, desc="Embedding"):
        if _stop:
            break
        dur = duration_of(p)
        if dur is None or dur > MAX_SECONDS:
            with bad_f.open("a") as f:
                f.write(str(p) + "\n")
            continue
        pending_paths.append(p)
        if len(pending_paths) >= batch:
            flush()
    flush()


def _build_faiss(emb_f: Path, idx_f: Path) -> None:
    total = _rows_in_bin(emb_f)
    if total == 0:
        print("No embeddings to index.")
        return
    print(f"Building FAISS IndexFlatIP from {total:,} vectors…")
    vecs = np.fromfile(emb_f, dtype="float32").reshape(total, DIM)
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    faiss.write_index(index, str(idx_f))
    print(f"💾 Saved {idx_f}\n✅ Done — {total:,} samples indexed.")


def main():
    ap = argparse.ArgumentParser(
        description="Recursively index an audio library with CLAP + FAISS."
    )
    ap.add_argument(
        "--root", required=True, type=Path, help="Folder to scan recursively"
    )
    ap.add_argument(
        "--out",
        default=Path("opencrate_index"),
        type=Path,
        help="Where to store the index",
    )
    ap.add_argument("--batch", default=8, type=int, help="Embedding batch size")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"❌ Not a directory: {args.root}")

    t0 = time.time()
    build(args.root, args.out, args.batch)
    print(f"⏱️  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
