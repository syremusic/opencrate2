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
    manifest.jsonl   one JSON record per vector (path + metadata)
    bad.txt          files that failed to read/embed
    index.faiss      the final searchable index (built at the end)
"""

from __future__ import annotations

# import torch before faiss to avoid segfaults on Apple Silicon
import torch
import faiss

import argparse
import json
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
    MANIFEST,
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


def header_info(path: Path) -> tuple[float, int] | None:
    """Read duration + sample rate from the header only (fast, no decode)."""
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.frames <= 0:
            return None
        return info.frames / info.samplerate, info.samplerate
    except Exception:
        return None


# ───────── crash recovery ─────────────────────────────────────────────────
def _rows_in_bin(p: Path) -> int:
    return p.stat().st_size // (DIM * 4) if p.exists() else 0


def _count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def repair(emb_f: Path, man_f: Path) -> None:
    """Trim embeddings.f32 / manifest.jsonl to the same length after a crash."""
    n_emb, n_man = _rows_in_bin(emb_f), _count_lines(man_f)
    keep = min(n_emb, n_man)
    if n_emb > keep:
        with emb_f.open("r+b") as fh:
            fh.truncate(keep * DIM * 4)
    if n_man > keep and man_f.exists():
        lines = man_f.open().readlines()[:keep]
        with man_f.open("w") as fh:
            fh.writelines(lines)


def load_set(p: Path, key: str | None = None) -> set[str]:
    if not p.exists():
        return set()
    if key is None:
        return {line.strip() for line in p.open() if line.strip()}
    return {json.loads(line)[key] for line in p.open() if line.strip()}


# ───────── append ─────────────────────────────────────────────────────────
def append(emb_f: Path, man_f: Path, vecs: np.ndarray, records: list[dict]) -> None:
    with emb_f.open("ab") as fh:
        fh.write(vecs.astype("float32", copy=False).tobytes(order="C"))
        fh.flush()
        os.fsync(fh.fileno())
    with man_f.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ───────── build ──────────────────────────────────────────────────────────
def build(root: Path, out: Path, batch: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    emb_f, man_f, bad_f, idx_f = (
        out / EMB_FILE,
        out / MANIFEST,
        out / BAD_FILE,
        out / INDEX_FILE,
    )

    repair(emb_f, man_f)
    done = load_set(man_f, key="path")
    bad = load_set(bad_f)

    print("Scanning files…")
    todo = [p for p in scan_audio(root) if str(p) not in done and str(p) not in bad]
    print(f"{len(done):,} already indexed · {len(todo):,} new files to process")
    if not todo:
        print("Nothing new to do.")
    else:
        _process(todo, emb_f, man_f, bad_f, batch)

    if _stop:
        print("🛑 Stopped by user. Re-run to resume.")
        return

    _build_faiss(emb_f, idx_f)


def _process(todo, emb_f, man_f, bad_f, batch):
    pending_paths: list[Path] = []
    pending_meta: list[tuple[float, int]] = []

    def flush():
        if not pending_paths:
            return
        try:
            vecs = embed_audio([str(p) for p in pending_paths])
        except Exception as e:
            # If a whole batch fails, fall back to one-by-one so one bad
            # file doesn't poison the rest.
            print(f"Batch embed failed ({e}); retrying individually…")
            for p, m in zip(pending_paths, pending_meta):
                try:
                    v = embed_audio([str(p)])
                    append(emb_f, man_f, v, [_record(p, m)])
                except Exception:
                    with bad_f.open("a") as f:
                        f.write(str(p) + "\n")
        else:
            records = [_record(p, m) for p, m in zip(pending_paths, pending_meta)]
            append(emb_f, man_f, vecs, records)
        pending_paths.clear()
        pending_meta.clear()

    for p in tqdm(todo, ncols=88, desc="Embedding"):
        if _stop:
            break
        info = header_info(p)
        if info is None or info[0] > MAX_SECONDS:
            with bad_f.open("a") as f:
                f.write(str(p) + "\n")
            continue
        pending_paths.append(p)
        pending_meta.append(info)
        if len(pending_paths) >= batch:
            flush()
    flush()


def _record(p: Path, meta: tuple[float, int]) -> dict:
    duration, samplerate = meta
    stat = p.stat()
    return {
        "path": str(p),
        "filename": p.name,
        "bytes": stat.st_size,
        "mtime": int(stat.st_mtime),
        "duration": round(duration, 3),
        "samplerate": samplerate,
    }


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
