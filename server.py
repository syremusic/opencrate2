"""
OpenCrate API — search the library and get audio bytes back.

The web client can't read your local filesystem, so the server reads the
matched file from disk and streams the actual audio bytes. Search endpoints
return JSON (id + score + an audio_url); the /audio/{id} endpoint serves the
bytes (with HTTP range support, so players can seek).

Run:
    uvicorn server:app --reload --port 8000
    # or: python server.py

Try:
    curl -X POST localhost:8000/search -H 'content-type: application/json' \
         -d '{"query": "dusty lo-fi snare with room ambience", "k": 10}'

    # consume a query, get an audio file straight back:
    curl -L "localhost:8000/play?query=deep%20analog%20kick" --output match.wav
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine import Library, embed_audio, embed_text

INDEX_DIR = Path(os.environ.get("OPENCRATE_INDEX", "opencrate_index"))

# Browsers/mimetypes are unreliable for audio; map the ones we care about.
MIME = {
    ".wav": "audio/wav",
    ".aif": "audio/x-aiff",
    ".aiff": "audio/x-aiff",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
}

app = FastAPI(
    title="OpenCrate", description="Semantic search over a local sample library."
)

# Open CORS so a separate web frontend can call this directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

library: Library | None = None


@app.on_event("startup")
def _startup():
    global library
    library = Library(INDEX_DIR)  # loads index + manifest, then warms nothing else
    print(f"Loaded {len(library):,} samples from {INDEX_DIR}")


class SearchRequest(BaseModel):
    query: str
    k: int = 20


def _to_json(rec: dict, request: Request) -> dict:
    base = str(request.base_url).rstrip("/")
    return {
        "id": rec["id"],
        "filename": rec["filename"],
        "score": round(rec["score"], 4),
        "duration": rec.get("duration"),
        "samplerate": rec.get("samplerate"),
        "audio_url": f"{base}/audio/{rec['id']}",
    }


@app.get("/")
def root():
    return {
        "samples": len(library) if library else 0,
        "endpoints": ["/search", "/search/audio", "/audio/{id}", "/play"],
    }


@app.post("/search")
def search(req: SearchRequest, request: Request):
    """Text query → ranked matches (JSON). Embeds the query with CLAP."""
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    vec = embed_text([req.query])
    hits = library.search(vec, req.k)
    return [_to_json(h, request) for h in hits]


@app.post("/search/audio")
async def search_audio(request: Request, file: UploadFile = File(...), k: int = 20):
    """Upload an audio file → find similar samples (JSON)."""
    data = await file.read()
    suffix = Path(file.filename or "q.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        vec = embed_audio([tmp_path])
        hits = library.search(vec, k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return [_to_json(h, request) for h in hits]


@app.get("/audio/{sample_id}")
def get_audio(sample_id: int):
    """Stream the actual audio bytes for a sample id (supports range requests)."""
    rec = library.get(sample_id)
    if rec is None:
        raise HTTPException(404, "unknown sample id")
    path = Path(rec["path"])
    if not path.exists():
        raise HTTPException(410, "file no longer on disk")
    media = MIME.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/play")
def play(query: str, rank: int = 0):
    """Consume a text query, return one audio file directly (the rank-th match)."""
    if not query.strip():
        raise HTTPException(400, "query is empty")
    vec = embed_text([query])
    hits = library.search(vec, rank + 1)
    if rank >= len(hits):
        raise HTTPException(404, "no match at that rank")
    return get_audio(hits[rank]["id"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000)
