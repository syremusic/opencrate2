"""
OpenCrate API — text query in, ranked audio file paths out.

The caller is another local Python process, so it opens the matched files by
path itself; the server just does the CLAP embed + FAISS search. The API exists
mainly to queue requests: a single lock serializes model inference so a burst of
concurrent queries runs one at a time instead of overwhelming the laptop.

Run:
    uvicorn server:app --reload --port 8000
    # or: python server.py

Try:
    curl -X POST localhost:8000/search -H 'content-type: application/json' \
         -d '{"query": "dusty lo-fi snare with room ambience", "k": 10}'
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import Library, embed_text

INDEX_DIR = Path(os.environ.get("OPENCRATE_INDEX", "opencrate_index"))

library: Library | None = None
# Serialize CLAP inference so queued requests hit the model one at a time.
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global library
    library = Library(INDEX_DIR)
    print(f"Loaded {len(library):,} samples from {INDEX_DIR}")
    yield


app = FastAPI(
    title="OpenCrate",
    description="Semantic search over a local sample library.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str
    k: int = 20


@app.get("/")
def root():
    return {"samples": len(library) if library else 0}


@app.post("/search")
def search(req: SearchRequest):
    """Text query → ranked [{path, score}]. Embeds the query with CLAP."""
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    with _lock:
        vec = embed_text([req.query])
        hits = library.search(vec, req.k)
    return [{"path": h["path"], "score": round(h["score"], 4)} for h in hits]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000)
