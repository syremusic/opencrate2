# OpenCrate

Semantic search over a local sample library. CLAP turns audio (and text) into
embeddings, FAISS does nearest-neighbor search, and a small FastAPI server takes
a text query and returns the top matching audio file paths.

## Layout

- `engine.py` — loads CLAP once, embeds text/audio, wraps the FAISS index.
- `index_library.py` — recursively scans a folder and builds the index.
- `server.py` — the API (one `/search` endpoint).

## Setup

```bash
pip install -r requirements.txt
```

On first run the CLAP music checkpoint (~2 GB) downloads automatically from
Hugging Face. Set `OPENCRATE_MODEL` to use a different checkpoint.

## 1. Build the index

```bash
python index_library.py --root /Users/dmitriyburenok/Splice
```

This walks the tree, embeds every `.wav/.aiff/.flac/.mp3/.ogg`, and writes
everything to `opencrate_index/`. It's resumable — re-run it any time and it
skips what's already done. Ctrl-C stops cleanly.

## 2. Run the API

```bash
uvicorn server:app --port 8000        # or: python server.py
```

## 3. Use it

Text search returns a ranked JSON list of `{path, score}`:

```bash
curl -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "dusty lo-fi snare with room ambience", "k": 10}'
```

```json
[
  {"path": "/Users/.../snare_dusty_01.wav", "score": 0.4821},
  {"path": "/Users/.../room_snare.aiff", "score": 0.461}
]
```
