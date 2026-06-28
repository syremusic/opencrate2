# OpenCrate

Semantic search over a local sample library. CLAP turns audio (and text) into
embeddings, FAISS does nearest-neighbor search, and a small FastAPI server
returns the matched audio bytes — so a web client that can't see your disk
still gets a playable file.

## Layout

- `engine.py` — loads CLAP once, embeds text/audio, wraps the FAISS index.
- `index_library.py` — recursively scans a folder and builds the index.
- `server.py` — the API.

## Setup

```bash
pip install -r requirements.txt
```

On first run the CLAP music checkpoint (~2 GB) downloads automatically from
Hugging Face. Set `OPENCRATE_CKPT` to point at an existing copy if you have one.

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

Text search (returns JSON: id, filename, score, `audio_url`):

```bash
curl -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "dusty lo-fi snare with room ambience", "k": 10}'
```

Find similar to an audio file you upload:

```bash
curl -X POST localhost:8000/search/audio -F 'file=@mykick.wav'
```

Get the bytes for a result:

```bash
curl localhost:8000/audio/4213 --output sample.wav
```

Consume a query and get one audio file straight back:

```bash
curl -L "localhost:8000/play?query=deep%20analog%20kick" --output match.wav
```

## Notes

- Text and audio share the same embedding space, which is why a text query
  works against an audio-only index.
- The index is `IndexFlatIP` over normalized vectors = exact cosine search.
  Fine up to a few hundred thousand samples; swap in an IVF/HNSW index later
  if your library gets huge.
- `engine.py` is the only file that knows how to load the model and index, so
  both the indexer and server stay thin.
