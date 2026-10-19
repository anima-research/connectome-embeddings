# Connectome Embeddings

Shared embedding and vector-index service for Connectome, with an OpenAI-compatible embedding API and an Apple Silicon backend.

This repository starts as a **source-only import of an existing deployment**. `server.py`, `index_store.py`, and `run.sh` are preserved byte-for-byte; this import does not migrate, restart, or reconfigure the running service. See [import notes](docs/import-notes.md).

## What runs

- `server.py`: aiohttp service, one embedding worker, dynamic micro-batching.
- `index_store.py`: per-namespace SQLite vector indexes with in-memory cosine search.
- Default model: `jinaai/jina-embeddings-v5-omni-small`, text modality, MPS/bfloat16, 1024-dimensional embeddings (optional truncation to 32–1024 dimensions).
- Default port: **8804**. `GET /healthz`, `GET /v1/models`, `POST /v1/embeddings`.

Run **one model worker per deployment**, not multiple GPU workers competing for the same memory. CPU and other devices are CLI options, not platforms verified by this import.

## Setup

Use a Python/PyTorch combination supported on your machine. The imported deployment uses system Python 3.9 on Apple Silicon; its observed package versions are listed in `constraints-macos-observed.txt`. Those constraints are a record, not a fully resolved lockfile. A fresh installation has not been tested as part of this import.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
# Set EMBED_TOKEN in .env to a fresh, strong secret before proceeding.
set -a; . ./.env; set +a
export HF_HOME="$PWD/hf-home"
python server.py --host 127.0.0.1 --port 8804 --modality text
```

Initial model loading can download weights and model implementation code: the server uses `trust_remote_code=True`. Review that code and the model's license before running. For an already populated cache, set `HF_HUB_OFFLINE=1`. No weights are included here.

The retained `run.sh` is the **existing Mac launcher**, not a portable installer: it uses `/usr/bin/python3`, sources `.env` in its own directory, forces offline Hugging Face loading, and uses `$HOME/embed-service/hf-home`. The generic setup above avoids assuming that layout. An uninstalled [launchd template](deploy/com.anima.connectome-embeddings.plist.example) is provided; replace its paths before use. Do not start a second instance beside the existing deployment.

## Security and data

**Set `EMBED_TOKEN`.** If it is empty, authentication is disabled. The CLI defaults to `0.0.0.0`; the setup above deliberately binds loopback. Use a trusted private network or authenticated proxy and firewall; do not expose an unauthenticated index service publicly.

The bearer token grants access to every namespace; namespace names are not an authorization boundary. Indexes contain original text and metadata, not just vectors. `.env`, indexes, logs, model caches, weights and backups are excluded from Git and must remain private. Back them up separately where appropriate.

Text-only Jina mode pins input resolution to text and refuses startup if it cannot do so. URL-looking strings must not turn into media downloads or local file reads. Non-text modalities are not hardened or exercised by this source-import test suite.

## Embedding API

```sh
curl -H "Authorization: Bearer $EMBED_TOKEN" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8804/v1/embeddings \
  -d '{"input":["example document"],"input_type":"document","dimensions":256}'
```

`input`: string or array of strings. `task`: retrieval (default), text-matching, classification or clustering. `input_type`: query or document. `encoding_format`: float (default) or base64. Responses contain OpenAI-shaped embeddings and token usage.

## Index API

Namespaces commonly use `<agent>/<sessionId>`; URL-encode them in the path.

- `GET /v1/index`: list namespaces and counts.
- `GET /v1/index/{ns}/stats`: count and per-kind timestamp/cursor ranges.
- `POST /v1/index/{ns}/upsert`: `{ "items": [{ "id", "text", "ts", "channel", "kind", "level", "cursor", "meta" }] }`; only id/text required. Text changes re-embed; unchanged text refreshes metadata.
- `POST /v1/index/{ns}/delete`: `{ "ids": [...] }`; tombstones entries.
- `POST /v1/index/{ns}/search`: `{ "query", "k", "ts_from", "ts_to", "channel", "kinds", "level", "min_score", "snippet", "with_text" }`; query required.

SQLite under `--index-dir` is the persisted source; the float32 search matrix is rebuilt lazily. The Connectome host/framework own history synchronization. Local one-off resident-specific ingestion and rehearsal scripts were deliberately not imported; never open a live Chronicle store from a second process for backfilling.

## Memory controls

The imported implementation includes length buckets, token/attention budgets, per-cycle MPS cache release, a default 24 GiB MPS allocator limit, and traceback-free worker error propagation. `/healthz` reports MPS allocation metrics. `python server.py --help` lists controls without loading model weights. These were introduced in the source deployment after variable-length workloads accumulated large MPS allocations; the initial repository import does not independently reproduce those load tests.

## Tests

```sh
python -m unittest discover -s tests -v
```

Synthetic temporary indexes and fake model helpers only: no resident stores, live service calls, model downloads or GPU inference. This is limited smoke/regression coverage, not a complete concurrency, API, security or load audit.

## Licensing

The default Jina model was documented by the source deployment as **CC BY-NC 4.0**; check its current model card and dependency licenses for your use. That is a model license, not a license grant for this repository's own source. No software license has been selected yet; this repository starts private pending the maintainer's decision.
