#!/usr/bin/env python3
"""embed-service — shared text-embedding server for the fleet (Mac Studio, MPS).

OpenAI-compatible `POST /v1/embeddings` in front of jinaai/jina-embeddings-v5-omni-small
(text tower only by default; same vector space as jina-embeddings-v5-text-small).
All requests funnel through ONE model worker with dynamic micro-batching: the worker
takes whatever arrived, waits up to --batch-window ms for more, groups by (task,input_type)
and encodes each group as a single batch. Concurrency therefore comes from batching,
not from extra model instances (measured 2026-09-22: 1 proc bs16 23 docs/s, bs128 43 docs/s;
4 procs bs16 aggregate 51 docs/s — same GPU ceiling).

Request (JSON):
  input        str | [str]                        required, <= --max-inputs items
  task         "retrieval"|"text-matching"|"classification"|"clustering"   default retrieval
               (Jina API style "retrieval.query"/"retrieval.passage" also accepted)
  input_type   "query"|"document"                 default document (retrieval only)
  dimensions   int in {32..1024}                  Matryoshka truncation + re-normalisation
  encoding_format "float"|"base64"                default float
Response: OpenAI embeddings shape + usage.prompt_tokens.
Auth: `Authorization: Bearer $EMBED_TOKEN` when EMBED_TOKEN is set (healthz is open).
"""
import argparse, asyncio, base64, gc, json, os, queue, sys, threading, time
import numpy as np
from aiohttp import web
from index_store import IndexRegistry

def log(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), *a, flush=True)

TASKS = {"retrieval", "text-matching", "classification", "clustering"}

# Padded sequence lengths. Every distinct (batch, length) shape costs MPS a fresh graph + fresh Metal buffers
# that the caching allocator keeps; with padding-to-longest over 1..8192 that was thousands of shapes and
# ~108 GB of owned-unmapped memory after 8 days (2026-10-14). Buckets: 8 per octave (≤12.5% padding), min 16
# → ~56 lengths up to 8192 instead of 8192.
def bucket_len(n, max_len):
    n = max(1, int(n))
    step = max(16, 1 << max(0, (n - 1).bit_length() - 4))
    return min(max_len, -(-n // step) * step)

def pin_text_only(model):
    """jina-v5-omni's custom_st resolves EVERY input string as possible media before tokenising: a string that
    starts with http(s):// is urlretrieve()d, a string naming an existing local file is opened and sniffed.
    For a text service that is (a) an SSRF — any token holder makes the Studio fetch any URL — and (b) a poison
    pill: a message whose text is a dead link raises HTTPError 404 and fails its whole (task, input_type) group,
    so the client's retry fails forever (Mythos indexer, every ~16 min 10-10 → 10-14). Text modality ⇒ a str is text."""
    mod = sys.modules.get(type(model[0]).__module__)
    if mod is None or not callable(getattr(mod, "_resolve_input", None)) or not callable(getattr(mod, "_is_media_string", None)):
        raise RuntimeError(f"cannot find _resolve_input in {type(model[0]).__module__}; refusing to serve with URL fetching live")
    orig = mod._resolve_input
    def _resolve_text_only(x):
        return ("text", x) if isinstance(x, str) else orig(x)
    mod._resolve_input = _resolve_text_only
    try: leaked = mod._is_media_string("http://127.0.0.1:9/embed-service-selfcheck.png") or mod._is_media_string(sys.executable)
    except Exception as e: raise RuntimeError(f"text-only pin did not take (probe raised {e!r})")
    if leaked: raise RuntimeError("text-only pin did not take (probe resolved as media)")

def install_length_buckets(model, max_len):
    """Pad each tokenised batch up to the next LENGTH_BUCKETS size (same side the tokenizer pads, mask = 0, so
    the last-token pool reads the same position). Returns the pad side, or None when it cannot be done safely."""
    import torch
    tr = model[0]; tok = tr.tokenizer
    pad_id, side = tok.pad_token_id, getattr(tok, "padding_side", "right")
    if pad_id is None or side not in ("left", "right"): return None
    prompts = set((getattr(model, "prompts", None) or {}).values())
    orig = tr.tokenize
    def tokenize(texts, *a, **kw):
        f = orig(texts, *a, **kw)
        ids = f.get("input_ids") if hasattr(f, "get") else None
        if not torch.is_tensor(ids) or ids.dim() != 2: return f
        if len(texts) == 1 and texts[0] in prompts: return f  # ST's prompt-length probe: leave it exact
        L = ids.shape[1]; extra = bucket_len(L, max_len) - L
        if extra <= 0: return f
        spec = (0, extra) if side == "right" else (extra, 0)
        for k in list(f.keys()):
            v = f[k]
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == L:
                f[k] = torch.nn.functional.pad(v, spec, value=pad_id if k == "input_ids" else 0)
        return f
    tr.tokenize = tokenize
    return side

def mps_stats(limit_bytes):
    try:
        import torch
        return {"allocated_gb": round(torch.mps.current_allocated_memory() / 2**30, 2),
                "driver_gb": round(torch.mps.driver_allocated_memory() / 2**30, 2),
                "limit_gb": round(limit_bytes / 2**30, 1) if limit_bytes else None}
    except Exception as e:
        return {"error": repr(e)}

class Worker(threading.Thread):
    def __init__(self, model, loop, batch_window_ms, max_batch, encode_batch, token_budget, attn_budget=0,
                 release_cache=False, bucketed=False):
        super().__init__(daemon=True, name="embed-worker")
        self.model, self.loop = model, loop
        self.window = batch_window_ms / 1000.0
        self.max_batch, self.encode_batch, self.token_budget = max_batch, encode_batch, token_budget
        self.attn_budget = attn_budget
        self.release_cache, self.bucketed = release_cache, bucketed
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.stats = {"requests": 0, "inputs": 0, "batches": 0, "tokens": 0, "errors": 0, "busy_s": 0.0}
        self.busy = False

    def submit(self, items, task, input_type, dims, fut):
        self.q.put((items, task, input_type, dims, fut))

    def run(self):
        while True:
            first = self.q.get()
            jobs = [first]
            deadline = time.monotonic() + self.window
            n = len(first[0])
            while n < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0: break
                try: j = self.q.get(timeout=remaining)
                except queue.Empty: break
                jobs.append(j); n += len(j[0])
            self.busy = True
            t0 = time.monotonic()
            try: self._process(jobs)
            finally:
                if self.release_cache:  # hand pooled Metal buffers back; otherwise they accumulate per shape
                    try:
                        import torch; torch.mps.empty_cache()
                    except Exception as e: log("empty_cache failed:", repr(e))
                self.busy = False
                self.stats["busy_s"] += time.monotonic() - t0

    def _encode(self, texts, task, input_type):
        """Length-sorted, token-budgeted sub-batches: a batch is padded to its longest member, so
        64 x 8k-token documents would be one 512k-token attention pass. Cap tokens-per-batch instead,
        and batch x len^2 (attention scores are materialised: 3 x 8192 = 6 GiB per score tensor)."""
        lens = [len(ids) for ids in self.model.tokenizer(texts, truncation=True, max_length=self.model.max_seq_length,
                                                         add_special_tokens=True)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: -lens[i])
        out = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        kw = {"task": task, "normalize_embeddings": True}
        if task == "retrieval": kw["prompt_name"] = input_type
        i = 0
        while i < len(order):
            longest = max(1, lens[order[i]])
            if self.bucketed: longest = bucket_len(longest + 8, self.model.max_seq_length)  # +8 ≈ "Document: " prompt
            cap = max(1, min(self.encode_batch, self.token_budget // longest,
                             self.attn_budget // (longest * longest) if self.attn_budget else self.encode_batch))
            sub = order[i:i + cap]
            out[sub] = np.asarray(self.model.encode([texts[j] for j in sub], batch_size=len(sub), **kw), dtype=np.float32)
            i += cap
        return out, lens

    def _process(self, jobs):
        groups = {}
        for jid, (items, task, input_type, dims, fut) in enumerate(jobs):
            groups.setdefault((task, input_type), []).append(jid)
        results = {}
        for (task, input_type), jids in groups.items():
            texts = [t for jid in jids for t in jobs[jid][0]]
            kw = {"task": task, "batch_size": self.encode_batch, "normalize_embeddings": True}
            if task == "retrieval": kw["prompt_name"] = input_type
            try:
                emb, toks = self._encode(texts, task, input_type)
                off = 0
                for jid in jids:
                    k = len(jobs[jid][0]); results[jid] = (emb[off:off + k], toks[off:off + k]); off += k
            except Exception as e:  # fail the whole group, not the process
                self.stats["errors"] += 1
                log("encode error:", repr(e))
                # Hand clients a traceback-free copy: the original's traceback -> this frame -> `results` -> it is a
                # cycle that pins every activation tensor of the failed forward until the next gc pass (after an
                # MPS OOM that left ~22 GB allocated and failed every later request, 2026-10-14 test).
                failure = RuntimeError(f"{type(e).__name__}: {e}")
                for jid in jids: results[jid] = failure
        if any(isinstance(r, Exception) for r in results.values()):
            gc.collect()  # outside the except block (its frame state holds the traceback); then run() empties the MPS cache
        self.stats["batches"] += 1
        for jid, (items, task, input_type, dims, fut) in enumerate(jobs):
            r = results[jid]
            self.stats["requests"] += 1; self.stats["inputs"] += len(items)
            if isinstance(r, Exception):
                self.loop.call_soon_threadsafe(fut.set_exception, r); continue
            emb, toks = r
            if dims and dims < emb.shape[1]:
                emb = emb[:, :dims]
                emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            self.stats["tokens"] += sum(toks)
            self.loop.call_soon_threadsafe(fut.set_result, (emb, toks))

def make_app(args, model, worker, model_id):
    app = web.Application(client_max_size=64 * 1024 * 1024)
    registry = IndexRegistry(args.index_dir, model_id, model.get_sentence_embedding_dimension())
    token = os.environ.get("EMBED_TOKEN", "")
    started = time.time()

    def authed(req):
        if not token: return True
        h = req.headers.get("Authorization", "")
        return h == f"Bearer {token}"

    async def healthz(req):
        return web.json_response({"status": "ok", "model": model_id, "modality": args.modality,
            "device": args.device, "dim": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length, "queue_depth": worker.q.qsize(), "busy": worker.busy,
            "uptime_s": round(time.time() - started, 1), "stats": worker.stats, "auth": bool(token),
            "mps": mps_stats(args.mps_limit_bytes) if args.device == "mps" else None})

    async def models(req):
        if not authed(req): return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        return web.json_response({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "jinaai"}]})

    async def embeddings(req):
        if not authed(req): return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        try: body = await req.json()
        except Exception: return web.json_response({"error": {"message": "invalid JSON"}}, status=400)
        inp = body.get("input")
        if isinstance(inp, str): items = [inp]
        elif isinstance(inp, list) and all(isinstance(x, str) for x in inp): items = inp
        else: return web.json_response({"error": {"message": "input must be a string or list of strings"}}, status=400)
        if not items or len(items) > args.max_inputs:
            return web.json_response({"error": {"message": f"input must have 1..{args.max_inputs} items"}}, status=400)
        if any(len(x) > args.max_chars for x in items):
            return web.json_response({"error": {"message": f"an input exceeds {args.max_chars} chars"}}, status=400)
        task = str(body.get("task", "retrieval")); input_type = str(body.get("input_type", "document"))
        if "." in task:  # Jina API style
            task, sub = task.split(".", 1); input_type = "query" if sub == "query" else "document"
        if task not in TASKS: return web.json_response({"error": {"message": f"task must be one of {sorted(TASKS)}"}}, status=400)
        if input_type not in ("query", "document"): return web.json_response({"error": {"message": "input_type must be query|document"}}, status=400)
        dims = body.get("dimensions")
        if dims is not None and (not isinstance(dims, int) or dims < 32 or dims > model.get_sentence_embedding_dimension()):
            return web.json_response({"error": {"message": "dimensions must be an int in 32..1024"}}, status=400)
        fmt = body.get("encoding_format", "float")
        fut = asyncio.get_running_loop().create_future()
        worker.submit(items, task, input_type, dims, fut)
        try: emb, toks = await asyncio.wait_for(fut, timeout=args.request_timeout)
        except asyncio.TimeoutError: return web.json_response({"error": {"message": "timeout"}}, status=504)
        except Exception as e: return web.json_response({"error": {"message": f"encode failed: {e}"}}, status=500)
        data = []
        for i, v in enumerate(emb):
            val = base64.b64encode(v.astype(np.float32).tobytes()).decode() if fmt == "base64" else v.tolist()
            data.append({"object": "embedding", "index": i, "embedding": val})
        return web.json_response({"object": "list", "data": data, "model": model_id,
            "usage": {"prompt_tokens": sum(toks), "total_tokens": sum(toks)}})


    async def embed(items, task, input_type, dims=None):
        fut = asyncio.get_running_loop().create_future()
        worker.submit(items, task, input_type, dims, fut)
        return await asyncio.wait_for(fut, timeout=args.request_timeout)

    def err(msg, status=400): return web.json_response({"error": {"message": msg}}, status=status)

    async def index_list(req):
        if not authed(req): return err("unauthorized", 401)
        return web.json_response({"namespaces": registry.list()})

    async def index_stats(req):
        if not authed(req): return err("unauthorized", 401)
        try: idx = registry.get(req.match_info["ns"])
        except ValueError as e: return err(str(e))
        if idx is None: return err("no such namespace", 404)
        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, idx.stats))

    async def index_upsert(req):
        if not authed(req): return err("unauthorized", 401)
        try: body = await req.json()
        except Exception: return err("invalid JSON")
        items = body.get("items")
        if not isinstance(items, list) or not items or len(items) > args.max_inputs: return err(f"items must be a list of 1..{args.max_inputs}")
        clean = []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not isinstance(it.get("text"), str) or not it["text"].strip():
                return err("each item needs string id and non-empty text")
            if len(it["text"]) > args.max_chars: return err(f"item {it['id']} exceeds {args.max_chars} chars")
            ts = it.get("ts"); cur = it.get("cursor")
            if ts is not None and not isinstance(ts, (int, float)): return err("ts must be epoch seconds (number)")
            if cur is not None and not isinstance(cur, (int, float)): return err("cursor must be a number")
            clean.append({"id": it["id"], "text": it["text"], "ts": ts, "channel": it.get("channel"), "kind": it.get("kind"),
                          "level": it.get("level"), "meta": it.get("meta") or {}, "cursor": cur})
        try: idx = registry.get(req.match_info["ns"], create=True)
        except ValueError as e: return err(str(e))
        loop = asyncio.get_running_loop()
        to_embed, unchanged = await loop.run_in_executor(None, idx.plan_upsert, clean)
        inserted = updated = 0
        if to_embed:
            try: vecs, toks = await embed([it["text"] for it in to_embed], "retrieval", "document")
            except Exception as e: return err(f"encode failed: {e}", 500)
            inserted, updated = await loop.run_in_executor(None, idx.commit_vectors, to_embed, vecs)
        return web.json_response({"namespace": idx.ns, "inserted": inserted, "updated": updated, "unchanged": len(unchanged), "count": idx.n})

    async def index_delete(req):
        if not authed(req): return err("unauthorized", 401)
        try: body = await req.json()
        except Exception: return err("invalid JSON")
        ids = body.get("ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids): return err("ids must be a list of strings")
        idx = registry.get(req.match_info["ns"])
        if idx is None: return err("no such namespace", 404)
        n = await asyncio.get_running_loop().run_in_executor(None, idx.delete, ids)
        return web.json_response({"deleted": n})

    async def index_search(req):
        if not authed(req): return err("unauthorized", 401)
        try: body = await req.json()
        except Exception: return err("invalid JSON")
        q = body.get("query")
        if not isinstance(q, str) or not q.strip(): return err("query must be a non-empty string")
        idx = registry.get(req.match_info["ns"])
        if idx is None: return err("no such namespace", 404)
        k = body.get("k", 10)
        if not isinstance(k, int) or k < 1 or k > 200: return err("k must be 1..200")
        kinds = body.get("kinds")
        if kinds is not None and (not isinstance(kinds, list) or not all(isinstance(x, str) for x in kinds)): return err("kinds must be a list of strings")
        t0 = time.monotonic()
        try: vecs, toks = await embed([q], "retrieval", "query")
        except Exception as e: return err(f"encode failed: {e}", 500)
        t1 = time.monotonic()
        hits = await asyncio.get_running_loop().run_in_executor(None, lambda: idx.search(
            vecs[0], k=k, ts_from=body.get("ts_from"), ts_to=body.get("ts_to"), channel=body.get("channel"), kinds=kinds,
            level=body.get("level"), min_score=body.get("min_score"), with_text=body.get("with_text", True), snippet=body.get("snippet", 400)))
        return web.json_response({"namespace": idx.ns, "hits": hits, "count_indexed": int(idx.alive[:idx.n].sum()) if idx.n else 0,
                                  "timing_ms": {"embed": round(1000 * (t1 - t0)), "search": round(1000 * (time.monotonic() - t1))}})

    app.add_routes([web.get("/v1/index", index_list), web.get("/v1/index/{ns}/stats", index_stats),
                    web.post("/v1/index/{ns}/upsert", index_upsert), web.post("/v1/index/{ns}/delete", index_delete),
                    web.post("/v1/index/{ns}/search", index_search)])
    app.add_routes([web.get("/healthz", healthz), web.get("/v1/models", models), web.post("/v1/embeddings", embeddings)])
    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v5-omni-small")
    ap.add_argument("--modality", default="text", choices=["text", "vision", "audio", "omni"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8804)
    ap.add_argument("--batch-window-ms", type=float, default=15.0)
    ap.add_argument("--max-batch", type=int, default=256, help="max inputs gathered per worker cycle")
    ap.add_argument("--encode-batch", type=int, default=64, help="max items per model.encode sub-batch")
    ap.add_argument("--token-budget", type=int, default=24576, help="max (longest_len x items) tokens per encode sub-batch")
    ap.add_argument("--attn-budget", type=int, default=8192 * 8192,
                    help="max items x longest_len^2 per encode sub-batch (bounds the materialised attention scores); 0 = off")
    ap.add_argument("--max-inputs", type=int, default=256); ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--index-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"))
    ap.add_argument("--mps-mem-limit-gb", type=float, default=24.0,
                    help="MPS allocator ceiling (weights + activations + cache); over it the allocator frees its cache, "
                         "then fails the batch with OOM instead of growing. 0 = torch default (~1.7x recommended working set)")
    ap.add_argument("--no-release-cache", action="store_true", help="keep pooled MPS buffers between worker cycles")
    ap.add_argument("--no-length-buckets", action="store_true", help="pad to the batch's longest member, not to a bucket")
    args = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    args.mps_limit_bytes = 0
    if args.device == "mps" and args.mps_mem_limit_gb > 0:
        rec = torch.mps.recommended_max_memory()
        args.mps_limit_bytes = int(args.mps_mem_limit_gb * 2**30)
        torch.mps.set_per_process_memory_fraction(min(2.0, args.mps_limit_bytes / rec))
    t = time.time()
    kw = {"torch_dtype": getattr(torch, args.dtype)}
    if "omni" in args.model: kw["modality"] = args.modality
    model = SentenceTransformer(args.model, trust_remote_code=True, device=args.device, model_kwargs=kw)
    model.max_seq_length = args.max_seq_length
    if "omni" in args.model and args.modality == "text": pin_text_only(model)
    pad_side = None if args.no_length_buckets else install_length_buckets(model, args.max_seq_length)
    model.encode(["warm up"], task="retrieval", prompt_name="query")
    log(f"loaded {args.model} modality={args.modality} device={args.device} dtype={args.dtype} "
        f"max_seq={model.max_seq_length} dim={model.get_sentence_embedding_dimension()} in {time.time()-t:.1f}s; "
        f"text_only={'omni' in args.model and args.modality == 'text'} length_buckets={pad_side or 'off'} "
        f"release_cache={args.device == 'mps' and not args.no_release_cache} mps_limit_gb={args.mps_mem_limit_gb if args.mps_limit_bytes else 'default'}")
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    worker = Worker(model, loop, args.batch_window_ms, args.max_batch, args.encode_batch, args.token_budget,
                    attn_budget=args.attn_budget, release_cache=args.device == "mps" and not args.no_release_cache, bucketed=pad_side is not None); worker.start()
    app = make_app(args, model, worker, args.model)
    log(f"listening on {args.host}:{args.port} auth={'on' if os.environ.get('EMBED_TOKEN') else 'OFF'}")
    web.run_app(app, host=args.host, port=args.port, print=None, loop=loop, access_log=None)

if __name__ == "__main__":
    main()
