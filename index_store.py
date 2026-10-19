"""Per-namespace vector index for embed-service.

SQLite is the source of truth (one file per namespace under --index-dir): id, text, sha, ts, channel,
kind, level, meta(JSON), vec(BLOB f16), deleted. An in-memory f32 matrix + parallel metadata arrays are
rebuilt lazily from SQLite on first use and appended on upsert, so search is one matmul over normalised
vectors (cosine) followed by a boolean filter mask and a top-k. Brute force: 100k x 1024 f32 is ~0.4 GB
and a few ms per query on this box; shard/ANN only if a namespace outgrows that.
"""
import hashlib, json, os, re, sqlite3, threading, time
import numpy as np

SAFE = re.compile(r"[^A-Za-z0-9._@:-]")

def ns_file(index_dir, ns):
    return os.path.join(index_dir, SAFE.sub("_", ns) + ".sqlite")

class NamespaceIndex:
    def __init__(self, path, ns, model_id, dim):
        self.path, self.ns, self.model_id, self.dim = path, ns, model_id, dim
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS items(
            row INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, text TEXT NOT NULL, sha TEXT NOT NULL,
            ts REAL, channel TEXT, kind TEXT, level INTEGER, meta TEXT, vec BLOB NOT NULL, deleted INTEGER DEFAULT 0,
            updated REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS items_ts ON items(ts)")
        if "cursor" not in [r[1] for r in self.db.execute("PRAGMA table_info(items)").fetchall()]:
            self.db.execute("ALTER TABLE items ADD COLUMN cursor REAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS info(k TEXT PRIMARY KEY, v TEXT)")
        cur = self.db.execute("SELECT v FROM info WHERE k='model'").fetchone()
        if cur is None:
            self.db.execute("INSERT INTO info VALUES('model',?),('dim',?),('ns',?)", (model_id, str(dim), ns)); self.db.commit()
        elif cur[0] != model_id:
            raise ValueError(f"namespace {ns} was built with {cur[0]}, service runs {model_id}")
        self._loaded = False

    # ---- in-memory mirror -------------------------------------------------
    def _load(self):
        if self._loaded: return
        rows = self.db.execute("SELECT row,id,ts,channel,kind,level,deleted,vec,cursor FROM items ORDER BY row").fetchall()
        n = len(rows)
        self.ids = [r[1] for r in rows]; self.row_of = {r[1]: i for i, r in enumerate(rows)}
        self.ts = np.array([r[2] if r[2] is not None else np.nan for r in rows], dtype=np.float64)
        self.channel = [r[3] for r in rows]; self.kind = [r[4] for r in rows]
        self.level = np.array([r[5] if r[5] is not None else -1 for r in rows], dtype=np.int32)
        self.alive = np.array([not r[6] for r in rows], dtype=bool)
        self.cursor = np.array([r[8] if r[8] is not None else np.nan for r in rows], dtype=np.float64)
        self.M = np.zeros((max(n, 1), self.dim), dtype=np.float32)
        if n: self.M[:n] = np.frombuffer(b"".join(r[7] for r in rows), dtype=np.float16).reshape(n, self.dim).astype(np.float32)
        self.n = n; self._loaded = True

    def _grow(self, k):
        need = self.n + k
        if need > self.M.shape[0]:
            newM = np.zeros((max(need, int(self.M.shape[0] * 1.5)), self.dim), dtype=np.float32)
            newM[:self.n] = self.M[:self.n]; self.M = newM
        # metadata arrays grow with concatenate (cheap relative to embedding)
    # ---- API ---------------------------------------------------------------
    @staticmethod
    def sha(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def plan_upsert(self, items):
        """Return (to_embed, unchanged, changed) — which items need a fresh vector."""
        with self.lock:
            self._load()
            to_embed, unchanged = [], []
            for it in items:
                s = self.sha(it["text"])
                r = self.db.execute("SELECT sha, deleted FROM items WHERE id=?", (it["id"],)).fetchone()
                if r and r[0] == s and not r[1]:
                    # metadata-only refresh
                    self._update_meta(it); unchanged.append(it["id"])
                else:
                    it["_sha"] = s; to_embed.append(it)
            self.db.commit()
            return to_embed, unchanged

    def _update_meta(self, it):
        self.db.execute("UPDATE items SET ts=?, channel=?, kind=?, level=?, meta=?, cursor=?, updated=? WHERE id=?",
            (it.get("ts"), it.get("channel"), it.get("kind"), it.get("level"), json.dumps(it.get("meta") or {}), it.get("cursor"), time.time(), it["id"]))
        i = self.row_of.get(it["id"])
        if i is not None:
            self.ts[i] = it.get("ts") if it.get("ts") is not None else np.nan
            self.channel[i] = it.get("channel"); self.kind[i] = it.get("kind")
            self.level[i] = it.get("level") if it.get("level") is not None else -1
            self.cursor[i] = it.get("cursor") if it.get("cursor") is not None else np.nan

    def commit_vectors(self, items, vecs):
        """items: the to_embed list from plan_upsert, vecs: (len(items), dim) normalised f32."""
        with self.lock:
            self._load()
            now = time.time()
            new_rows = []
            for it, v in zip(items, vecs):
                blob = v.astype(np.float16).tobytes()
                i = self.row_of.get(it["id"])
                if i is not None:  # changed text: overwrite in place
                    self.db.execute("UPDATE items SET text=?, sha=?, ts=?, channel=?, kind=?, level=?, meta=?, vec=?, cursor=?, deleted=0, updated=? WHERE id=?",
                        (it["text"], it["_sha"], it.get("ts"), it.get("channel"), it.get("kind"), it.get("level"),
                         json.dumps(it.get("meta") or {}), blob, it.get("cursor"), now, it["id"]))
                    self.cursor[i] = it.get("cursor") if it.get("cursor") is not None else np.nan
                    self.M[i] = v; self.alive[i] = True
                    self.ts[i] = it.get("ts") if it.get("ts") is not None else np.nan
                    self.channel[i] = it.get("channel"); self.kind[i] = it.get("kind")
                    self.level[i] = it.get("level") if it.get("level") is not None else -1
                else:
                    self.db.execute("INSERT INTO items(id,text,sha,ts,channel,kind,level,meta,vec,cursor,deleted,updated) VALUES(?,?,?,?,?,?,?,?,?,?,0,?)",
                        (it["id"], it["text"], it["_sha"], it.get("ts"), it.get("channel"), it.get("kind"), it.get("level"),
                         json.dumps(it.get("meta") or {}), blob, it.get("cursor"), now))
                    new_rows.append((it, v))
            if new_rows:
                self._grow(len(new_rows))
                k = len(new_rows)
                self.M[self.n:self.n + k] = np.stack([v for _, v in new_rows])
                self.ts = np.concatenate([self.ts, np.array([it.get("ts") if it.get("ts") is not None else np.nan for it, _ in new_rows], dtype=np.float64)])
                self.level = np.concatenate([self.level, np.array([it.get("level") if it.get("level") is not None else -1 for it, _ in new_rows], dtype=np.int32)])
                self.alive = np.concatenate([self.alive, np.ones(k, dtype=bool)])
                self.cursor = np.concatenate([self.cursor, np.array([it.get("cursor") if it.get("cursor") is not None else np.nan for it, _ in new_rows], dtype=np.float64)])
                for it, _ in new_rows:
                    self.row_of[it["id"]] = self.n; self.ids.append(it["id"]); self.channel.append(it.get("channel")); self.kind.append(it.get("kind")); self.n += 1
            self.db.commit()
            return len(new_rows), len(items) - len(new_rows)

    def delete(self, ids):
        with self.lock:
            self._load(); n = 0
            for i_ in ids:
                i = self.row_of.get(i_)
                if i is not None and self.alive[i]:
                    self.alive[i] = False; n += 1
                    self.db.execute("UPDATE items SET deleted=1, updated=? WHERE id=?", (time.time(), i_))
            self.db.commit(); return n

    def search(self, qvec, k=10, ts_from=None, ts_to=None, channel=None, kinds=None, level=None, min_score=None, with_text=True, snippet=400):
        with self.lock:
            self._load()
            if self.n == 0: return []
            mask = self.alive[:self.n].copy()
            if ts_from is not None: mask &= (self.ts[:self.n] >= ts_from)
            if ts_to is not None: mask &= (self.ts[:self.n] <= ts_to)
            if channel is not None: mask &= np.array([c == channel for c in self.channel[:self.n]], dtype=bool)
            if kinds: mask &= np.array([kd in kinds for kd in self.kind[:self.n]], dtype=bool)
            if level is not None: mask &= (self.level[:self.n] == level)
            if not mask.any(): return []
            scores = self.M[:self.n] @ qvec.astype(np.float32)
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
            top = np.argpartition(-scores, k - 1)[:k]; top = top[np.argsort(-scores[top])]
            out = []
            for i in top:
                if not np.isfinite(scores[i]): continue
                if min_score is not None and scores[i] < min_score: continue
                r = self.db.execute("SELECT text, meta FROM items WHERE id=?", (self.ids[i],)).fetchone()
                text = r[0] if r else ""
                hit = {"id": self.ids[i], "score": round(float(scores[i]), 4), "ts": None if np.isnan(self.ts[i]) else float(self.ts[i]),
                       "channel": self.channel[i], "kind": self.kind[i], "level": int(self.level[i]) if self.level[i] >= 0 else None,
                       "meta": json.loads(r[1]) if r and r[1] else {}}
                if with_text: hit["text"] = text if snippet is None or len(text) <= snippet else text[:snippet] + "…"
                hit["chars"] = len(text)
                out.append(hit)
            return out

    def stats(self):
        with self.lock:
            self._load()
            alive = self.alive[:self.n]
            by_kind = {}
            for kd, a, t, c in zip(self.kind[:self.n], alive, self.ts[:self.n], self.cursor[:self.n]):
                if not a: continue
                d = by_kind.setdefault(kd or "?", {"count": 0, "max_ts": None, "min_ts": None, "max_cursor": None})
                d["count"] += 1
                if not np.isnan(c): d["max_cursor"] = c if d["max_cursor"] is None else max(d["max_cursor"], c)
                if not np.isnan(t):
                    d["max_ts"] = t if d["max_ts"] is None else max(d["max_ts"], t)
                    d["min_ts"] = t if d["min_ts"] is None else min(d["min_ts"], t)
            return {"namespace": self.ns, "model": self.model_id, "dim": self.dim, "count": int(alive.sum()),
                    "rows": self.n, "by_kind": by_kind, "file_bytes": os.path.getsize(self.path)}

class IndexRegistry:
    def __init__(self, index_dir, model_id, dim):
        os.makedirs(index_dir, exist_ok=True)
        self.dir, self.model_id, self.dim = index_dir, model_id, dim
        self.lock = threading.Lock(); self.open = {}

    def get(self, ns, create=False):
        if not ns or len(ns) > 200 or ns.startswith("."): raise ValueError("bad namespace")
        with self.lock:
            if ns in self.open: return self.open[ns]
            path = ns_file(self.dir, ns)
            if not os.path.exists(path) and not create: return None
            idx = NamespaceIndex(path, ns, self.model_id, self.dim); self.open[ns] = idx; return idx

    def list(self):
        with self.lock:
            out = []
            for f in sorted(os.listdir(self.dir)):
                if f.endswith(".sqlite"):
                    p = os.path.join(self.dir, f)
                    try:
                        c = sqlite3.connect(p); r = c.execute("SELECT v FROM info WHERE k='ns'").fetchone(); n = c.execute("SELECT count(*) FROM items WHERE deleted=0").fetchone()[0]; c.close()
                    except Exception: r, n = None, None
                    ns = r[0] if r else f[:-7]
                    out.append({"namespace": ns, "count": n, "file_bytes": os.path.getsize(p), "open": ns in self.open})
            return out
