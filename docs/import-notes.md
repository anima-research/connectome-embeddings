# Initial import — 2026-10-19

Source: the existing `~/embed-service` deployment on the operator's Mac. The three files below were copied explicitly, without copying their directory or its history. No live service restart, model load or resident-data read was performed to prepare this import.

```json
{
  "server.py": {
    "bytes": 23852,
    "sha256": "b00eddda0c49bcbc61c607713aa52697d7081c0eb084cd964f8a7f50eb0391da"
  },
  "index_store.py": {
    "bytes": 12346,
    "sha256": "159fe47d6bb1752a8a72788eae7dc9440c9f6aee8cc2d470b02ac4c50511f116"
  },
  "run.sh": {
    "bytes": 467,
    "sha256": "828a74f61747e3eb0ff98beb60140671a097f1b4c48a99e692caf8537eedaa54"
  }
}
```

Dependency versions were read from installed distribution metadata, without importing model packages. README and the launchd example are newly prepared documentation, not a verbatim copy of private operational notes. Benchmark claims in the original local README were not independently rerun.

Excluded: `.env`, `index/` (resident text and vectors), `hf-home/`, logs, backups, and resident-specific one-off `tools/` scripts. Existing service definitions and source files remain unchanged. The source remains unmodified at runtime; future deployment should be an explicit operator decision, not a side effect of this import.
