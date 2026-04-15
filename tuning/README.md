# Hyperparameter Tuning

Automated grid sweep over `chunk_size`, `chunk_overlap`, `chat_top_k`, and `autocomplete_top_k`.

## How it works

1. For each `(chunk_size, chunk_overlap)` pair, the PDF is re-chunked and a fresh FAISS index is built (one embedding call per index).
2. For each `(chat_top_k, ac_top_k)` pair on that same index, QA and autocomplete eval is run in-process — no live server needed.
3. Results are ranked by a composite score and written to `tuning/reports/`.

**Composite score formula:**
```
composite = 0.6 × avg_groundedness
          + 0.2 × avg_keyword_recall
          + 0.2 × avg_ac_best_similarity
```

## Prerequisites

- Ollama running locally (`http://localhost:11434`)
- The `embeddinggemma` and `gemma3` models pulled in Ollama
- Python environment with project dependencies installed (same venv as the main app)

## Usage

Run from the project root:

```bash
python tuning/sweep.py \
    --pdf eval/data/sample_policy.pdf \
    --qa eval/data/qa_dataset.jsonl \
    --autocomplete eval/data/autocomplete_dataset.jsonl
```

**Custom search space:**

```bash
python tuning/sweep.py \
    --pdf eval/data/sample_policy.pdf \
    --qa eval/data/qa_dataset.jsonl \
    --autocomplete eval/data/autocomplete_dataset.jsonl \
    --chunk-sizes 512 768 900 1200 \
    --chunk-overlaps 100 150 200 \
    --chat-top-ks 3 4 6 \
    --ac-top-ks 2 3 4
```

**Custom output directory:**

```bash
python tuning/sweep.py ... --out tuning/reports/my_run
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--pdf` | _(required)_ | Reference PDF to evaluate against |
| `--qa` | _(required)_ | QA dataset JSONL |
| `--autocomplete` | _(required)_ | Autocomplete dataset JSONL |
| `--out` | `tuning/reports/` | Output directory for reports |
| `--chunk-sizes` | `512 768 900 1200` | List of chunk sizes to try |
| `--chunk-overlaps` | `100 150` | List of overlap values to try |
| `--chat-top-ks` | `3 4 6` | List of chat top_k values to try |
| `--ac-top-ks` | `2 3` | List of autocomplete top_k values to try |

## Outputs

Each run writes two files to the output directory:

- `sweep_YYYYMMDD_HHMMSS.md` — Ranked markdown report with a table of all combos and a "Best configuration" block ready to paste into `config.py`.
- `sweep_YYYYMMDD_HHMMSS.json` — Raw JSON with all metrics per combination, for programmatic post-processing.

## Applying the best result

After the sweep, copy the best config block from the report directly into `config.py`:

```python
# config.py
CHUNK_SIZE         = 900   # ← replace with sweep winner
CHUNK_OVERLAP      = 150
CHAT_TOP_K         = 4
AUTOCOMPLETE_TOP_K = 3
```

## Cost estimate

- **Embedding cost:** one batch embed call per `(chunk_size, chunk_overlap)` combo.
- **LLM inference cost:** one call per `(qa_sample + ac_sample)` per `(chat_top_k, ac_top_k)` combo.
- With default settings (4×2 chunk combos, 3×2 top_k combos) this is **8 index builds** and **48 inference combos × dataset size** LLM calls.
