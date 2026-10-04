# Text Summarizer

Summarize articles, documents and chats with **classical NLP algorithms implemented from scratch** and an **LSTM pointer-generator network** trained from scratch. No pretrained transformers. It includes a FastAPI backend, a lightweight web UI, and a reproducible ROUGE benchmark suite.

![CI](https://github.com/aldol07/Text-Summarizer/actions/workflows/ci.yml/badge.svg)

| | |
|---|---|
| **Extractive** | TextRank, LexRank, LSA, TF-IDF centroid (MEAD), Luhn, plus MMR redundancy control and a lead-position prior. PageRank, TF-IDF and SVD scoring are written in NumPy. |
| **Abstractive** | BiLSTM encoder + Bahdanau attention + pointer (copy) mechanism + coverage (See et al., 2017), trained on SAMSum. Beam search with trigram blocking. |
| **Evaluation** | ROUGE-1/2/L/Lsum implemented from scratch (exact match with Google's `rouge-score`), bootstrap confidence intervals, paired significance tests, greedy extractive oracle. |
| **Serving** | FastAPI + ONNX Runtime (int8). The LSTM runs **without PyTorch** in ~230 MB of RAM. |
| **Input** | Paste text, fetch a URL (SSRF-protected), or upload a PDF / TXT / Markdown file. |

---

## Results

All numbers below were measured in this repository (`reports/`) and are ROUGE F1 × 100.

| Task | Our best system | ROUGE-1 | Baseline |
|---|---|---|---|
| Chat summarization (SAMSum, 819 test dialogues) | LSTM pointer-generator + coverage | **39.89** | Lead-3: 31.44 (**+8.45**, p < 0.05) |
| News summarization (CNN/DailyMail, 1,000 test articles) | TextRank + position prior + MMR | **39.32** | TextRank alone: 36.15 (**+3.17**) |

Sanity checks: our Lead-3 reproduces published numbers (CNN/DailyMail 40.58 vs ~40.4 in See et al., 2017; SAMSum 31.44 vs 31.40 in Gliwa et al., 2019). Our ROUGE implementation matches `rouge-score` on all 1,200 scores we cross-checked.

### SAMSum: extractive vs abstractive

Same 819 test dialogues, all systems on CPU (1 thread).

| System | R-1 | R-2 | R-L | R-Lsum | Latency (mean) |
|---|---|---|---|---|---|
| Lead-3 | 31.44 | 8.75 | 24.24 | 29.53 | 2 ms |
| TextRank | 26.77 | 6.51 | 21.22 | 25.19 | 3 ms |
| TF-IDF centroid (best extractive) | 32.29 | 10.35 | 24.70 | 29.90 | 2 ms |
| LSTM seq2seq + attention | 33.93 | 12.66 | 28.81 | 30.76 | 200 ms |
| LSTM + pointer | 38.83 | 16.43 | 33.10 | 34.89 | 180 ms |
| **LSTM + pointer + coverage** | **39.89** | **17.01** | **33.33** | **35.49** | 202 ms |
| LSTM + pointer + coverage, **int8 ONNX (deployed)** | 39.76 | 16.78 | 33.14 | 35.34 | 147 ms |
| *Oracle (extractive upper bound)* | *49.15* | *19.78* | *38.88* | *45.71* | |

The copy mechanism adds **+4.9 ROUGE-1** (names and rare words are copied from the chat), and coverage adds another **+1.05** by reducing repetition. int8 quantization costs only 0.13 ROUGE-1.

![SAMSum results](reports/samsum_abstractive/rouge_chart.png)

### CNN/DailyMail: extractive methods

1,000 randomly sampled test articles (seed 42), 3 sentences per summary.

| System | R-1 | R-2 | R-L | R-Lsum | Latency (mean) |
|---|---|---|---|---|---|
| Random | 29.15 | 8.51 | 17.93 | 26.17 | 5 ms |
| Lead-3 | 40.58 | 17.70 | 25.14 | 36.64 | 4 ms |
| Luhn | 35.87 | 14.33 | 23.18 | 32.24 | 5 ms |
| TF-IDF centroid | 35.47 | 14.01 | 22.55 | 31.79 | 5 ms |
| LSA | 33.10 | 12.51 | 20.94 | 29.44 | 10 ms |
| LexRank | 35.75 | 13.46 | 22.40 | 32.10 | 6 ms |
| TextRank | 36.15 | 14.43 | 23.18 | 32.48 | 6 ms |
| **TextRank + position + MMR** | **39.32** | **16.71** | **24.87** | **35.49** | 4 ms |
| *Oracle (extractive upper bound)* | *57.09* | *33.19* | *39.62* | *52.71* | |

News articles put the key facts first ("lead bias"), so Lead-3 is famously hard to beat without supervision. A small position prior plus MMR closes most of that gap while still working on documents that aren't news.

### Ablations (CNN/DailyMail)

| Change (vs plain TextRank, 36.15) | R-1 | Δ |
|---|---|---|
| + MMR redundancy control | 36.93 | +0.78 |
| + lead-position prior (0.3) | 38.12 | +1.97 |
| + position prior + MMR | **39.32** | **+3.17** |
| Cosine edges instead of word overlap | 36.19 | +0.04 |
| WordNet lemmatization instead of Porter stemming | 36.03 | −0.12 |
| No stemming / lemmatization | 36.24 | +0.09 |
| Keep stopwords | 33.82 | −2.33 |

![CNN/DailyMail ablations](reports/cnn_dailymail_ablations/rouge_chart.png)

Full tables with 95% confidence intervals, coverage, redundancy and latency are in [`reports/`](reports/).

### Serving footprint (LSTM)

Measured in a fresh process, CPU.

| Backend | Peak memory | Model size | Same output as PyTorch |
|---|---|---|---|
| PyTorch | 745 MB | 32 MB | (reference) |
| ONNX fp32 | 258 MB | 39 MB | 50 / 50 summaries |
| **ONNX int8** | **230 MB** | **9.8 MB** | 38 / 50 (−0.13 ROUGE-1) |

Exporting to ONNX is what lets the model fit a 512 MB free-tier server.

---

## How it works

```mermaid
flowchart LR
    A[Text / URL / PDF] --> B{Chat?}
    B -- yes --> C[LSTM pointer-generator<br/>ONNX Runtime, beam search]
    B -- no --> D[Preprocess: clean, split sentences,<br/>tokenize, stem, stopwords]
    D --> E[TextRank: sentence graph + PageRank]
    E --> F[Position prior + MMR selection]
    C --> G[Summary + keywords + stats]
    F --> G
```

- **Preprocessing:** rule-based sentence splitter (handles "Dr.", "U.S.", initials and decimals), Porter stemming or WordNet lemmatization, built-in stopword list. It works without downloading NLTK data.
- **Extractive scoring:** every algorithm only scores sentences. Selection (length, MMR, position prior) is shared, so the methods are compared fairly.
- **Pointer-generator:** at each step the model mixes "generate a word from the vocabulary" with "copy a word from the input" (`p_gen`). Coverage tracks what has already been attended to and penalizes repeated attention. Beam search is written in NumPy over a small backend interface, so PyTorch and ONNX Runtime decode with the same code.
- **Keywords:** TF-IDF terms and RAKE key phrases.

---

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/aldol07/Text-Summarizer.git
cd Text-Summarizer
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Web app + API

```bash
uvicorn textSummarizer.api.main:app --port 8010
```

Open http://127.0.0.1:8010 for the UI or http://127.0.0.1:8010/docs for the interactive API docs.

The extractive summarizers work out of the box. The chat (LSTM) summarizer needs the trained ONNX model in `artifacts/onnx/pointer_coverage/int8/` (see [Train the LSTM](#train-the-lstm)). Without it, chats fall back to the best extractive method.

### Command line

```bash
python -m textSummarizer.cli examples/sample_article.txt
python -m textSummarizer.cli examples/sample_article.txt -m lexrank -n 2 --mmr
python -m textSummarizer.cli examples/sample_article.txt --compare
python -m textSummarizer.cli --list-methods
```

### API examples

```bash
curl -X POST http://127.0.0.1:8010/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Sam: Are you coming to dinner on Saturday?\nLucy: Of course! What time?\nSam: 7 pm at the Italian place.\"}"
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, and whether the LSTM is loaded |
| `GET /api/v1/methods` | Available methods |
| `POST /api/v1/summarize` | `text`, optional `method` (default `auto`), `mode` (`auto` / `article` / `dialogue`), `num_sentences` or `ratio`, `use_mmr`, `position_weight`, `reference` (adds ROUGE) |
| `POST /api/v1/compare` | Every method on the same text, with latency and optional ROUGE |
| `POST /api/v1/extract/url` | Main text of a web page |
| `POST /api/v1/extract/file` | Text of a PDF / TXT / Markdown upload (max 5 MB) |

`method=auto` uses the LSTM for chats and TextRank + position prior + MMR for everything else.

### Configuration

| Environment variable | Default |
|---|---|
| `LSTM_MODEL_DIR` | `artifacts/onnx/pointer_coverage/int8` |
| `ENABLE_ABSTRACTIVE` | `true` |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` (comma-separated) |
| `MAX_TEXT_CHARS` | `50000` |
| `MAX_UPLOAD_MB` | `5` |
| `SERVE_FRONTEND` | `true` |
| `ONNX_THREADS` | `1` |

The frontend (`frontend/`) is plain HTML/CSS/JS. To host it separately, set `API_BASE_URL` in `frontend/config.js`.

---

## Reproduce the results

```bash
pip install -r requirements-dev.txt
python scripts/download_nltk.py          # optional: Punkt + WordNet
```

### Benchmarks

Datasets are downloaded from the Hugging Face Hub on first use (CNN/DailyMail 3.0.0 test split, SAMSum).

```bash
python scripts/benchmark.py --dataset cnn_dailymail --limit 1000
python scripts/benchmark.py --dataset cnn_dailymail --suite ablations
python scripts/benchmark.py --dataset samsum
python scripts/benchmark.py --dataset samsum --suite abstractive     # after training + export
python scripts/benchmark.py --dataset my_docs.jsonl                  # your own {"id","text","summary"} lines
```

Experiments are defined in [`config/benchmark.yaml`](config/benchmark.yaml). Each run writes a Markdown table, CSV/JSON and a chart to `reports/`.

### Train the LSTM

A GPU helps but isn't required. On an RTX 5060 laptop GPU the full model trains in ~32 minutes.

```bash
python scripts/train_lstm.py --variant pointer_coverage     # full model
python scripts/train_lstm.py --variant pointer              # ablation
python scripts/train_lstm.py --variant seq2seq_attention    # ablation
python scripts/export_onnx.py                               # ONNX fp32 + int8, parity + memory report
```

The pipeline runs five stages (ingest → validate → transform → train → evaluate). Hyperparameters are in [`config/lstm.yaml`](config/lstm.yaml). Models are written to `artifacts/` and are not committed.

---

## Project structure

```
src/textSummarizer/
├── preprocessing/   cleaning, sentence segmentation, tokenization, stemming/lemmatization
├── extractive/      TextRank, LexRank, LSA, TF-IDF, Luhn, baselines, MMR, PageRank, TF-IDF vectorizer
├── abstractive/     vocabulary, pointer-generator model, trainer, beam search, PyTorch/ONNX backends, export
├── evaluation/      ROUGE, oracle, bootstrap/significance tests, datasets, benchmark runner, reports
├── features/        keywords (TF-IDF, RAKE), readability stats, dialogue detection
├── api/             FastAPI app, request schemas, URL/PDF ingestion, settings
├── service.py       one-call summarize/compare used by the CLI and the API
└── cli.py
frontend/            web UI (no build step)
scripts/             benchmark, train, export, NLTK download
config/              benchmark and LSTM configuration
reports/             benchmark results and charts
```

---

## Limitations

- The LSTM is a small model (8M parameters) trained on 14.7k chats. Its summaries are fluent for everyday conversations but can mix up who did what. It is not reliable on long or non-chat text, which is why `auto` uses it only for chats.
- Extractive summaries reuse the original sentences, so they are faithful to the source but can be less concise.
- URL extraction is heuristic (no JavaScript rendering). Scanned PDFs need OCR, which isn't supported.

## References

- Mihalcea & Tarau (2004). *TextRank: Bringing Order into Texts.*
- Erkan & Radev (2004). *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization.*
- Steinberger & Ježek (2004). *Using Latent Semantic Analysis in Text Summarization.*
- Radev et al. (2004). *Centroid-based summarization of multiple documents* (MEAD).
- Luhn (1958). *The Automatic Creation of Literature Abstracts.*
- Carbonell & Goldstein (1998). *The Use of MMR, Diversity-Based Reranking.*
- See, Liu & Manning (2017). *Get To The Point: Summarization with Pointer-Generator Networks.*
- Gliwa et al. (2019). *SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization.*
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*

## License

MIT. See [LICENSE](LICENSE).
