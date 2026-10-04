---
title: NotebookLM Clone
emoji: 📓
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# NotebookLM / Gemini Notebook Clone

A small NotebookLM-style app (CS/IT 5010 project). Create notebooks, add sources (PDF, PPTX, TXT, web URL), chat with them using RAG with visible citations, and generate a Report or a Quiz (with answer key) as downloadable Markdown.

- **Live demo:** _add HF Space URL here_
- **Docs:** [Spec](SPEC.md) · [Architecture](docs/ARCHITECTURE.md) · [RAG evaluation](docs/EVALUATION.md) · [Demo checklist](docs/DEMO_CHECKLIST.md)

## Stack
Gradio UI · ChromaDB (one collection per notebook) · `sentence-transformers/all-MiniLM-L6-v2` embeddings · cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) · Groq-hosted LLM (`openai/gpt-oss-120b`).

## Run locally
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env
python app.py               # http://localhost:7860
```
Tests (offline, no key needed): `pip install -r requirements-dev.txt && python -m pytest tests -q`
RAG evaluation (needs the key): `python -m eval.run_eval`

## Configuration (environment variables)
| Variable | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | yes | none |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b` |

Never commit keys. `.env` and `data/` are gitignored. On Hugging Face set `GROQ_API_KEY` under Space settings -> Secrets.

## Using the app
1. Create a notebook (top bar), or pick an existing one.
2. **Sources** tab: upload PDF/PPTX/TXT or paste a URL. The table shows status and chunk counts; failures show the error.
3. **Chat** tab: ask questions. Choose `rerank` (vector + cross-encoder) or `vector`. Citations (source, page/slide, chunk, snippet) appear under the answer. History is saved per notebook.
4. **Artifacts** tab: generate a Report or Quiz, view it, and download the `.md`.

## Storage and persistence
Everything for a notebook lives in `data/notebooks/<notebook_id>/` (`info.json`, `chats.json`, `raw/`, `text/`, `chroma/`, `artifacts/`) behind the `StorageManager` class.

**Hugging Face limitation:** on the free Space tier the disk is ephemeral. Data survives normal app restarts but **is lost when the Space rebuilds** (every deploy from GitHub) or is recreated. Persistence on HF is best effort; this is a deliberate simplification. Details and upgrade options are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#persistence-limits).

## CI/CD
`.github/workflows/deploy.yml`: on push to `main`, run tests, then force-push to the HF Space git remote.

One-time setup:
1. Create a Gradio Space on Hugging Face.
2. GitHub repo -> Settings -> Secrets and variables -> Actions:
   - **Secret** `HF_TOKEN`: HF access token with write permission.
   - **Variables** `HF_USERNAME` and `HF_SPACE`: your HF username and Space name.
3. In the Space settings add the secret `GROQ_API_KEY`.

## Project layout
```
app.py            entry point        ui/          Gradio wiring
storage/          notebook storage   ingestion/   extract, chunk, embed, index
retrieval/        embedder, Chroma, reranker      generation/  chat, artifacts, LLM client
eval/             retrieval comparison            tests/       offline tests
docs/             architecture, evaluation, demo checklist
```

## Limitations
- Scanned/image-only PDFs have no extractable text and fail with a clear error (no OCR).
- Web pages that need JavaScript to render are not supported.
- Ephemeral storage on the free HF tier (see above).
