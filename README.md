# NotebookLM / Gemini Notebook Clone

A small NotebookLM-style app (CS/IT 5010 project). Create notebooks, add sources (PDF, PPTX, TXT, web URL), chat with them using RAG with visible citations, and generate a Report or a Quiz (with answer key) as downloadable Markdown.

- **Live demo:** _add Cloud Run URL here_
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

Never commit keys. `.env` and `data/` are gitignored. On Cloud Run set `GROQ_API_KEY` as an environment variable or Secret Manager secret (see CI/CD below).

## Using the app
1. Create a notebook (top bar), or pick an existing one.
2. **Sources** tab: upload PDF/PPTX/TXT or paste a URL. The table shows status and chunk counts; failures show the error.
3. **Chat** tab: ask questions. Choose `rerank` (vector + cross-encoder) or `vector`. Citations (source, page/slide, chunk, snippet) appear under the answer. History is saved per notebook.
4. **Artifacts** tab: generate a Report or Quiz, view it, and download the `.md`.

## Storage and persistence
Everything for a notebook lives in `data/notebooks/<notebook_id>/` (`info.json`, `chats.json`, `raw/`, `text/`, `chroma/`, `artifacts/`) behind the `StorageManager` class.

**Cloud Run limitation:** the container disk is ephemeral. Data survives while an instance is running but **is lost when the instance is replaced** (idle scale-to-zero, every deploy, or a crash). Persistence is best effort; this is a deliberate simplification. The service is capped at one instance so a session never splits across instances. Options for real persistence (not implemented): mount a Cloud Storage bucket at `data/`, or use Filestore. Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#persistence-limits).

## Docker
```bash
docker build -t notebooklm-clone .
docker run --rm -p 8080:8080 -e PORT=8080 -e GROQ_API_KEY=... notebooklm-clone   # http://localhost:8080
```

## CI/CD (GitHub Actions -> Google Cloud Run)
`.github/workflows/deploy.yml`: on push to `main`, run tests, build the image, push it to Artifact Registry, and deploy to Cloud Run.

One-time GCP setup (replace `PROJECT` and region as needed):
```bash
gcloud config set project PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create notebooklm --repository-format=docker --location=us-central1

# deploy service account
gcloud iam service-accounts create gh-deployer
SA=gh-deployer@PROJECT.iam.gserviceaccount.com
for r in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding PROJECT --member=serviceAccount:$SA --role=$r
done
gcloud iam service-accounts keys create key.json --iam-account=$SA   # paste into GitHub secret, then DELETE key.json
```
GitHub repo -> Settings -> Secrets and variables -> Actions:
- **Secret** `GCP_SA_KEY`: contents of `key.json`.
- **Variables** `GCP_PROJECT_ID` and `GCP_REGION` (e.g. `us-central1`).

After the first deploy, set the LLM key once (the workflow never touches it):
```bash
gcloud run services update notebooklm-clone --region us-central1 --update-env-vars GROQ_API_KEY=...
```

## Project layout
```
app.py            entry point        ui/          Gradio wiring
Dockerfile        container image    .github/     CI/CD workflow
storage/          notebook storage   ingestion/   extract, chunk, embed, index
retrieval/        embedder, Chroma, reranker      generation/  chat, artifacts, LLM client
eval/             retrieval comparison            tests/       offline tests
docs/             architecture, evaluation, demo checklist
```

## Limitations
- Scanned/image-only PDFs have no extractable text and fail with a clear error (no OCR).
- Web pages that need JavaScript to render are not supported.
- Ephemeral storage on Cloud Run (see above); first request after idle takes several seconds (cold start).
